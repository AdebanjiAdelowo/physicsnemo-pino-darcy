# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The ``Diffusion`` class is copied from examples/cfd/darcy_physics_informed/utils.py,
# and the PhysicsInformer construction, the border treatment and the L1 reduction
# from darcy_physics_informed_fno.py, of NVIDIA PhysicsNeMo, commit
# b45a5c810c741e6b41f8515be24c51121f8fc21f. Modified by Adebanji Adelowo (2026):
# the forcing term and the grid spacing are arguments selected by name
# (``RESIDUALS``), the residual is evaluated sample by sample, the PhysicsInformer
# is placed on CPU when the model is on Apple MPS, and residual statistics are added.

import torch
import torch.nn.functional as F
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from sympy import Function, Number, Symbol

from .data import K_SCALE, U_SCALE


class Diffusion(PDE):
    """Diffusion equation: ``dT/dt - div(D * grad(T)) = Q``.

    Equivalent to ``physicsnemo-sym``'s ``Diffusion`` class for the 2-D,
    steady-state case with variable diffusivity ``D`` as a SymPy Function.

    Reference: https://en.wikipedia.org/wiki/Diffusion_equation
    """

    def __init__(self, T="T", D="D", Q=0, dim=2, time=False):
        """Initialize with variable name *T*, diffusivity *D*, and source *Q*."""
        self.dim = dim
        x, y = Symbol("x"), Symbol("y")
        iv = {"x": x, "y": y}
        T_var = Function(T)(*iv.values())
        D_var = Function(D)(*iv.values()) if isinstance(D, str) else Number(D)
        Q_var = Number(Q) if isinstance(Q, (int, float)) else Q
        self.equations = {
            f"diffusion_{T}": (
                (T_var.diff(Symbol("t")) if time else 0)
                - (D_var * T_var.diff(x)).diff(x)
                - (D_var * T_var.diff(y)).diff(y)
                - Q_var
            ),
        }


# Forcing term and grid spacing of the residual  -div(k' grad u') - Q  in normalised
# variables k' = k / K_SCALE, u' = u / U_SCALE.
#   shipped:    the two expressions of darcy_physics_informed_fno.py, unchanged.
#   consistent: the values the dataset satisfies. The data solve -div(k grad u) = 1 on a
#               cell-centred 241 x 241 grid of the unit square (spacing 1/241), so the
#               normalised forcing is 1 / (K_SCALE * U_SCALE). See scripts/verify_data.py.
RESIDUALS = {
    "shipped": {"forcing": 1.0 * 4.49996e00 * 3.88433e-03, "fd_dx": 1 / 240},
    "consistent": {"forcing": 1.0 / (K_SCALE * U_SCALE), "fd_dx": 1 / 241},
}
# Upstream multiplies the PDE loss by 1 / 240 * physics_weight.
PDE_LOSS_FACTOR = 1 / 240
# Upstream sets the residual of the two outermost rows and columns to zero.
BORDER = 2


def physics_device(device: torch.device) -> torch.device:
    """Device of the residual evaluation.

    PhysicsInformer's finite-difference derivatives dispatch to NVIDIA Warp, which runs on
    CPU and CUDA. With the model on Apple MPS the residual is evaluated on CPU; autograd
    carries the gradient across the transfer.
    """
    return torch.device("cpu") if device.type == "mps" else device


def build_informer(residual: str, device: torch.device) -> PhysicsInformer:
    """The PhysicsInformer of the upstream script for one entry of ``RESIDUALS``."""
    spec = RESIDUALS[residual]
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=spec["forcing"])
    return PhysicsInformer(
        required_outputs=["diffusion_u"],
        equations=darcy,
        grad_method="finite_difference",
        device=physics_device(torch.device(device)),
        fd_dx=spec["fd_dx"],
    )


def residual_field(informer: PhysicsInformer, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Residual of one sample with the border set to zero, as in the upstream training loop.

    Parameters
    ----------
    u, k : torch.Tensor
        Normalised pressure and permeability of shape ``[1, 1, H, W]``. The finite-difference
        module of PhysicsInformer differentiates the first sample of a batch only, so larger
        batches are rejected here.

    Returns
    -------
    torch.Tensor
        Shape ``[1, 1, H, W]`` on the device of ``u``.
    """
    if u.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("the finite-difference residual is defined for a batch of one sample.")
    where = informer.device
    pde_out_arr = informer.forward({"u": u.to(where), "k": k.to(where)})["diffusion_u"]
    pde_out_arr = F.pad(pde_out_arr[:, :, BORDER:-BORDER, BORDER:-BORDER], [BORDER] * 4, "constant", 0)
    return pde_out_arr.to(u.device)


def pde_loss(residual: torch.Tensor) -> torch.Tensor:
    """Upstream's ``loss_pde``: mean absolute residual over all cells, zeroed border included."""
    return F.l1_loss(residual, torch.zeros_like(residual))


def interface_mask(k: torch.Tensor) -> torch.Tensor:
    """Cells whose five-point stencil sees more than one permeability value. Shape as ``k``."""
    same = torch.ones_like(k, dtype=torch.bool)
    for dim in (-2, -1):
        for shift in (-1, 1):
            same &= torch.roll(k, shift, dim) == k
    return ~same


def residual_statistics(residual: torch.Tensor, k: torch.Tensor) -> dict:
    """Statistics of residual fields of shape ``[N, 1, H, W]`` (border already zero).

    ``l1`` is upstream's ``loss_pde`` averaged over the samples. The other entries are taken
    over the interior cells, where the residual is defined: root mean square, signed mean,
    median and 95th percentile of the absolute value, and the mean absolute value over the
    cells with locally constant permeability (``smooth``) and over the others (``interface``).
    """
    inner = (slice(None), slice(None), slice(BORDER, -BORDER), slice(BORDER, -BORDER))
    r = residual[inner].double()
    mask = interface_mask(k)[inner]
    a = r.abs()
    flat = a.flatten()
    if flat.numel() > 2_000_000:  # torch.quantile has a size limit; a fixed stride keeps this deterministic
        flat = flat[:: flat.numel() // 2_000_000 + 1]
    return {
        "l1": residual.double().abs().mean().item(),
        "rms": r.square().mean().sqrt().item(),
        "signed_mean": r.mean().item(),
        "abs_median": flat.median().item(),
        "abs_p95": torch.quantile(flat, 0.95).item(),
        "abs_mean_smooth": a[~mask].mean().item(),
        "abs_mean_interface": a[mask].mean().item(),
        "interface_fraction": mask.double().mean().item(),
    }
