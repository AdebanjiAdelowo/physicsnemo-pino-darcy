"""Differentiable PyTorch port of the discrete operator that the reference pressure solves.

``pino_darcy.reference`` re-implements the data generator ``solve_gwf.m`` in NumPy and
evaluates its residual for fixed fields. This module computes the same residual in torch, so
that it can be a training loss: it is linear in the pressure and differentiable with respect to
it, and runs on CPU, CUDA or Apple MPS.

For a cell-centred pressure ``u`` (the model output, cells ``0 .. K-2`` of the ``K x K`` cell grid
after the upstream crop) and a cell-centred coefficient ``c`` on all ``K x K`` cells:

1. ``u`` is extended to ``K x K`` cells by cubic extrapolation of the last row and column
   (``reference.extend_crop``);
2. it is mapped to the ``K x K`` nodes ``i / (K - 1)`` by the inverse of the node-to-cell
   not-a-knot cubic spline, ``p = M^{-1} u M^{-T}``;
3. the node coefficient is the cell-to-node spline of ``c``, ``a = S c S^T``;
4. ``r = (K - 1)^2 * sum_faces a_face (p_centre - p_neighbour) - f`` on the interior nodes, with
   arithmetic face means of ``a``: the conservative five-point scheme of ``solve_gwf.m``.

The pressure enters only through linear maps, so the gradient of any loss of ``r`` is exact.
Nodes within ``margin`` of the boundary are set to zero: the inverse spline is non-local and the
extrapolated edge makes the nodes next to it inexact (``results/audit/reference_check.json``
measures 4 nodes; 6 are used as in ``scripts/evaluate_solver_residual.py``).
"""

from functools import lru_cache

import numpy as np
import torch

from . import reference as R

MARGIN = 6


@lru_cache(maxsize=8)
def _matrices(K: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell-to-node spline ``S`` and the inverse of the node-to-cell spline ``M^{-1}``, float64."""
    to_nodes = R.spline_matrix(R.cell_centres(K), R.nodes(K))
    from_nodes_inverse = np.linalg.inv(R.spline_matrix(R.nodes(K), R.cell_centres(K)))
    return to_nodes, from_nodes_inverse


def extend_crop(u: torch.Tensor) -> torch.Tensor:
    """Torch version of ``reference.extend_crop``: ``[..., K-1, K-1]`` to ``[..., K, K]``, linear in ``u``."""
    row = 4 * u[..., -1:, :] - 6 * u[..., -2:-1, :] + 4 * u[..., -3:-2, :] - u[..., -4:-3, :]
    u = torch.cat([u, row], dim=-2)
    col = 4 * u[..., :, -1:] - 6 * u[..., :, -2:-1] + 4 * u[..., :, -3:-2] - u[..., :, -4:-3]
    return torch.cat([u, col], dim=-1)


def node_coefficient(coeff: torch.Tensor) -> torch.Tensor:
    """Spline interpolation of a cell-centred ``[..., K, K]`` coefficient to the ``K x K`` nodes."""
    S = torch.as_tensor(_matrices(coeff.shape[-1])[0], dtype=coeff.dtype, device=coeff.device)
    return S @ coeff @ S.T


def cells_to_nodes(u: torch.Tensor) -> torch.Tensor:
    """Node values whose node-to-cell spline is the cell-centred ``[..., K, K]`` field ``u``."""
    M_inv = torch.as_tensor(_matrices(u.shape[-1])[1], dtype=u.dtype, device=u.device)
    return M_inv @ u @ M_inv.T


def node_operator(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``-div_h(a grad_h p)`` on the interior nodes of ``[..., K, K]`` node fields; shape ``[..., K-2, K-2]``.

    Conservative five-point scheme with arithmetic face means and spacing ``1 / (K - 1)``, as in
    ``reference.node_matrix`` (which is this operator with ``p = 0`` on the boundary nodes).
    """
    K = p.shape[-1]
    c = a[..., 1:-1, 1:-1]
    q = p[..., 1:-1, 1:-1]
    flux = (
        (c + a[..., :-2, 1:-1]) / 2 * (q - p[..., :-2, 1:-1])
        + (c + a[..., 2:, 1:-1]) / 2 * (q - p[..., 2:, 1:-1])
        + (c + a[..., 1:-1, :-2]) / 2 * (q - p[..., 1:-1, :-2])
        + (c + a[..., 1:-1, 2:]) / 2 * (q - p[..., 1:-1, 2:])
    )
    return (K - 1) ** 2 * flux


def solver_residual(u: torch.Tensor, a: torch.Tensor, forcing: float, margin: int = MARGIN, extend: bool = True) -> torch.Tensor:
    """Residual of cell-centred pressures under the discrete operator of ``solve_gwf.m``.

    Parameters
    ----------
    u : torch.Tensor
        Cell-centred pressure, ``[..., K-1, K-1]`` (cropped, ``extend=True``) or ``[..., K, K]``.
    a : torch.Tensor
        Node coefficient ``[..., K, K]`` (``node_coefficient`` of the cell coefficient), broadcastable
        against ``u``.
    forcing : float
        Right-hand side in the units of ``a`` and ``u``.
    margin : int
        Interior nodes closer than this to the boundary are set to zero.

    Returns
    -------
    torch.Tensor
        ``[..., K-2, K-2]``, zero within ``margin``. For the reference solution it is zero up to
        rounding everywhere, including at permeability jumps.
    """
    if extend:
        u = extend_crop(u)
    if u.shape[-2:] != a.shape[-2:]:
        raise ValueError(f"pressure {tuple(u.shape)} and node coefficient {tuple(a.shape)} are on different grids.")
    r = node_operator(cells_to_nodes(u), a) - forcing
    if margin > 0:
        keep = torch.zeros(r.shape[-2:], dtype=torch.bool, device=r.device)
        keep[margin:-margin, margin:-margin] = True
        r = torch.where(keep, r, torch.zeros((), dtype=r.dtype, device=r.device))
    return r
