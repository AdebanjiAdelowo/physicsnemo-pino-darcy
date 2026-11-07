"""Check the training residuals on the real 241 x 241 fields (G3, G4), on CPU.

    python scripts/check_solver_operator.py [--samples 16]

The unit tests of ``pino_darcy.discrete`` use synthetic fields up to 129 x 129. This script
evaluates the residuals exactly as training does (``physics.PhysicsTerm``, float32, normalised
units, upstream crop) on the stored reference pressure of the first training and test samples:

1. the torch operator in float64 against the NumPy re-implementation of ``solve_gwf.m``
   (``reference.solver_residual``), which reproduces the dataset to 2.5e-8;
2. the residual of the reference pressure under each training residual, relative to the
   normalised forcing: the floor that a model equal to the reference would leave in the loss;
3. the training loss of the reference (``pde_loss``, before the 1/240 x weight factor);
4. a directional-derivative check of the ``solver`` L1 loss on a real field, float64.

Runs on CPU only and needs neither a GPU nor the network. Writes
``results/audit/solver_operator_check.json``.
"""

import argparse

import h5py
import numpy as np
import torch

from _bootstrap import ROOT
from pino_darcy import data as D
from pino_darcy import discrete as Dz
from pino_darcy import physics as P
from pino_darcy import reference as R
from pino_darcy.config import load_config
from pino_darcy.provenance import collect_metadata, write_json

NAMES = ("consistent", "consistent_coeff", "solver_kcoeff", "solver")


def load(path, n):
    with h5py.File(path, "r") as f:
        coeff = np.asarray(f["coeff"][:n, 0], dtype=np.float64)
        sol = np.asarray(f["sol"][:n, 0], dtype=np.float64)
    split = D.load_split(path, n)
    split.update(D.load_full_grid(path, 0, n))
    return coeff, sol, split


def check_split(path, n: int) -> dict:
    coeff, sol, split = load(path, n)
    K, C = coeff.shape[-1], D.CROP
    inner = (slice(Dz.MARGIN, -Dz.MARGIN),) * 2

    # 1. torch float64 against NumPy, physical units, forcing 1
    diffs, scale = [], []
    for i in range(n):
        expected = R.solver_residual(R.extend_crop(sol[i, :C, :C]), coeff[i])[inner]
        got = Dz.solver_residual(torch.as_tensor(sol[i, :C, :C]), Dz.node_coefficient(torch.as_tensor(coeff[i])), 1.0).numpy()[inner]
        diffs.append(np.abs(got - expected).max())
        scale.append(np.abs(expected).max())
    agreement = {"max_abs_difference": float(max(diffs)), "max_abs_residual": float(max(scale)), "units": "physical, forcing 1"}

    # 2, 3. the reference pressure under each training residual, exactly as in training
    cpu = torch.device("cpu")
    forcing = P.RESIDUALS["consistent"]["forcing"]
    mask = torch.as_tensor(R.interface_nodes(coeff))  # [n, K-2, K-2], node grid
    floors = {}
    for name in NAMES:
        term = P.PhysicsTerm(name, cpu, split)
        abs_mean, near, away, loss = [], [], [], []
        for i in range(n):
            r = term.residual(split["u"][i : i + 1], split["k"][i : i + 1], i).detach()
            loss.append(P.pde_loss(r).item())
            a = r.abs()[0, 0]
            if name.startswith("solver"):
                a, m = a[inner], mask[i][inner]
            else:
                # cell grid: cells 1..C-2 carry the residual; the node of cell j is node j, the mask row j-1
                a, m = a[1:-1, 1:-1], mask[i][: C - 2, : C - 2]
            abs_mean.append(a.mean().item())
            near.append(a[m].mean().item())
            away.append(a[~m].mean().item())
        floors[name] = {
            "operator": P.TRAINING_RESIDUALS[name]["operator"],
            "coefficient": P.TRAINING_RESIDUALS[name]["coefficient"],
            "abs_mean_over_forcing": float(np.mean(abs_mean) / forcing),
            "near_jumps_over_forcing": float(np.mean(near) / forcing),
            "away_from_jumps_over_forcing": float(np.mean(away) / forcing),
            "pde_loss_of_reference": float(np.mean(loss)),
        }
    return {"samples": n, "grid": K, "torch_vs_numpy_float64": agreement, "reference_under_training_residuals": floors}


def directional_derivative(path) -> dict:
    """Autograd against a central difference along a random direction, float64, real coefficient.

    For the L1 training loss the two agree only up to the residual nodes whose sign changes within
    +-eps (kinks of |r|); the quadratic loss of the same linear residual has no kinks and checks
    the gradient of the operator itself.
    """
    coeff, sol, _ = load(path, 1)
    a = Dz.node_coefficient(torch.as_tensor(coeff[0] / D.K_SCALE))
    forcing = P.RESIDUALS["consistent"]["forcing"]
    g = torch.Generator().manual_seed(0)
    u = torch.as_tensor(sol[0, : D.CROP, : D.CROP] / D.U_SCALE)[None, None].clone()
    u = u + 0.01 * torch.randn(u.shape, generator=g, dtype=torch.float64)
    v = torch.randn(u.shape, generator=g, dtype=torch.float64)
    losses = {
        "l1 (training loss)": lambda w: P.pde_loss(Dz.solver_residual(w, a, forcing)),
        "quadratic": lambda w: Dz.solver_residual(w, a, forcing).pow(2).mean(),
    }
    out = {}
    for name, loss in losses.items():
        w = u.clone().requires_grad_(True)
        loss(w).backward()
        ad = (w.grad * v).sum().item()
        out[name] = {}
        for eps in (1e-4, 1e-6):
            with torch.no_grad():
                fd = ((loss(u + eps * v) - loss(u - eps * v)) / (2 * eps)).item()
            out[name][f"eps={eps:g}"] = {"autograd": ad, "finite_difference": fd, "relative_difference": abs(ad - fd) / abs(fd)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(2)  # leave the machine to the training queue
    cfg = load_config("full_mps", [])
    data_dir = ROOT / cfg.data.dir
    result = {
        "environment": collect_metadata(torch.device("cpu")),
        "forcing_normalised": P.RESIDUALS["consistent"]["forcing"],
        "margin_nodes": Dz.MARGIN,
        "notes": [
            "Residuals of the stored reference pressure (float32 in the dataset), evaluated in float32 as in training.",
            "near_jumps: interior nodes (cells) where coeff is not constant on the surrounding 5 x 5 cells (reference.interface_nodes).",
            "pde_loss_of_reference is the mean |r| over the whole residual field including zeroed borders and margin, as the training loss.",
            "Evaluated on CPU; training ran on MPS, whose float32 kernels can differ in the last bits.",
        ],
        "splits": {split: check_split(data_dir / f"{split}.hdf5", args.samples) for split in ("train", "test")},
        "directional_derivative_solver_loss": directional_derivative(data_dir / "train.hdf5"),
    }
    write_json(ROOT / "results" / "audit" / "solver_operator_check.json", result)
    for split, s in result["splits"].items():
        print(f"[{split}] torch vs numpy (float64): max |diff| {s['torch_vs_numpy_float64']['max_abs_difference']:.2e}")
        for name, f in s["reference_under_training_residuals"].items():
            print(
                f"[{split}] reference under {name:16s} |r|/f {f['abs_mean_over_forcing']:.3e} "
                f"(near jumps {f['near_jumps_over_forcing']:.3e}, away {f['away_from_jumps_over_forcing']:.3e}), loss {f['pde_loss_of_reference']:.3e}"
            )
    print("directional derivative:", result["directional_derivative_solver_loss"])


if __name__ == "__main__":
    main()
