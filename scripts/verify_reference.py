"""Check which discrete problem the reference pressure of the dataset solves.

    python scripts/verify_reference.py

Reads the prepared splits and writes ``results/audit/reference_check.json``. Nothing that
``scripts/verify_data.py`` wrote is changed.

1. Regeneration: ``pino_darcy.reference.solve_reference`` (a re-implementation of the FNO
   data-generation script ``solve_gwf.m``) applied to ``coeff`` and to ``Kcoeff`` (the model
   input), compared with the stored ``sol``.
2. Residual of ``sol`` under the discrete operator of that script, by cell type.
3. The same residual after cropping to 240 x 240 and extending back, as a function of the
   distance to the extended edge: the margin needed to evaluate cropped model outputs.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from _bootstrap import ROOT
from pino_darcy import data as D
from pino_darcy import reference as R
from pino_darcy.config import load_config
from pino_darcy.provenance import collect_metadata, write_json

MARGINS = (2, 4, 6, 8, 10, 12, 16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "reference_check.json")
    args = parser.parse_args()
    cfg = load_config("official")
    manifest = D.prepare(cfg.data, ROOT)
    data_dir = ROOT / cfg.data.dir

    raw = {key: [] for key in ("coeff", "Kcoeff", "sol")}
    for split in ("train", "validation", "test"):
        with h5py.File(data_dir / f"{split}.hdf5") as f:
            for key in raw:
                raw[key].append(np.asarray(f[key][:, 0], dtype=np.float64))
    coeff, kcoeff, sol = (np.concatenate(raw[key]) for key in ("coeff", "Kcoeff", "sol"))
    norm = np.linalg.norm(sol.reshape(len(sol), -1), axis=1)

    regen = {}
    for name, field in (("coeff", coeff), ("Kcoeff", kcoeff)):
        err = np.array([np.linalg.norm(R.solve_reference(field[s]) - sol[s]) for s in range(len(sol))]) / norm
        regen[name] = {"rel_l2_mean": float(err.mean()), "rel_l2_max": float(err.max()), "rel_l2_min": float(err.min())}
        print(f"solve_reference({name}) against sol: relative L2 mean {err.mean():.3e}, max {err.max():.3e}")

    mask = R.interface_nodes(coeff)
    residual = {}
    for name, field in (("coeff", coeff), ("Kcoeff", kcoeff)):
        r = np.abs(R.solver_residual(sol, field))
        residual[name] = {
            "abs_mean": float(r.mean()),
            "abs_max": float(r.max()),
            "abs_mean_interface_nodes": float(r[mask].mean()),
            "abs_mean_other_nodes": float(r[~mask].mean()),
        }
        print(f"solver residual of sol with {name}: mean {r.mean():.3e}, interface nodes {r[mask].mean():.3e}, other {r[~mask].mean():.3e}")

    # cropped to the 240 x 240 grid of the model and extended back
    r = np.abs(R.solver_residual(R.extend_crop(sol[:, :-1, :-1]), coeff))
    margins = {}
    for m in MARGINS:
        inner = r[:, m:-m, m:-m]
        margins[str(m)] = {"abs_mean": float(inner.mean()), "abs_max": float(inner.max())}
        print(f"cropped and extended, margin {m:2d} nodes: mean {inner.mean():.3e}, max {inner.max():.3e}")

    out = {
        "environment": collect_metadata(torch.device("cpu")),
        "dataset": manifest,
        "samples": len(sol),
        "solver": "pino_darcy.reference: re-implementation of solve_gwf.m (FNO data generation); "
        "coefficient spline-interpolated from cell centres (i+1/2)/241 to nodes i/240, conservative five-point "
        "scheme with arithmetic face means, h = 1/240, p = 0 on boundary nodes, solution spline-interpolated back.",
        "units": "physical: forcing f = 1; multiply by 1/(K_SCALE*U_SCALE) = 57.21 for the normalised units of the training residual",
        "regeneration_from": regen,
        "solver_residual_of_reference": residual,
        "interface_node_fraction": float(mask.mean()),
        "cropped_extended_residual_by_margin": margins,
    }
    write_json(args.out, out)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
