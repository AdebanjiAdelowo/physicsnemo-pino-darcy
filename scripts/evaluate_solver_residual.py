"""Evaluate trained models under the discrete operator the reference pressure solves.

    python scripts/evaluate_solver_residual.py --config local [device=mps]

Inference only, from the checkpoints in ``runs/<name>/``. Writes
``results/audit/<name>_solver_residual.json``; the study's own result files are not changed.

For every run and for the reference pressure of the test set:

- the residual under ``pino_darcy.reference.solver_residual`` with the two-valued ``coeff`` and
  forcing 1 (physical units), on interior nodes at least ``MARGIN`` nodes from the boundary,
  split into nodes near a jump of ``coeff`` and the others;
- the relative L2 error of the pressure restricted to cells where ``Kcoeff`` is locally constant
  and to the other cells (the split used by ``residual_statistics``).
"""

import argparse
import json

import h5py
import numpy as np
import torch
from physicsnemo.utils.checkpoint import load_checkpoint

from _bootstrap import ROOT
from pino_darcy import data as D
from pino_darcy import engine
from pino_darcy import physics as P
from pino_darcy import reference as R
from pino_darcy.config import load_config
from pino_darcy.model import build_fno
from pino_darcy.provenance import collect_metadata, write_json

MARGIN = 6  # nodes; scripts/verify_reference.py shows 4 is enough for the cropped grid


def residual_stats(u_phys: np.ndarray, coeff: np.ndarray, mask: np.ndarray) -> dict:
    inner = (slice(None), slice(MARGIN, -MARGIN), slice(MARGIN, -MARGIN))
    r = np.abs(R.solver_residual(R.extend_crop(u_phys), coeff))[inner]
    m = mask[inner]
    return {"abs_mean": float(r.mean()), "abs_mean_interface_nodes": float(r[m].mean()), "abs_mean_other_nodes": float(r[~m].mean())}


def split_error(pred: torch.Tensor, truth: torch.Tensor, k: torch.Tensor) -> dict:
    """Relative L2 error per sample over the cells with constant ``Kcoeff`` and over the others, then averaged."""
    mask = P.interface_mask(k)
    inner = (slice(None), slice(None), slice(P.BORDER, -P.BORDER), slice(P.BORDER, -P.BORDER))
    d, t, m = (pred - truth)[inner].double(), truth[inner].double(), mask[inner]
    out = {}
    for name, sel in (("constant_k_cells", ~m), ("other_cells", m)):
        num = (d.square() * sel).flatten(1).sum(1).sqrt()
        den = (t.square() * sel).flatten(1).sum(1).sqrt()
        out[name] = float((num / den).mean())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="local")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    out_dir, device, sets = engine.prepare_study(cfg, ROOT, write_metadata=False, with_train=False)
    with h5py.File(ROOT / cfg.data.dir / "test.hdf5") as f:
        coeff = np.asarray(f["coeff"][:, 0], dtype=np.float64)
    k, u = sets["test"]["k"], sets["test"]["u"]
    mask = R.interface_nodes(coeff)

    result = {
        "environment": collect_metadata(device),
        "study": str(out_dir.relative_to(ROOT)),
        "margin_nodes": MARGIN,
        "units": "residual in physical units (forcing f = 1); errors are relative L2 fractions",
        "reference": {"solver_residual": residual_stats(u[:, 0].double().numpy() * D.U_SCALE, coeff, mask)},
        "runs": {},
    }
    print(f"reference: solver residual {result['reference']['solver_residual']['abs_mean']:.3e}", flush=True)
    for run in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(*run)
        model = build_fno(cfg).to(device)
        load_checkpoint(str(run_dir / "checkpoints"), models=model, device=device)
        pred = engine.predict(model, k, device)
        entry = {
            "solver_residual": residual_stats(pred[:, 0].double().numpy() * D.U_SCALE, coeff, mask),
            "rel_l2_by_region": split_error(pred, u, k),
            "rel_l2_mean": float(((pred - u).flatten(1).norm(dim=1) / u.flatten(1).norm(dim=1)).mean()),
        }
        stored = json.loads((run_dir / "eval_metrics.json").read_text())["test_rel_l2"]["mean"]
        entry["stored_rel_l2_mean"] = stored
        result["runs"][run_dir.name] = entry
        s, e = entry["solver_residual"], entry["rel_l2_by_region"]
        print(
            f"{run_dir.name:24s} rel L2 {entry['rel_l2_mean']:.4f} (stored {stored:.4f}); constant-k {e['constant_k_cells']:.4f}, "
            f"other {e['other_cells']:.4f}; solver residual {s['abs_mean']:.3f} (interface {s['abs_mean_interface_nodes']:.3f}, "
            f"other {s['abs_mean_other_nodes']:.3f})",
            flush=True,
        )
    path = ROOT / "results" / "audit" / f"{cfg.name}_solver_residual.json"
    write_json(path, result)
    print(f"written {path}")


if __name__ == "__main__":
    main()
