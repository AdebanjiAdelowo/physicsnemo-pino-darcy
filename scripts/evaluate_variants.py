"""Evaluate every saved model of every run of a study, in two stages that keep the test set closed.

    python scripts/evaluate_variants.py --config reg_cuda_wd1e-2 --stage validation
    python scripts/regularisation_study.py select                       # freezes the selection
    python scripts/evaluate_variants.py --config reg_cuda_wd1e-2 --stage test

A run's *variants* are its final checkpoint (``final``, epoch ``max_epochs``) and its snapshots
(``train_metrics.json`` lists them: ``epoch050``, the fixed-step model, and ``es_cap050``,
``es_cap051``, the models early stopping selects with each cap). Studies without snapshots, such
as ``full_cuda``, have the ``final`` variant only.

- ``--stage validation`` writes ``results/audit/<name>_variants_validation.json``: relative L2
  error on the training and validation splits. Nothing on the test split is computed.
- ``--stage test`` writes ``results/audit/<name>_variants_test.json``: test relative L2 (mean and
  per sample), error on constant-``Kcoeff`` cells and near jumps, and the residual under the
  operator the reference solves (as ``scripts/evaluate_solver_residual.py``). It refuses to run
  without the committed freeze and verified restored study files.

Inference only; the study's own files are not changed.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from physicsnemo.utils.checkpoint import load_checkpoint

from _bootstrap import ROOT
from evaluate_solver_residual import MARGIN, residual_stats, split_error
from pino_darcy import data as D
from pino_darcy import engine
from pino_darcy import freeze
from pino_darcy import reference as R
from pino_darcy import selection as S
from pino_darcy.config import load_config
from pino_darcy.gpulock import gpu_lock
from pino_darcy.metrics import relative_l2
from pino_darcy.model import build_fno
from pino_darcy.provenance import collect_metadata, write_json

def variants(run_dir: Path) -> dict:
    """``{label: record}`` for the final checkpoint and every snapshot of a run."""
    train = json.loads((run_dir / "train_metrics.json").read_text())
    out = {"final": {"kind": "final", "optimiser_steps": train["optimiser_steps"], "train_seconds": train["train_seconds"]}}
    for label, record in (train.get("snapshots") or {}).items():
        out[label] = dict(record)
    return out


def load_variant(cfg, run_dir: Path, label: str, device: torch.device) -> torch.nn.Module:
    model = build_fno(cfg).to(device)
    if label == "final":
        epoch = load_checkpoint(str(run_dir / "checkpoints"), models=model, device=device)
        if epoch != cfg.max_epochs:
            raise RuntimeError(f"{run_dir}: checkpoint at epoch {epoch}, expected {cfg.max_epochs}")
    else:
        model.load_state_dict(torch.load(run_dir / "snapshots" / f"{label}.pt", map_location=device, weights_only=True))
    model.eval()
    return model


def evaluate(cfg, stage: str, root: Path = ROOT, selection: Path | None = None, out: Path | None = None) -> dict:
    if stage == "test":
        if selection is not None and Path(selection).absolute() != Path(root).resolve() / S.SELECTION_FILE:
            raise SystemExit("Only the canonical frozen selection is accepted before any test evaluation")
        selection_digest = freeze.check_restore(cfg, root)
    out_dir, device, sets = engine.prepare_study(
        cfg, root, write_metadata=False, with_train=(stage == "validation"), with_test=(stage == "test"),
    )
    result = {"environment": collect_metadata(device), "study": cfg.name, "stage": stage, "runs": {}}
    if stage == "test":
        result["selection_file_sha256"] = selection_digest
        with h5py.File(root / cfg.data.dir / "test.hdf5") as f:
            coeff = np.asarray(f["coeff"][:, 0], dtype=np.float64)
        mask = R.interface_nodes(coeff)
        result["margin_nodes"] = MARGIN
    for run in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(*run)
        entries = {}
        for label, record in variants(run_dir).items():
            model = load_variant(cfg, run_dir, label, device)
            entry = dict(record)
            splits = ("train", "validation") if stage == "validation" else ("test",)
            for split in splits:
                pred = engine.predict(model, sets[split]["k"], device)
                rel = relative_l2(pred, sets[split]["u"])
                entry[f"{split}_rel_l2"] = float(rel.mean())
                if split == "test":
                    entry["test_rel_l2_per_sample"] = rel.tolist()
                    entry["test_rel_l2_by_region"] = split_error(pred, sets["test"]["u"], sets["test"]["k"])
                    entry["test_solver_residual"] = residual_stats(pred[:, 0].double().numpy() * D.U_SCALE, coeff, mask)
            entries[label] = entry
            shown = {k: round(v, 5) for k, v in entry.items() if k.endswith("_rel_l2")}
            print(f"[{stage}] {run_dir.name} {label}: {shown}", flush=True)
        result["runs"][run_dir.name] = entries
    path = Path(out) if out else root / "results" / "audit" / f"{cfg.name}_variants_{stage}.json"
    write_json(path, result)
    print(f"written {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("validation", "test"))
    parser.add_argument("--selection", type=Path, help="optional path to the canonical committed selection")
    parser.add_argument("--out", type=Path, default=None, help="default results/audit/<name>_variants_<stage>.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    with gpu_lock(cfg.device):
        evaluate(cfg, args.stage, selection=args.selection, out=args.out)


if __name__ == "__main__":
    main()
