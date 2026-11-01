"""Whether a repeated training run gives the same model, and how far devices diverge.

    python scripts/check_determinism.py [--epochs 1]

Trains the data-only model and the ``consistent`` 0.1 model of seed 0 with the settings of
``configs/full_mps.yaml``, twice on MPS and once on CPU, for a short number of epochs, in a
temporary directory. Compares the final parameters and the validation error. Writes
``results/audit/determinism.json``. No study result is read or changed.
"""

import argparse
import tempfile
from pathlib import Path

import torch
from physicsnemo.utils.checkpoint import load_checkpoint

from _bootstrap import ROOT
from pino_darcy import engine
from pino_darcy.config import load_config
from pino_darcy.gpulock import gpu_lock
from pino_darcy.model import build_fno
from pino_darcy.provenance import collect_metadata, write_json


def train_once(cfg, residual, weight, device, sets, run_dir) -> tuple[dict, float]:
    engine.train_run(cfg, residual, weight, 0, run_dir, torch.device(device), sets)
    model = build_fno(cfg)
    load_checkpoint(str(run_dir / "checkpoints"), models=model, device="cpu")
    params = {name: p.detach().cpu().double() for name, p in model.named_parameters()}
    pred = engine.predict(model.to("cpu"), sets["validation"]["k"], torch.device("cpu"))
    return params, engine.relative_l2(pred, sets["validation"]["u"]).mean().item()


def distance(a: dict, b: dict) -> dict:
    largest = max((a[n] - b[n]).abs().max().item() for n in a)
    norm = sum((a[n] - b[n]).square().sum().item() for n in a) ** 0.5
    return {"max_abs_parameter_difference": largest, "parameter_l2_distance": norm, "identical": largest == 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config("full_mps", [f"max_epochs={args.epochs}"])
    sets = engine.load_datasets(cfg, ROOT)
    result = {"environment": collect_metadata(torch.device("mps")), "epochs": args.epochs, "seed": 0, "cases": {}}
    with gpu_lock("mps"), tempfile.TemporaryDirectory() as tmp:
        for name, residual, weight in (("data only", None, 0.0), ("consistent 0.1", "consistent", 0.1)):
            runs = {}
            for label, device in (("mps_a", "mps"), ("mps_b", "mps"), ("cpu", "cpu")):
                runs[label] = train_once(cfg, residual, weight, device, sets, Path(tmp) / f"{residual}_{label}")
            result["cases"][name] = {
                "validation_rel_l2": {label: run[1] for label, run in runs.items()},
                "mps_repeat": distance(runs["mps_a"][0], runs["mps_b"][0]),
                "mps_vs_cpu": distance(runs["mps_a"][0], runs["cpu"][0]),
            }
            print(name, result["cases"][name], flush=True)
    write_json(ROOT / "results" / "audit" / "determinism.json", result)


if __name__ == "__main__":
    main()
