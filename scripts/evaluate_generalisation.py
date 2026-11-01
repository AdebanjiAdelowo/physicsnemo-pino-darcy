"""Relative L2 error of trained models on the training, validation and test splits.

    python scripts/evaluate_generalisation.py --config full_mps

Inference only, from the checkpoints in ``runs/<name>/``. Writes
``results/audit/<name>_generalisation.json``; the study's own result files are not changed.
The test fields are unseen but drawn from the same prior as the training fields, so the gap
measures in-distribution generalisation only.
"""

import argparse

from physicsnemo.utils.checkpoint import load_checkpoint

from _bootstrap import ROOT
from pino_darcy import engine
from pino_darcy.config import load_config
from pino_darcy.gpulock import gpu_lock
from pino_darcy.metrics import relative_l2
from pino_darcy.model import build_fno
from pino_darcy.provenance import collect_metadata, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    with gpu_lock(cfg.device):
        out_dir, device, sets = engine.prepare_study(cfg, ROOT, write_metadata=False, with_train=True)
        result = {"environment": collect_metadata(device), "study": str(out_dir.relative_to(ROOT)), "runs": {}}
        for run in engine.study_runs(cfg):
            run_dir = out_dir / engine.run_name(*run)
            model = build_fno(cfg).to(device)
            load_checkpoint(str(run_dir / "checkpoints"), models=model, device=device)
            entry = {
                split: relative_l2(engine.predict(model, sets[split]["k"], device), sets[split]["u"]).mean().item()
                for split in ("train", "validation", "test")
            }
            entry["test_minus_train"] = entry["test"] - entry["train"]
            entry["test_over_train"] = entry["test"] / entry["train"]
            result["runs"][run_dir.name] = entry
            print(run_dir.name, entry, flush=True)
    write_json(ROOT / "results" / "audit" / f"{cfg.name}_generalisation.json", result)


if __name__ == "__main__":
    main()
