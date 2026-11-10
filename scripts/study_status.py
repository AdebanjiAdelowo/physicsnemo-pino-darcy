"""Which runs of one or more studies are finished, as JSON (read only; used by kaggle/run_cuda.ipynb).

    python scripts/study_status.py full_cuda full_cuda_more

A run is finished when its ``train_metrics.json`` exists. For every finished run the final
checkpoint must exist as well (``*.<max_epochs>.mdlus``); otherwise the run is listed under
``missing_checkpoint``, because ``train.py --resume`` would skip it and its evaluation would fail.
"""

import argparse
import json

from _bootstrap import ROOT
from pino_darcy import engine
from pino_darcy.config import load_config


def status(name: str, root=ROOT) -> dict:
    cfg = load_config(name)
    out = root / cfg.output_dir
    runs = engine.study_runs(cfg)
    rows = {
        "study": cfg.name,
        "output_dir": cfg.output_dir,
        "device": cfg.device,
        "max_epochs": int(cfg.max_epochs),
        "finished": [],
        "to_train": [],
        "missing_checkpoint": [],
    }
    for residual, weight, seed in runs:
        run = engine.run_name(residual, weight, seed)
        entry = {"run": run, "physics_weight": float(weight), "seed": int(seed)}
        if (out / run / "train_metrics.json").exists():
            rows["finished"].append(entry)
            if not list((out / run / "checkpoints").glob(f"*.{cfg.max_epochs}.mdlus")):
                rows["missing_checkpoint"].append(run)
        else:
            rows["to_train"].append(entry)
    rows["runs"] = len(runs)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    print(json.dumps({name: status(name) for name in args.configs}))


if __name__ == "__main__":
    main()
