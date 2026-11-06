"""Whether the current training code gives the same models as an earlier commit for the existing residuals.

    python scripts/check_code_regression.py --against eaca64c

Checks out ``--against`` in a temporary git worktree, trains the data-only, ``shipped`` 0.1 and
``consistent`` 0.1 models of seed 0 on CPU (two epochs, four training samples, smoke
architecture) with both versions of the code, and compares the final parameters bit for bit.
Writes ``results/audit/code_regression.json``. CPU kernels are deterministic here; the check says
nothing about other devices.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT
from pino_darcy.provenance import collect_metadata, write_json

CHILD = r'''
import hashlib, json, sys, tempfile
from pathlib import Path
root, data_dir = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root / "src"))
import torch
torch.set_num_threads(1)
from physicsnemo.utils.checkpoint import load_checkpoint
from pino_darcy import engine
from pino_darcy.config import load_config
from pino_darcy.model import build_fno
cfg = load_config("smoke", ["device=cpu", "max_epochs=2", f"data.dir={data_dir}", "data.train_samples=4"])
sets = engine.load_datasets(cfg, root)
for split in ("validation", "test"):
    sets[split] = {k: v[:4] for k, v in sets[split].items()}
out = {}
with tempfile.TemporaryDirectory() as tmp:
    for residual, weight in ((None, 0.0), ("shipped", 0.1), ("consistent", 0.1)):
        run = Path(tmp) / str(residual)
        m = engine.train_run(cfg, residual, weight, 0, run, torch.device("cpu"), sets)
        model = build_fno(cfg)
        load_checkpoint(str(run / "checkpoints"), models=model, device="cpu")
        digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        out[engine.run_name(residual, weight, 0)] = {"parameters_sha256": digest, "final_train_loss_total": m["final_train_loss_total"]}
print("RESULT" + json.dumps(out))
'''


def run(tree: Path, data_dir: Path) -> dict:
    proc = subprocess.run([sys.executable, "-c", CHILD, str(tree), str(data_dir)], capture_output=True, text=True, check=True)
    line = next(x for x in proc.stdout.splitlines() if x.startswith("RESULT"))
    return json.loads(line[len("RESULT"):])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--against", required=True, help="earlier commit")
    args = parser.parse_args()
    data_dir = (ROOT / "datasets" / "Darcy_241").resolve()
    against = subprocess.run(["git", "rev-parse", args.against], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        tree = Path(tmp) / "earlier"
        subprocess.run(["git", "worktree", "add", "--detach", str(tree), against], cwd=ROOT, check=True, capture_output=True)
        try:
            earlier = run(tree, data_dir)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, check=True, capture_output=True)
    current = run(ROOT, data_dir)
    result = {
        "environment": collect_metadata(__import__("torch").device("cpu")),
        "against_commit": against,
        "settings": "smoke architecture, 4 training samples, 2 epochs, seed 0, CPU, one thread",
        "earlier": earlier,
        "current": current,
        "identical": {run: earlier[run]["parameters_sha256"] == current[run]["parameters_sha256"] for run in earlier},
    }
    write_json(ROOT / "results" / "audit" / "code_regression.json", result)
    print(json.dumps(result["identical"]))


if __name__ == "__main__":
    main()
