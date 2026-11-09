# ruff: noqa: E501  (markdown cells are kept as single lines)
"""Write kaggle/run_regularisation.ipynb (run from the repository root: python kaggle/build_regularisation_notebook.py).

The notebook is generated so that its cells stay plain, reviewable strings; tests/test_regularisation_notebook.py
checks the generated file. The logic of every step is in scripts/regularisation_session.py and the study scripts.
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# Regularisation controls for the PhysicsNeMo physics-informed Darcy FNO (Kaggle GPU)

Launcher for the study pre-registered in [`docs/REGULARISATION_PLAN.md`](https://github.com/AdebanjiAdelowo/physicsnemo-pino-darcy/blob/research/regularization-controls/docs/REGULARISATION_PLAN.md): data-only FNO training with AdamW weight decay and validation-based checkpoint selection, against the corrected physics residual at weight 0.1. It checks out one commit and calls the repository's scripts; it contains no model or training code. It is separate from `run_cuda.ipynb`, which is not changed.

**Modes.** One session per mode; nothing is trained unless `MODE` is set.

| `MODE` | What it does | Writes to `/kaggle/working` |
|---|---|---|
| `None` | environment, data and test-suite checks only | `environment.txt`, data check |
| `"pilot"` | the checks, restores and verifies the physics-informed inputs, and the timing pilot that fixes budget B | `regularisation_pilot.json` |
| `"train_select"` | trains the weight-decay grid (resumable), evaluates it on the **validation** split only and freezes the selection | per study `-records.zip`, `-bundle.zip`; `regularisation_archives.sha256`; validation files; `regularisation_selection.json` |
| `"test"` | only with the selection committed: the test evaluation and the analysis | test files, summaries, `regularisation_study.json` |

Between sessions, the pilot result (and budget B in `configs/reg_cuda.yaml`), then the selection, the validation files and the archive manifest are committed, and `REF` is set to the new commit. The test split is not read before the session of `MODE = "test"`.

**Inputs** (Kaggle: *Add input*, notebook outputs): always the output of `pino-darcy-cuda-minimal-416bfe6` (the verified CUDA checkpoints of the physics-informed runs); for `"test"`, and to resume an interrupted `"train_select"`, also the output of the `"train_select"` session. Every archive is checked against a tracked SHA-256 manifest before and after extraction, and no file is overwritten.

Accelerator: one NVIDIA GPU (the study is planned for a Tesla T4). Internet on (0.8 GB dataset download, package installation).
""")

code(r'''
REPO_URL = "https://github.com/AdebanjiAdelowo/physicsnemo-pino-darcy.git"
REF = "SET_TO_FULL_40_CHARACTER_COMMIT_SHA"   # output of `git rev-parse HEAD` for the commit to run
MODE = None                                   # None (checks only), "pilot", "train_select" or "test"
SESSION_BUDGET_HOURS = 9.0                    # GPU time you are willing to spend in this session; check Kaggle's current limits
INPUT_ROOT = "/kaggle/input"                  # where Kaggle mounts the attached notebook outputs
GITHUB_TOKEN_SECRET = None                    # name of a Kaggle secret holding a GitHub token; only for a private repository

GRID = ["reg_cuda_wd0", "reg_cuda_wd1e-3", "reg_cuda_wd1e-2", "reg_cuda_wd1e-1", "reg_cuda_wd1e0", "reg_cuda_wd1e1"]
EXTENSION = "reg_cuda_wd1e2"                  # trained only if the boundary rule of the plan asks for it
PINO_STUDIES = ["full_cuda", "full_cuda_more"]  # never trained here: restored and verified
MODES = (None, "pilot", "train_select", "test")

import re, time
T0 = time.time()
assert re.fullmatch(r"[0-9a-f]{40}", REF), "REF must be a full 40-character commit SHA, not a branch or tag."
assert MODE in MODES, f"MODE must be one of {MODES}"
assert all(s.startswith("reg_cuda_wd") for s in GRID + [EXTENSION]), "only the regularisation studies may be trained here"
WORK = "/kaggle/working"
REPO = f"{WORK}/physicsnemo-pino-darcy"
VENV = "/tmp/pino-venv"   # outside /kaggle/working, so not part of the output

def hours_left():
    return SESSION_BUDGET_HOURS - (time.time() - T0) / 3600
''')

md(r"""
## Hardware and checkout
""")

code(r'''
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
with open(f"{WORK}/gpu.txt", "w") as f:
    f.write(subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
''')

code(r'''
import os, json, glob, zipfile
def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def tree_is_clean():
    return not subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip()

if not os.path.exists(REPO):
    url = REPO_URL
    if GITHUB_TOKEN_SECRET:
        from kaggle_secrets import UserSecretsClient
        url = REPO_URL.replace("https://", f"https://{UserSecretsClient().get_secret(GITHUB_TOKEN_SECRET)}@")
    subprocess.run(["git", "clone", "--quiet", url, REPO], check=True)   # not echoed: the URL may hold a token
    subprocess.run(["git", "remote", "set-url", "origin", REPO_URL], cwd=REPO, check=True)
subprocess.run(["git", "checkout", "--quiet", REF], cwd=REPO, check=True)
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
assert head == REF, (head, REF)
assert tree_is_clean(), "the working tree is not clean"
print("commit", head, "| MODE", MODE)
''')

md(r"""
## Isolated environment with the versions of the physics-informed runs

The study runs in its own Python 3.12 environment, as the CUDA replication did. Kaggle sets `UV_SYSTEM_PYTHON` (which would send `uv pip install` to the system Python) and the notebook kernel exports an inline `MPLBACKEND` (which matplotlib in the environment rejects), so both are reset. `kaggle/regularisation_constraints.txt` pins PyTorch, NumPy, Warp and SymPy to the versions that trained the physics-informed checkpoints; `env-check` stops the session if any installed version, the CUDA version of PyTorch or the PhysicsNeMo commit differs from `results/full_cuda/study_metadata.json`. The complete package list is saved to `environment.txt`.
""")

code(r'''
if not os.path.exists(f"{VENV}/bin/python"):
    sh("pip install -q uv")
    sh(f"uv venv --python 3.12 {VENV}")
os.environ.pop("UV_SYSTEM_PYTHON", None)   # set by Kaggle; it would send `uv pip install` to the system Python
os.environ["MPLBACKEND"] = "Agg"           # the kernel exports the inline backend, which the environment cannot import
os.environ["VIRTUAL_ENV"] = VENV
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh(f"uv pip install -q --python {VENV}/bin/python -r requirements.txt -c kaggle/regularisation_constraints.txt", cwd=REPO)
sh("which python && python --version", cwd=REPO)
sh(f'python -c "import sys, torch; print(sys.prefix, torch.__file__); assert sys.prefix == \'{VENV}\' and torch.__file__.startswith(\'{VENV}/\')"', cwd=REPO)
sh('python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"', cwd=REPO)
sh(f"uv pip freeze --python {VENV}/bin/python > {WORK}/environment.txt", cwd=REPO)
sh("python scripts/regularisation_session.py env-check", cwd=REPO)
''')

md(r"""
## Data and tests
""")

code(r'''
sh("python scripts/prepare_data.py", cwd=REPO)
CHECK = f"{WORK}/data-check"
sh(f"python scripts/verify_data.py --out-root {CHECK}", cwd=REPO)
check = json.load(open(f"{CHECK}/results/data_check.json"))
local = json.load(open(f"{REPO}/results/data_check.json"))
assert check["dataset"]["files"] == local["dataset"]["files"], "the dataset differs from the tracked check"
print("DATASET IDENTICAL TO THE TRACKED CHECK")
''')

code(r'''
sh("python -m pytest -q", cwd=REPO)
''')

md(r"""
## Physics-informed inputs

`restore-inputs` finds the four archives of `pino-darcy-cuda-minimal-416bfe6` under `INPUT_ROOT`, checks their SHA-256 against `kaggle/regularisation_inputs.sha256`, extracts only files that do not exist, checks the SHA-256 of all 30 checkpoint files after extraction, and checks that the restored study metadata, configuration and run records are the tracked ones (`consistent` 0.1, seeds 0 to 4, 5100 steps, clean CUDA study at `416bfe6`).
""")

code(r'''
if MODE is None:
    print("MODE is None: inputs not restored.")
else:
    sh(f"python scripts/regularisation_session.py restore-inputs --input-root {INPUT_ROOT}", cwd=REPO)
    assert tree_is_clean(), "restoring modified tracked files"
''')

md(r"""
## Pilot (`MODE = "pilot"`)

Seed 0, 3 epochs each, on this GPU, with the code of this commit: `reg_cuda_wd0` (AdamW without decay, for the comparison with `full_cuda`), `reg_cuda_wd1e-2` (AdamW with decay and the selection bookkeeping) and `consistent` 0.1 as in `full_cuda`. Pre-registered rule: `t` is the mean time of epochs 2 and 3 including validation; `B = ceil(50 * t_pino / t_data)` with `t_data` from `reg_cuda_wd1e-2`. The first three epochs of the `wd0` and `consistent` runs are compared with those of `full_cuda` (comparable if they differ by at most 1e-3 relative). Nothing is evaluated on the test split.
""")

code(r'''
if MODE == "pilot":
    sh(f"python scripts/regularisation_session.py pilot --out {WORK}/regularisation_pilot.json", cwd=REPO)
    print("PILOT COMPLETE. Commit regularisation_pilot.json as results/audit/regularisation_pilot.json, set max_epochs in configs/reg_cuda.yaml to budget_B_epochs, and use the new commit as REF.")
''')

md(r"""
## Training and validation-only selection (`MODE = "train_select"`)

Needs the committed pilot (`check-pilot`: budget B in `configs/reg_cuda.yaml` equals the pilot's, and the runs were comparable). Every grid study is trained with `--resume --no-evaluate` (finished runs are skipped, a changed configuration is refused, nothing is evaluated on the test split) and archived as soon as it is finished. A session that runs out of time continues in the next one: attach this session's output; its archives are restored after their hashes are checked against its manifest. Then every model of every run, and the physics-informed runs, are evaluated on the training and validation splits, and `select` fixes the weight decay. If the boundary rule asks for the extension, it is trained, evaluated and selected once more.
""")

code(r'''
def train_and_archive(study):
    sh(f"python scripts/train.py --config {study} --resume --no-evaluate", cwd=REPO)
    sh(f"python scripts/regularisation_session.py archive {study} --out {WORK}", cwd=REPO)

if MODE == "train_select":
    sh("python scripts/regularisation_session.py check-pilot", cwd=REPO)
    assert not os.path.exists(f"{REPO}/results/audit/regularisation_selection.json"), "a selection already exists at this commit"
    earlier = glob.glob(f"{INPUT_ROOT}/**/regularisation_archives.sha256", recursive=True)
    if earlier:   # resume an interrupted session
        assert len(earlier) == 1, earlier
        sh(f"python scripts/regularisation_session.py restore-runs --input-root {INPUT_ROOT} --manifest {earlier[0]}", cwd=REPO)
        sh(f"cp {earlier[0]} {WORK}/regularisation_archives.sha256")
    pilot = json.load(open(f"{REPO}/results/audit/regularisation_pilot.json"))
    status = json.loads(subprocess.run(["python", "scripts/study_status.py", *GRID], cwd=REPO, capture_output=True, text=True, check=True).stdout)
    to_train = sum(len(st["to_train"]) for st in status.values())
    projection_h = to_train * pilot["t_data"] * pilot["budget_B_epochs"] / 3600
    print(f"{to_train} runs to train, projected {projection_h:.1f} h; {hours_left():.1f} h left of the session budget")
    for study in GRID:   # a study is archived when all its runs are finished; restored studies are skipped
        if status[study]["to_train"]:
            if pilot["t_data"] * pilot["budget_B_epochs"] * len(status[study]["to_train"]) / 3600 > hours_left():
                raise SystemExit(f"not enough session time left for {study}; attach this output and continue in a new session")
            train_and_archive(study)
''')

code(r'''
def validate(studies):
    for study in studies:
        sh(f"python scripts/evaluate_variants.py --config {study} --stage validation", cwd=REPO)

if MODE == "train_select":
    validate(GRID + PINO_STUDIES)
    sh("python scripts/regularisation_study.py select", cwd=REPO)
    selection = json.load(open(f"{REPO}/results/audit/regularisation_selection.json"))
    if selection["status"] == "extend":
        train_and_archive(EXTENSION)
        validate([EXTENSION])
        sh("python scripts/regularisation_study.py select", cwd=REPO)
        selection = json.load(open(f"{REPO}/results/audit/regularisation_selection.json"))
    assert selection["status"] == "final" and selection["test_set_read"] is False
    for path in glob.glob(f"{REPO}/results/audit/*_variants_validation.json") + [f"{REPO}/results/audit/regularisation_selection.json"]:
        sh(f"cp {path} {WORK}/")
    print(json.dumps({k: selection[k] for k in ("mean_validation_rel_l2_budget_A", "selected_weight_decay", "status")}, indent=1))
    print("SELECTION FROZEN. Commit regularisation_selection.json, the *_variants_validation.json files and regularisation_archives.sha256 to results/audit/, then run MODE = 'test' with the new commit as REF and this output attached.")
''')

md(r"""
## Test evaluation and analysis (`MODE = "test"`)

Only with the selection committed (`check-selection`: tracked, unchanged, `status: final`, `test_set_read: false`). The trained controls are restored from the `"train_select"` output after their hashes are checked against the committed manifest. Then the test evaluation of every model, the summaries of the grid studies, and the analysis.
""")

code(r'''
if MODE == "test":
    sh("python scripts/regularisation_session.py check-selection", cwd=REPO)
    sh("python scripts/regularisation_session.py check-pilot", cwd=REPO)
    sh(f"python scripts/regularisation_session.py restore-runs --input-root {INPUT_ROOT} --manifest results/audit/regularisation_archives.sha256", cwd=REPO)
    assert tree_is_clean(), "restoring modified tracked files"
    selection = json.load(open(f"{REPO}/results/audit/regularisation_selection.json"))
    studies = list(selection["studies"].values())
    for study in studies + PINO_STUDIES:
        sh(f"python scripts/evaluate_variants.py --config {study} --stage test", cwd=REPO)
    for study in studies:
        sh(f"python scripts/evaluate.py --config {study}", cwd=REPO)   # summaries of the final models; allowed after the selection
        sh(f"python scripts/results.py package runs/{study}", cwd=REPO)
        sh(f"cp runs/{study}.zip {WORK}/physicsnemo-pino-darcy-{study}-evaluated.zip", cwd=REPO)
    sh("python scripts/regularisation_study.py analyse", cwd=REPO)
    for path in glob.glob(f"{REPO}/results/audit/*_variants_test.json") + [f"{REPO}/results/audit/regularisation_study.json"]:
        sh(f"cp {path} {WORK}/")
    print("TEST EVALUATION AND ANALYSIS COMPLETE.")
''')

md(r"""
## Outputs
""")

code(r'''
print(f"Files in {WORK}:")
for path in sorted(glob.glob(f"{WORK}/*")):
    if os.path.isfile(path):
        print(f"  {path}  {os.path.getsize(path) / 2**20:.1f} MB")
print(f"Elapsed: {(time.time() - T0) / 3600:.2f} h. MODE: {MODE}.")
print("KAGGLE REGULARISATION SESSION COMPLETE.")
''')

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 4,
}
Path("kaggle/run_regularisation.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
print(f"{len(cells)} cells")
