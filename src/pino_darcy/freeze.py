"""Integrity gate for test evaluation of the completed CUDA regularisation study."""

import json
import subprocess
from hashlib import file_digest, sha256
from pathlib import Path

from omegaconf import OmegaConf

from .selection import SELECTION_FILE

RECEIPT = "results/audit/regularisation_freeze_v1.json"
RECEIPT_SHA256 = "181fe237d4a8cbbfe3babe68f88d5b91585537f32465155f4dcc23814cda13b9"
DATA_CHECK = "results/data_check.json"


def verified_file(root: Path, name: str, digest: str) -> Path:
    path = root / name
    if not path.is_file() or path.resolve() != path:
        raise SystemExit(f"{name}: missing or non-canonical file")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != digest:
            raise SystemExit(f"{name}: SHA-256 differs from the frozen study")
    return path


def verify_committed(root: Path, name: str, digest: str) -> Path:
    path = verified_file(root, name, digest)
    committed = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root, capture_output=True)
    if committed.returncode or sha256(committed.stdout).hexdigest() != digest:
        raise SystemExit(f"{name}: exact frozen bytes must be committed at HEAD")
    return path


def check_selection(root: Path) -> tuple[dict, dict]:
    """Authenticate the complete original selection schema and decisions by their exact bytes."""
    root = Path(root).resolve()
    if not (root / SELECTION_FILE).is_file():
        raise SystemExit(f"{SELECTION_FILE} does not exist and must be frozen before any test evaluation")
    receipt = json.loads(verify_committed(root, RECEIPT, RECEIPT_SHA256).read_text())
    for name, digest in receipt["files"].items():
        verify_committed(root, name, digest)
    return json.loads((root / SELECTION_FILE).read_text()), receipt


def check_restore(cfg, root: Path) -> str:
    """Require the pinned study config, dataset and exact restored inputs before opening test tensors."""
    root = Path(root).resolve()
    selection, receipt = check_selection(root)
    historical = cfg.name in ("full_cuda", "full_cuda_more")
    if not historical and cfg.name not in selection["studies"].values():
        raise SystemExit(f"{cfg.name}: study is outside the frozen selection")
    run_root = Path("runs") / cfg.name
    members = {name: digest for name, digest in receipt["restored"].items() if name.startswith(f"{run_root}/")}
    meta_name = str(run_root / "study_metadata.json")
    meta = json.loads(verified_file(root, meta_name, members[meta_name]).read_text())
    if not historical and meta["git_commit"] != receipt["training_commit"]:
        raise SystemExit(f"{cfg.name}: training commit differs from the frozen study")
    if meta["config"] != OmegaConf.to_container(cfg, resolve=True):
        raise SystemExit(f"{cfg.name}: configuration differs from the frozen training configuration")

    dataset = json.loads((root / DATA_CHECK).read_text())["dataset"]
    data_dir = root / cfg.data.dir
    manifest = data_dir / "manifest.json"
    if meta["dataset"] != dataset or not manifest.is_file() or json.loads(manifest.read_text()) != dataset:
        raise SystemExit(f"{cfg.name}: dataset manifest differs from the frozen training dataset")
    for name, digest in members.items():
        if name != meta_name:
            verified_file(root, name, digest)
    expected = {name for name in members if Path(name).parent.name in ("checkpoints", "snapshots")}
    actual = {
        str(path.relative_to(root))
        for run in (root / run_root).iterdir() if run.is_dir()
        for folder in (run / "checkpoints", run / "snapshots") if folder.is_dir()
        for path in folder.iterdir()
    }
    if actual != expected:
        raise SystemExit(f"{cfg.name}: unexpected files in the restored checkpoint inventory")
    for name, digest in dataset["files"].items():
        verified_file(root, str((data_dir / name).relative_to(root)), digest)
    return receipt["files"][SELECTION_FILE]
