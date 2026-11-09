"""Reject tampered inputs before the evaluator prepares datasets or loads a model."""

import json
import subprocess
import sys
from hashlib import sha256

import pytest
from omegaconf import OmegaConf

from pino_darcy import freeze
from pino_darcy.config import REPO_ROOT, load_config
from pino_darcy.selection import SELECTION_FILE

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate_variants  # noqa: E402


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    cfg = load_config("reg_cuda_wd1e-3")
    train_commit = "a" * 40
    manifest = {"files": {"test.hdf5": sha256(b"synthetic fixture").hexdigest()}}
    checkpoint = f"{cfg.output_dir}/dataonly_seed0/snapshots/vb_cap050.pt"
    metadata = f"{cfg.output_dir}/study_metadata.json"
    archive = "results/audit/regularisation_archives.sha256"

    def write(name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return sha256(data).hexdigest()

    def encode(obj):
        return json.dumps(obj).encode()

    files = {
        SELECTION_FILE: write(SELECTION_FILE, (REPO_ROOT / SELECTION_FILE).read_bytes()),
        freeze.DATA_CHECK: write(freeze.DATA_CHECK, encode({"dataset": manifest})),
        archive: write(archive, b"synthetic archive manifest\n"),
    }
    restored = {
        metadata: write(metadata, encode({"git_commit": train_commit, "dataset": manifest, "config": OmegaConf.to_container(cfg, resolve=True)})),
        checkpoint: write(checkpoint, b"verified checkpoint"),
    }
    write(f"{cfg.data.dir}/manifest.json", encode(manifest))
    write(f"{cfg.data.dir}/test.hdf5", b"synthetic fixture")
    receipt = {"version": 1, "training_commit": train_commit, "files": files, "restored": restored}
    monkeypatch.setattr(freeze, "RECEIPT_SHA256", write(freeze.RECEIPT, encode(receipt)))
    git(root, "init", "-q")
    commit(root)
    return root, cfg, checkpoint, archive


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def commit(root):
    git(root, "add", "results")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "Freeze fixture")


def test_committed_frozen_selection_and_exact_restore_pass_the_gate(frozen):
    root, cfg, _, _ = frozen
    assert freeze.check_restore(cfg, root) == sha256((root / SELECTION_FILE).read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["weight_decay", "epoch", "schema", "receipt", "manifest", "checkpoint", "extra_checkpoint", "config", "dataset"])
def test_evaluator_rejects_tampering_before_data_preparation(frozen, monkeypatch, change):
    root, cfg, checkpoint, archive = frozen
    path = root / SELECTION_FILE
    selection = json.loads(path.read_bytes())
    if change == "weight_decay":
        selection["selected_weight_decay"] = 0.01
    elif change == "epoch":
        selection["checkpoint_selections"]["0.001"]["vb_cap050"]["selected_epoch"]["0"] = 49
    elif change == "schema":
        selection = {"status": "final", "test_set_read": False}
    if change in ("weight_decay", "epoch", "schema"):
        path.write_text(json.dumps(selection))
        commit(root)  # a commit alone does not authenticate arbitrary choices
    elif change == "receipt":
        (root / freeze.RECEIPT).write_text("{}")
        commit(root)
    elif change == "manifest":
        (root / archive).write_text("tampered manifest")
    elif change == "checkpoint":
        (root / checkpoint).write_bytes(b"tampered weights")
    elif change == "extra_checkpoint":
        (root / checkpoint).with_name("extra.pt").write_bytes(b"unverified weights")
    elif change == "config":
        cfg.gamma *= 0.99
    elif change == "dataset":
        (root / cfg.data.dir / "test.hdf5").write_bytes(b"wrong synthetic split")

    def fail(*args, **kwargs):
        pytest.fail("the evaluator prepared datasets before verifying the freeze")

    monkeypatch.setattr(evaluate_variants.engine, "prepare_study", fail)
    with pytest.raises(SystemExit, match="frozen|checkpoint inventory"):
        evaluate_variants.evaluate(cfg, "test", root=root)


def test_staged_selection_is_not_a_committed_freeze(frozen):
    root, cfg, _, _ = frozen
    git(root, "rm", "--cached", "-q", SELECTION_FILE)
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "Remove selection")
    git(root, "add", SELECTION_FILE)
    with pytest.raises(SystemExit, match="must be committed at HEAD"):
        freeze.check_restore(cfg, root)
