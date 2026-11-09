"""The regularisation-study session: test-split gates, archives, restores, pilot rule and the Kaggle launcher.

Offline, CPU, synthetic data. Tests that need the local copies of the CUDA archives skip without them.
"""

import ast
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pino_darcy import engine, study
from pino_darcy import selection as S
from pino_darcy.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate_generalisation  # noqa: E402
import evaluate_solver_residual  # noqa: E402
import regularisation_session as RS  # noqa: E402
import results as RES  # noqa: E402

NOTEBOOK = REPO_ROOT / "kaggle" / "run_regularisation.ipynb"
LOCAL_ARCHIVES = REPO_ROOT / "runs" / "kaggle" / "pino-darcy-cuda-minimal-416bfe6"


def write_selection(root: Path, **changes) -> Path:
    path = root / S.SELECTION_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "final", "test_set_read": False, **changes}))
    return path


# --- the test split stays closed until the selection is final ---------------------------------------


@pytest.fixture
def gated(tiny_cfg, sets, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "load_datasets", lambda cfg, root, with_train=True, with_test=True: sets)
    cfg = OmegaConf.merge(tiny_cfg, {"evaluation": {"test_after_selection": True}, "study": {"physics_weights": [0]}})
    study.train_study(cfg, tmp_path)  # training never reads the test split
    return cfg, tmp_path


def test_no_test_evaluation_before_a_final_selection(gated):
    cfg, root = gated
    for attempt in (lambda: study.evaluate_study(cfg, root), lambda: study.verify_study(cfg, root)):
        with pytest.raises(SystemExit, match="does not exist"):
            attempt()
    write_selection(root)
    with pytest.raises(SystemExit, match="regularisation_freeze_v1.json"):
        study.evaluate_study(cfg, root)


def test_audit_scripts_are_gated_too(monkeypatch, tmp_path):
    cfg = load_config("reg_cuda_wd0")
    monkeypatch.setattr(evaluate_solver_residual, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="does not exist"):
        evaluate_solver_residual.evaluate(cfg)
    monkeypatch.setattr(evaluate_generalisation, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["evaluate_generalisation.py", "--config", "reg_cuda_wd0"])
    with pytest.raises(SystemExit, match="does not exist"):
        evaluate_generalisation.main()


def test_studies_without_the_flag_are_not_gated(tiny_cfg, tmp_path):
    S.guard_test_evaluation(tiny_cfg, tmp_path)  # no selection file, no flag: nothing raised
    for name in ("full_cuda", "full_mps", "local"):
        S.guard_test_evaluation(load_config(name), tmp_path)


# --- archives and restores ------------------------------------------------------------------------


def test_bundle_carries_snapshots_and_records_need_no_test_results(gated, tmp_path):
    cfg, root = gated
    run = root / cfg.output_dir / "dataonly_seed0"
    (run / "snapshots").mkdir()
    (run / "snapshots" / "es_cap002.pt").write_bytes(b"x")
    with pytest.raises(SystemExit, match="no summary.json"):
        RES.package(root / cfg.output_dir)
    with zipfile.ZipFile(RES.package(root / cfg.output_dir, records_only=True)) as z:
        names = z.namelist()
    assert any(n.endswith("history.csv") for n in names) and not any("snapshots" in n or "checkpoints" in n for n in names)
    with zipfile.ZipFile(RES.bundle(root / cfg.output_dir)) as z:
        names = z.namelist()
    assert any(n.endswith("snapshots/es_cap002.pt") for n in names) and any(n.endswith(".mdlus") for n in names)


def _archive(path: Path, members: dict) -> str:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_restore_verifies_hashes_and_never_overwrites(tmp_path):
    inp, root = tmp_path / "input", tmp_path / "root"
    (inp / "nested").mkdir(parents=True)
    digest = _archive(inp / "nested" / "physicsnemo-pino-darcy-x-bundle.zip", {"runs/x/a.pt": b"new"})
    manifest = tmp_path / "m.sha256"
    manifest.write_text(f"archive {digest} physicsnemo-pino-darcy-x-bundle.zip\n")
    (root / "runs" / "x").mkdir(parents=True)
    existing = root / "runs" / "x" / "a.pt"
    existing.write_bytes(b"kept")
    with pytest.raises(SystemExit, match="existing file differs"):
        RS.restore_runs(inp, manifest, root=root)
    assert existing.read_bytes() == b"kept"
    existing.write_bytes(b"new")
    assert RS.restore_runs(inp, manifest, root=root) == {"physicsnemo-pino-darcy-x-bundle.zip": 0}
    manifest.write_text(f"archive {'0' * 64} physicsnemo-pino-darcy-x-bundle.zip\n")
    with pytest.raises(SystemExit, match="SHA-256 differs"):
        RS.restore_runs(inp, manifest, root=root)


def test_unsafe_archive_members_are_refused(tmp_path):
    _archive(tmp_path / "bad.zip", {"../escape.txt": b"x"})
    with pytest.raises(SystemExit, match="unsafe member"):
        RS.extract_new(tmp_path / "bad.zip", tmp_path / "target")


def test_inputs_manifest_lists_every_physics_informed_checkpoint():
    rows = RS.read_manifest(REPO_ROOT / RS.INPUTS)
    archives = {n for k, _, n in rows if k == "archive"}
    assert archives == {f"physicsnemo-pino-darcy-{s}{x}.zip" for s in ("full_cuda", "full_cuda_more") for x in ("", "-eval-bundle")}
    checkpoints = {n for k, _, n in rows if k == "checkpoint"}
    for s, study_name in [(s, "full_cuda" if s < 3 else "full_cuda_more") for s in range(5)]:
        assert f"runs/{study_name}/consistent_w0.1_seed{s}/checkpoints/FNO.0.50.mdlus" in checkpoints
    assert len(checkpoints) == 30 and all(re.fullmatch(r"[0-9a-f]{64}", d) for _, d, _ in rows)


@pytest.mark.skipif(not LOCAL_ARCHIVES.exists(), reason="local copies of the CUDA session archives not present")
def test_inputs_manifest_matches_the_preserved_archives():
    sums = dict(line.split()[::-1] for line in (LOCAL_ARCHIVES / "SHA256SUMS").read_text().splitlines())
    for kind, digest, name in RS.read_manifest(REPO_ROOT / RS.INPUTS):
        if kind == "archive":
            assert sums[name] == digest, name


def test_restored_metadata_must_be_the_tracked_one(tmp_path):
    (tmp_path / "results").symlink_to(REPO_ROOT / "results")
    for s in ("full_cuda", "full_cuda_more"):
        (tmp_path / "runs" / s).mkdir(parents=True)
        meta = json.loads((REPO_ROOT / "results" / s / "study_metadata.json").read_text())
        (tmp_path / "runs" / s / "study_metadata.json").write_text(json.dumps({**meta, "git_dirty": True}))
    with pytest.raises(SystemExit, match="differs from the tracked"):
        RS.check_input_metadata(tmp_path)


# --- environment and pilot rule ----------------------------------------------------------------------


def test_environment_must_match_the_one_of_the_physics_informed_runs():
    reference = RS.reference_versions()
    assert reference == {
        "python_version": "3.12",
        "torch_version": "2.15.0+cu132",
        "cuda_version": "13.2",
        "numpy_version": "2.5.4",
        "warp_version": "1.19.0",
        "sympy_version": "1.14.0",
        "physicsnemo_commit": "b45a5c810c741e6b41f8515be24c51121f8fc21f",
    }
    assert RS.env_check(dict(reference), reference) == []
    assert RS.env_check({**reference, "torch_version": "2.16.0"}, reference) == ["torch_version: installed '2.16.0', full_cuda '2.15.0+cu132'"]
    constraints = (REPO_ROOT / "kaggle" / "regularisation_constraints.txt").read_text()
    for line in ("torch==2.15.0", "numpy==2.5.4", "warp-lang==1.19.0", "sympy==1.14.0"):
        assert line in constraints.splitlines()


@pytest.mark.parametrize("t_pino, t_data, expected", [(193.5, 190.0, 51), (190.0, 190.0, 50), (3.8, 3.8 * 1.02, 50), (7.0, 6.0, 59), (3.87, 3.8, 51)])
def test_budget_rule(t_pino, t_data, expected):
    assert RS.budget_epochs(t_pino, t_data) == expected


def test_epoch_times_and_history_difference(tmp_path):
    header = "epoch,learning_rate,train_loss_data,train_loss_pde,train_loss_total,validation_mse,validation_rel_l2,a,b,elapsed_seconds\n"
    (tmp_path / "a.csv").write_text(header + "1,0,1.0,,1.0,0,0.5,0,0,4.0\n2,0,0.5,,0.5,0,0.4,0,0,7.5\n3,0,0.4,,0.4,0,0.3,0,0,11.0\n")
    (tmp_path / "b.csv").write_text(header + "1,0,1.0,,1.0,0,0.5,0,0,5.0\n2,0,0.5005,,0.5,0,0.4,0,0,9.0\n3,0,0.4,,0.4,0,0.3003,0,0,13.0\n")
    assert RS.epoch_times(tmp_path / "a.csv") == pytest.approx([4.0, 3.5, 3.5])
    diff = RS.history_difference(tmp_path / "a.csv", tmp_path / "b.csv", 3)
    assert diff["train_loss_data"] == pytest.approx(0.0005 / 0.5005) and diff["validation_rel_l2"] == pytest.approx(0.0003 / 0.3003)


def test_pilot_runs_are_the_pre_registered_ones():
    assert set(RS.PILOT_RUNS) == {"data_wd0", "data_wd1e-2", "pino"}
    assert RS.PILOT_RUNS["data_wd1e-2"][0] == "reg_cuda_wd1e-2" and RS.PILOT_RUNS["pino"][0] == "full_cuda"
    assert "study.physics_weights=[0.1]" in RS.PILOT_RUNS["pino"][2] and "study.residuals=[consistent]" in RS.PILOT_RUNS["pino"][2]
    assert all(o[0].startswith("name=reg_pilot_") for _, _, o in RS.PILOT_RUNS.values())


def test_check_pilot_needs_the_committed_pilot_and_matching_budget(tmp_path):
    with pytest.raises(SystemExit, match="not tracked"):
        RS.check_pilot(tmp_path)
    configured = int(load_config("reg_cuda").max_epochs)
    (tmp_path / "results" / "audit").mkdir(parents=True)
    path = tmp_path / RS.PILOT
    path.write_text(json.dumps({"budget_B_epochs": configured + 1, "ratio": 1.0, "comparability": {"comparable": True}}))
    with pytest.raises(SystemExit, match="budget B"):
        RS.check_pilot(tmp_path)
    path.write_text(json.dumps({"budget_B_epochs": configured, "ratio": 1.0, "comparability": {"comparable": False}}))
    with pytest.raises(SystemExit, match="comparable"):
        RS.check_pilot(tmp_path)
    path.write_text(json.dumps({"budget_B_epochs": configured, "ratio": 1.0, "comparability": {"comparable": True}}))
    assert RS.check_pilot(tmp_path)["budget_B_epochs"] == configured


@pytest.mark.skipif(not (REPO_ROOT / RS.PILOT).exists(), reason="the pilot has not been run yet")
def test_configured_budget_is_the_pilots():
    RS.check_pilot(REPO_ROOT)


# --- the Kaggle launcher ---------------------------------------------------------------------------------


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_launcher_is_generated_from_its_builder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "kaggle").mkdir()
    exec(compile((REPO_ROOT / "kaggle" / "build_regularisation_notebook.py").read_text(), "build", "exec"), {"__name__": "__main__"})
    assert (tmp_path / "kaggle" / "run_regularisation.ipynb").read_text() == NOTEBOOK.read_text()


def test_launcher_defaults_and_pinning():
    text = _code()
    assert re.search(r'^REF = "SET_TO_FULL_40_CHARACTER_COMMIT_SHA"', text, re.M) and re.search(r"^MODE = None\b", text, re.M)
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in text and "assert head == REF" in text and 'assert tree_is_clean(), "the working tree is not clean"' in text
    assert 'os.environ.pop("UV_SYSTEM_PYTHON", None)' in text and 'os.environ["MPLBACKEND"] = "Agg"' in text
    assert "-r requirements.txt -c kaggle/regularisation_constraints.txt" in text and "regularisation_session.py env-check" in text
    assert "uv pip freeze" in text and "gpu.txt" in text


def test_launcher_trains_only_the_controls_resumably_and_never_tests_before_the_selection():
    text = _code()
    trains = re.findall(r"scripts/train\.py --config [^\"']*", text)
    assert trains == ["scripts/train.py --config {study} --resume --no-evaluate"]
    assert 'assert all(s.startswith("reg_cuda_wd") for s in GRID + [EXTENSION])' in text
    assert "full_mps" not in text and "results.py publish" not in text
    order = [
        "regularisation_session.py env-check",
        "scripts/verify_data.py --exclude-test",
        "python -m pytest -q",
        "regularisation_session.py restore-inputs",
        "regularisation_session.py pilot",
        "regularisation_session.py check-pilot",
        "--stage validation",
        "regularisation_study.py select",
        "regularisation_session.py check-selection",
        "regularisation_session.py check-restore",
        "--stage test",
        "scripts/evaluate.py --config {study}",
        "regularisation_study.py analyse",
    ]
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions)
    # the test stage and the evaluation of the final models are only in the MODE == "test" cell, after check-selection
    test_cell = text[text.index('if MODE == "test":') :]
    assert "--stage test" not in text[: text.index('if MODE == "test":')]
    assert test_cell.index("check-selection") < test_cell.index("--stage test")


def test_launcher_does_not_touch_the_replication_launcher():
    assert "run_cuda" not in _code()
    assert (REPO_ROOT / "kaggle" / "run_cuda.ipynb").exists()


def test_launcher_scripts_and_configs_exist():
    text = _code()
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    tree = ast.parse(text)
    grid = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "GRID")
    for name in grid + ["reg_cuda_wd1e2", "full_cuda", "full_cuda_more"]:
        assert (REPO_ROOT / "configs" / f"{name}.yaml").exists(), name


# --- amendment and train_select session ---------------------------------------------------------------


def test_historical_results_are_unchanged():
    """Every result and figure file tracked at 47c85a9 (MPS and CUDA studies) is byte-identical."""
    manifest = REPO_ROOT / "tests" / "data" / "historical_results.sha256"
    rows = [line.split(maxsplit=1) for line in manifest.read_text().splitlines() if line and not line.startswith("#")]
    assert len(rows) == 382
    for digest, path in rows:
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest, path
    listed = {path for _, path in rows}
    added = {
        str(p.relative_to(REPO_ROOT))
        for top in ("results", "figures")
        for p in (REPO_ROOT / top).rglob("*")
        if p.is_file() and p.name != ".DS_Store" and str(p.relative_to(REPO_ROOT)) not in listed
    }
    # new files may only belong to the regularisation study, never replace or sit inside an earlier study
    assert all(re.match(r"results/(audit/(regularisation_|reg_cuda_|full_cuda(_more)?_variants_)|reg_cuda_|g8_frozen_test/)", p) for p in added), added


def test_committed_pilot_is_the_kaggle_output_and_fixes_budget_b():
    pilot = json.loads((REPO_ROOT / RS.PILOT).read_text())
    assert pilot["budget_B_epochs"] == 52 == int(load_config("reg_cuda").max_epochs)
    assert pilot["environment"]["git_commit"] == "cd7ca41f6c39e572711fe87dd7a7d8f92f254a12" and pilot["environment"]["git_dirty"] is False
    assert pilot["environment"]["gpu_name"] == "Tesla T4" and pilot["comparability"]["comparable"] is True
    assert RS.budget_epochs(pilot["t_pino"], pilot["t_data"]) == 52 and pilot["ratio"] == pytest.approx(pilot["t_pino"] / pilot["t_data"])
    for run in pilot["runs"].values():  # the estimator: mean of epochs 2 and 3
        assert run["seconds_per_epoch"] == pytest.approx(sum(run["epoch_seconds"][1:3]) / 2)


def test_check_gate_requires_an_active_gate_and_no_selection(tmp_path):
    assert RS.check_gate(tmp_path) == list(RS.GATED_STUDIES)
    write_selection(tmp_path)
    with pytest.raises(SystemExit, match="exists already"):
        RS.check_gate(tmp_path)


def test_launcher_checks_budget_and_gate_and_stops_at_the_boundary():
    text = _code()
    assert re.search(r"^EXPECTED_BUDGET_B = 52\b", text, re.M) and re.search(r"^AUTO_EXTEND = False\b", text, re.M)
    train = text[text.index('if MODE == "train_select":') :]
    assert train.index("check-pilot") < train.index("check-gate") < train.index("train_and_archive(study)")
    assert 'assert pilot["budget_B_epochs"] == EXPECTED_BUDGET_B' in text
    assert 'if selection["status"] == "extend" and AUTO_EXTEND:' in text and "SELECTION NOT FINAL" in text
