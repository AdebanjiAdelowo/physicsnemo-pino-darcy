"""Static checks of the Kaggle launcher: it cannot be executed without a CUDA session."""

import ast
import json
import re

from pino_darcy import engine
from pino_darcy.config import CONFIG_DIR, REPO_ROOT, load_config

NOTEBOOK = REPO_ROOT / "kaggle" / "run_cuda.ipynb"


def _code():
    nb = json.loads(NOTEBOOK.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _plans():
    tree = ast.parse(_code()[0])
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PLANS")
    return ast.literal_eval(node.value)


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_nothing_trains_by_default_and_training_is_guarded():
    code = _code()
    assert re.search(r"^PLAN = None\b", code[0], re.M) and "RUN_FULL" not in "\n".join(code)
    cells = [c for c in code if "scripts/train.py --config {study}" in c]
    assert len(cells) == 1
    cell = cells[0]
    # training starts only after the PLAN check and the budget comparison, and never reduces a config
    assert cell.index("if PLAN is None:") < cell.index("elif projection_h > hours_left():") < cell.index("scripts/train.py --config {study}")
    assert "No configuration was reduced" in cell


def test_every_training_call_resumes_and_only_cuda_studies_are_trained():
    text = "\n".join(_code())
    for call in re.findall(r"scripts/train\.py --config [^\"']*", text):
        if "{study}" in call:
            assert "--resume" in call
        else:  # the smoke study and the probe write to their own run directories
            assert "name=smoke_cuda" in call or "name=full_probe" in call, call
    assert 'assert all(s.startswith("full_cuda") for s in STUDIES)' in text
    assert "full_mps" not in text and "labels" not in text


def test_plans_are_the_pre_registered_run_counts():
    plans = _plans()
    counts = {plan: sum(len(engine.study_runs(load_config(s))) for s in studies) for plan, studies in plans.items()}
    assert counts == {"minimal": 15, "g2": 27, "all": 33}
    for studies in plans.values():
        for name in studies:
            cfg = load_config(name)
            assert cfg.device == "cuda" and cfg.output_dir == f"runs/{name}" and name.startswith("full_cuda")
    plan_text = (REPO_ROOT / "docs" / "CUDA_REPLICATION_PLAN.md").read_text()
    assert 'PLAN = "minimal"' in plan_text and "15 runs" in plan_text and "27 runs" in plan_text and "33 runs" in plan_text


def test_commit_and_dependencies_are_pinned_and_verified():
    code = _code()
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in code[0]
    text = "\n".join(code)
    assert "assert head == REF" in text and 'assert tree_is_clean(), "the working tree is not clean"' in text
    assert "assert s['commit'] == UPSTREAM_COMMIT" in text
    assert 'assert check["dataset"]["files"] == local["dataset"]["files"]' in text


def test_order_of_the_steps():
    text = "\n".join(_code())
    order = [
        "scripts/verify_data.py",
        "pytest",
        "--config smoke",
        "name=full_probe",
        "scripts/study_status.py",
        "scripts/train.py --config {study} --resume",
        "scripts/evaluate_solver_residual.py",
        "scripts/results.py publish",
        "scripts/cuda_replication.py",
    ]
    positions = [text.index(s) for s in order]
    assert positions == sorted(positions)


def test_restore_never_overwrites_and_requires_checkpoints():
    text = "\n".join(_code())
    assert "not os.path.exists(os.path.join(target, m))" in text and "z.extractall(target, new)" in text
    assert 'assert not st["missing_checkpoint"]' in text
    assert 'assert not os.path.exists(f"{REPO}/results/{study}")' in text


def test_archive_names():
    text = "\n".join(_code())
    for name in ("data-check", "{study}", "{study}-eval-bundle", "{study}-audit"):
        assert f"physicsnemo-pino-darcy-{name}.zip" in text
    assert "cuda_replication.json" in text


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name
    for studies in _plans().values():
        for name in studies:
            assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_probe_trains_the_full_model_for_one_epoch():
    text = "\n".join(_code())
    probe = re.search(r"--config full --no-evaluate name=full_probe max_epochs=1[^\n]*\n[^\n]*", text).group(0)
    assert '"study.physics_weights=[0,0.1]"' in probe and '"study.residuals=[consistent]"' in probe
    cfg = load_config("full", ["name=full_probe", "max_epochs=1", "study.seeds=[0]", "study.residuals=[consistent]", "study.physics_weights=[0,0.1]"])
    assert {k: cfg[k] for k in ("model", "batch_size", "data")} == {k: load_config("full_cuda")[k] for k in ("model", "batch_size", "data")}
