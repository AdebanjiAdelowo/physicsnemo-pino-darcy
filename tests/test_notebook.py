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


def test_cells_parse_and_have_no_stored_output():
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == []


def test_full_study_is_off_by_default_and_guarded():
    code = _code()
    assert re.search(r"^RUN_FULL = False\b", code[0], re.M)
    cells = [c for c in code if "--config full --resume" in c]
    assert len(cells) == 1
    # the step starts only behind `is not True` and a budget comparison, never by reducing the config
    assert "if RUN_FULL is not True:" in cells[0]
    assert "> hours_left()" in cells[0] and "was not reduced" in cells[0]
    assert cells[0].index("if RUN_FULL is not True:") < cells[0].index("--config full --resume")


def test_commit_is_pinned_and_verified():
    code = _code()
    assert 'fullmatch(r"[0-9a-f]{40}", REF)' in code[0]
    text = "\n".join(code)
    assert "assert head == REF" in text and 'assert tree_is_clean(), "the working tree is not clean"' in text


def test_data_check_and_tests_run_before_any_training():
    text = "\n".join(_code())
    assert text.index("scripts/verify_data.py") < text.index("pytest") < text.index("scripts/train.py")


def test_archive_names():
    text = "\n".join(_code())
    for name in ("data-check", "full", "full-eval-bundle"):
        assert f"physicsnemo-pino-darcy-{name}.zip" in text


def test_referenced_scripts_and_configs_exist():
    text = "\n".join(_code())
    for script in set(re.findall(r"scripts/\w+\.py", text)):
        assert (REPO_ROOT / script).exists(), script
    for name in set(re.findall(r"--config (\w+)", text)):
        assert (CONFIG_DIR / f"{name}.yaml").exists(), name


def test_projection_matches_the_full_config():
    cfg = load_config("full")
    runs = engine.study_runs(cfg)
    text = "\n".join(_code())
    n_data_only = sum(1 for r in runs if r[1] == 0)
    assert f"n_data_only, n_physics = {n_data_only}, {len(runs) - n_data_only}" in text
    assert f"* {cfg.max_epochs} / 3600" in text
    assert "consistent" in cfg.study.residuals and 0.1 in cfg.study.physics_weights
