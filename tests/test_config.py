"""The configs: upstream values untouched, one factor varied, runs enumerated once."""

import hashlib

import pytest

from pino_darcy import engine
from pino_darcy.config import CONFIG_DIR, load_config

MARKER = "# ---------------------------------------------------------------------------"
# sha256 of conf/config_pino.yaml at PhysicsNeMo commit b45a5c810c741e6b41f8515be24c51121f8fc21f
UPSTREAM_SHA256 = "40bbd9aa0a50426757549c5237f7bc006a532958f1301386743fc86e7228d7ef"


def test_official_starts_with_the_unmodified_upstream_file():
    text = (CONFIG_DIR / "official.yaml").read_text()
    upstream = text.split("\n\n" + MARKER)[0]
    assert hashlib.sha256(upstream.encode()).hexdigest() == UPSTREAM_SHA256


def test_official_is_one_run_of_the_upstream_default():
    cfg = load_config("official")
    assert (cfg.start_lr, cfg.gamma, cfg.max_epochs, cfg.physics_weight, cfg.batch_size) == (0.001, 0.99948708, 50, 0.1, 1)
    assert engine.study_runs(cfg) == [("shipped", 0.1, 0)]
    assert (cfg.data.train_percent, cfg.data.validation_percent, cfg.data.train_samples) == (10, 10, None)


@pytest.mark.parametrize("name", ["smoke", "local", "full"])
def test_study_configs_keep_model_and_optimiser(name):
    official, cfg = load_config("official"), load_config(name)
    assert cfg.start_lr == official.start_lr and cfg.batch_size == official.batch_size and cfg.data == official.data or name == "smoke"
    if name != "smoke":
        assert cfg.model == official.model
    engine.validate_study(cfg)


def test_full_trains_exactly_as_official():
    official, full = load_config("official"), load_config("full")
    for key in ("start_lr", "gamma", "max_epochs", "batch_size", "model", "data"):
        assert full[key] == official[key], key
    assert ("shipped", official.physics_weight, 0) in engine.study_runs(full)
    assert full.device == "cuda"


def test_local_schedule_reaches_the_upstream_final_learning_rate():
    official, local = load_config("official"), load_config("local")
    steps = lambda cfg: cfg.max_epochs * 102  # noqa: E731
    assert local.gamma ** steps(local) == pytest.approx(official.gamma ** steps(official), rel=1e-3)


def test_runs_enumerate_data_only_once_per_seed():
    cfg = load_config("full")
    runs = engine.study_runs(cfg)
    assert len(runs) == 3 * (1 + 2 * 4)
    assert [r for r in runs if r[1] == 0] == [(None, 0.0, s) for s in (0, 1, 2)]
    names = [engine.run_name(*r) for r in runs]
    assert len(set(names)) == len(names)
    assert {"dataonly_seed0", "shipped_w0.1_seed0", "consistent_w0.001_seed2", "shipped_w1_seed1"} <= set(names)


def test_invalid_studies_are_rejected():
    for override in ("study.residuals=[unknown]", "study.physics_weights=[-1]", "batch_size=2"):
        with pytest.raises(ValueError):
            engine.validate_study(load_config("smoke", [override]))


def test_unavailable_device_is_an_error():
    import torch

    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            engine.resolve_device("cuda")
