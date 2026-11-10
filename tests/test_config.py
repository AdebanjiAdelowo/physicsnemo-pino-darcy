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


@pytest.mark.parametrize("name", ["full_mps", "full_mps_sweep"])
def test_full_mps_trains_exactly_as_official(name):
    official, cfg = load_config("official"), load_config(name)
    for key in ("start_lr", "gamma", "max_epochs", "batch_size", "model", "data"):
        assert cfg[key] == official[key], key
    assert cfg.device == "mps"
    engine.validate_study(cfg)


def test_full_mps_stages_are_the_runs_of_full():
    stages = engine.study_runs(load_config("full_mps")) + engine.study_runs(load_config("full_mps_sweep"))
    assert len(stages) == len(set(stages)) == 27
    assert set(stages) == set(engine.study_runs(load_config("full")))


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


TRAINING_KEYS = ("start_lr", "gamma", "model", "batch_size", "device")


def test_operator_studies_share_the_training_settings_of_full_mps():
    """G3/G4 runs are paired with the data-only and consistent runs of full_mps: nothing else may differ."""
    from omegaconf import OmegaConf

    base = OmegaConf.to_container(load_config("full_mps"), resolve=True)
    for name in ("full_mps_operator", "full_mps_solver_w1"):
        cfg = OmegaConf.to_container(load_config(name), resolve=True)
        assert {k: cfg[k] for k in (*TRAINING_KEYS, "max_epochs", "data")} == {k: base[k] for k in (*TRAINING_KEYS, "max_epochs", "data")}
        assert cfg["study"]["seeds"] == [0, 1, 2]
        engine.validate_study(load_config(name))
    assert load_config("full_mps_operator").study.residuals == ["consistent_coeff", "solver_kcoeff", "solver"]


@pytest.mark.parametrize("n", [25, 50])
def test_label_budget_studies_keep_the_optimiser_budget(n):
    from omegaconf import OmegaConf

    base = OmegaConf.to_container(load_config("full_mps"), resolve=True)
    labelled, pooled = load_config(f"labels{n}"), load_config(f"labels{n}_unlabelled")
    for cfg in (labelled, pooled):
        c = OmegaConf.to_container(cfg, resolve=True)
        assert {k: c[k] for k in TRAINING_KEYS} == {k: base[k] for k in TRAINING_KEYS}
        assert cfg.max_epochs * cfg.data.train_samples == 5100 == base["max_epochs"] * 102
        assert cfg.data.train_samples == n
        engine.validate_study(cfg)
    assert labelled.data.get("unlabelled_samples", 0) == 0
    assert pooled.data.unlabelled_samples == 102 - n  # the rest of the training split, nothing else
    assert list(labelled.study.physics_weights) == [0, 0.1] and list(pooled.study.physics_weights) == [0.1]


CUDA_STAGES = {"full_cuda": "full_mps", "full_cuda_sweep": "full_mps_sweep", "full_cuda_more": "full_mps"}
MPS_STUDIES = (
    "full_mps",
    "full_mps_more",
    "full_mps_sweep",
    "full_mps_operator",
    "full_mps_solver_w1",
    "labels25",
    "labels50",
    "labels25_unlabelled",
    "labels50_unlabelled",
)


@pytest.mark.parametrize("name", sorted(CUDA_STAGES))
def test_cuda_stages_train_exactly_as_full_and_full_mps(name):
    from omegaconf import OmegaConf

    cfg = load_config(name)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    for ref in ("full", "full_mps"):
        base = OmegaConf.to_container(load_config(ref), resolve=True)
        for key in ("start_lr", "gamma", "max_epochs", "batch_size", "model", "data", "physics_weight", "evaluation"):
            assert resolved[key] == base[key], (ref, key)
    assert cfg.device == "cuda" and cfg.name == name and cfg.output_dir == f"runs/{name}"
    engine.validate_study(cfg)


def test_cuda_stages_cannot_write_to_any_mps_study():
    for name in CUDA_STAGES:
        cfg = load_config(name)
        assert cfg.name not in MPS_STUDIES and not cfg.name.startswith("full_mps") and not cfg.name.startswith("labels")
        assert cfg.output_dir.split("/")[-1] == cfg.name


def test_cuda_stages_mirror_the_mps_stages_and_cover_the_plans():
    runs = {name: engine.study_runs(load_config(name)) for name in CUDA_STAGES}
    assert runs["full_cuda"] == engine.study_runs(load_config("full_mps"))
    assert runs["full_cuda_sweep"] == engine.study_runs(load_config("full_mps_sweep"))
    assert runs["full_cuda_more"] == engine.study_runs(load_config("full_mps", ["study.seeds=[3,4]"]))
    minimal = runs["full_cuda"] + runs["full_cuda_more"]
    assert len(minimal) == len(set(minimal)) == 15
    for setting in {(r, w) for r, w, _ in minimal}:
        assert sorted(s for r, w, s in minimal if (r, w) == setting) == [0, 1, 2, 3, 4]
    g2 = runs["full_cuda"] + runs["full_cuda_sweep"]
    assert len(g2) == len(set(g2)) == 27 and set(g2) == set(engine.study_runs(load_config("full")))
    everything = g2 + runs["full_cuda_more"]
    assert len(everything) == len(set(everything)) == 33
