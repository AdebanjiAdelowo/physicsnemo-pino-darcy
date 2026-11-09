"""Training with the G3/G4 residuals, with unlabelled fields (G5), and with sparse validation."""

import csv
import json

import pytest
import torch

from pino_darcy import engine, study
from pino_darcy import physics as P
from pino_darcy.config import load_config


def with_full_grid(sets, n_unlabelled=0, seed=1):
    """Add uncropped two-valued and smoothed coefficients (one row and column more than the model grid)."""
    g = torch.Generator().manual_seed(seed)
    train = sets["train"]
    n, _, H, W = train["k"].shape
    coeff = torch.where(torch.rand(n + n_unlabelled, 1, H + 1, W + 1, generator=g) > 0.5, 2.667, 0.667)
    smooth = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(coeff, (1, 1, 1, 1), mode="replicate"), 3, 1)
    train["coeff_full"], train["kcoeff_full"] = coeff[:n], smooth[:n]
    train["k"] = smooth[:n, :, :H, :W].clone()
    if n_unlabelled:
        sets["unlabelled"] = {"k": smooth[n:, :, :H, :W].clone(), "coeff_full": coeff[n:], "kcoeff_full": smooth[n:]}
    return sets


@pytest.mark.parametrize("residual", ["consistent_coeff", "solver_kcoeff", "solver"])
def test_new_residuals_train_and_are_recorded(tiny_cfg, sets, tmp_path, residual):
    sets = with_full_grid(sets)
    m = engine.train_run(tiny_cfg, residual, 0.1, 0, tmp_path / residual, torch.device("cpu"), sets)
    assert m["final_train_loss_pde"] > 0 and m["residual_spec"] == P.TRAINING_RESIDUALS[residual]
    with open(tmp_path / residual / "history.csv") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        total = float(row["train_loss_data"]) + 0.1 / 240 * float(row["train_loss_pde"])
        assert float(row["train_loss_total"]) == pytest.approx(total, rel=1e-4)


def test_existing_residuals_record_no_new_keys(tiny_cfg, sets, tmp_path):
    m = engine.train_run(tiny_cfg, "consistent", 0.1, 0, tmp_path / "c", torch.device("cpu"), sets)
    assert "residual_spec" not in m and "unlabelled_samples" not in m


def test_unlabelled_pool_adds_a_pde_term_without_changing_the_labelled_order(tiny_cfg, sets, tmp_path):
    base = with_full_grid(sets)
    pooled = with_full_grid({k: (dict(v) if isinstance(v, dict) else v) for k, v in sets.items()}, n_unlabelled=3)
    a = engine.train_run(tiny_cfg, "solver", 0.1, 0, tmp_path / "a", torch.device("cpu"), base)
    b = engine.train_run(tiny_cfg, "solver", 0.1, 0, tmp_path / "b", torch.device("cpu"), pooled)
    assert "unlabelled_samples" not in a and b["unlabelled_samples"] == 3
    assert b["final_train_loss_pde_unlabelled"] > 0
    with open(tmp_path / "b" / "history.csv") as f:
        rows = list(csv.DictReader(f))
    # the recorded total includes the unlabelled term: total > data + weighted labelled PDE term
    assert all(float(r["train_loss_total"]) > float(r["train_loss_data"]) + 0.1 / 240 * float(r["train_loss_pde"]) for r in rows)
    # data-only runs ignore the pool entirely
    d0 = engine.train_run(tiny_cfg, None, 0.0, 0, tmp_path / "d0", torch.device("cpu"), base)
    d1 = engine.train_run(tiny_cfg, None, 0.0, 0, tmp_path / "d1", torch.device("cpu"), pooled)
    assert d0["final_train_loss_total"] == d1["final_train_loss_total"]


def test_validate_every_keeps_the_last_epoch(tiny_cfg, sets, tmp_path):
    tiny_cfg.max_epochs = 5
    tiny_cfg.evaluation.validate_every = 2
    engine.train_run(tiny_cfg, None, 0.0, 0, tmp_path / "r", torch.device("cpu"), sets)
    with open(tmp_path / "r" / "history.csv") as f:
        assert [int(r["epoch"]) for r in csv.DictReader(f)] == [2, 4, 5]


def test_study_validation_of_the_new_options(tiny_cfg):
    tiny_cfg.study.residuals = ["solver", "consistent_coeff"]
    engine.validate_study(tiny_cfg)
    tiny_cfg.study.residuals = ["nonsense"]
    with pytest.raises(ValueError):
        engine.validate_study(tiny_cfg)
    tiny_cfg.study.residuals = ["solver"]
    tiny_cfg.data.unlabelled_samples = 100  # 8 labelled + 100 > 102
    with pytest.raises(ValueError):
        engine.validate_study(tiny_cfg)
    tiny_cfg.data.train_samples = None
    tiny_cfg.data.unlabelled_samples = 5
    with pytest.raises(ValueError):
        engine.validate_study(tiny_cfg)


def test_load_datasets_takes_the_pool_from_the_training_split_only(tmp_path):
    """The unlabelled fields are the training samples after the labelled ones, never validation or test."""
    import h5py
    import numpy as np

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    for split, n, offset in (("train", 102, 0), ("validation", 102, 1000), ("test", 256, 2000)):
        with h5py.File(data_dir / f"{split}.hdf5", "w") as f:
            ident = (np.arange(n) + offset)[:, None, None, None] * np.ones((1, 1, 241, 241))
            for key in ("Kcoeff", "coeff", "sol"):
                f.create_dataset(key, data=(ident + rng.uniform(0, 0.1, ident.shape)).astype("float32"))
    splits = {"train": [0, 102], "validation": [102, 204], "test": [204, 460]}
    (data_dir / "manifest.json").write_text(json.dumps({"splits": splits, "files": {}}))
    cfg = load_config("official", [f"data.dir={data_dir}", "data.train_samples=25", "+data.unlabelled_samples=40", "study.residuals=[solver]"])
    sets = engine.load_datasets(cfg, tmp_path)
    ids = lambda x: np.round((x * 4.49996)[:, 0, 0, 0].numpy()).astype(int)  # noqa: E731
    assert list(ids(sets["train"]["k"])) == list(range(25))
    assert list(ids(sets["unlabelled"]["k"])) == list(range(25, 65))
    assert list(ids(sets["unlabelled"]["coeff_full"])) == list(range(25, 65))
    assert "u" not in sets["unlabelled"] and sets["train"]["coeff_full"].shape[-1] == 241


def test_full_study_with_a_new_residual(tiny_cfg, sets, tmp_path, monkeypatch):
    sets = with_full_grid(sets)
    tiny_cfg.study.residuals = ["solver"]
    monkeypatch.setattr(engine, "load_datasets", lambda cfg, root, with_train=True, with_test=True: sets)
    out_dir = study.train_study(tiny_cfg, tmp_path)
    summary = study.evaluate_study(tiny_cfg, tmp_path)
    assert [r["run"] for r in summary["runs"]] == ["dataonly_seed0", "solver_w0.1_seed0"]
    meta = json.loads((out_dir / "study_metadata.json").read_text())
    assert meta["training_residuals"] == {"solver": P.TRAINING_RESIDUALS["solver"]}
