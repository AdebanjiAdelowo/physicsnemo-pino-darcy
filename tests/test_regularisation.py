"""Regularisation controls: optimiser, early stopping, snapshots, configurations and the selection protocol.

Offline, CPU, synthetic data; no checkpoint of a real study is needed.
"""

import json
import random
import sys

import pytest
import torch
from omegaconf import OmegaConf

from pino_darcy import engine, study
from pino_darcy import selection as S
from pino_darcy.config import CONFIG_DIR, REPO_ROOT, load_config
from pino_darcy.model import build_fno

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import evaluate_variants as V  # noqa: E402
import regularisation_study as RS  # noqa: E402

GRID_CONFIGS = sorted(p.stem for p in CONFIG_DIR.glob("reg_cuda_wd*.yaml"))


def reg_cfg(tiny_cfg, **changes):
    cfg = OmegaConf.merge(
        tiny_cfg,
        {
            "max_epochs": 4,
            "optimizer": {"name": "adamw", "weight_decay": 0.01},
            "snapshots": {"epochs": [2]},
            "early_stopping": {"caps": [2, 4], "rules": {"es": {"patience": 1, "min_delta": 0.0}, "vb": {"patience": None, "min_delta":0.0}}},
            "study": {"physics_weights": [0], "seeds": [0]},
        },
        changes,
    )
    OmegaConf.set_struct(cfg, False)
    return cfg


# --- optimiser -------------------------------------------------------------------------------


def test_existing_configurations_keep_the_upstream_adam():
    for name in ("official", "full", "full_mps", "full_cuda", "local", "smoke"):
        cfg = load_config(name)
        assert S.optimiser_settings(cfg) == ("adam", 0.0)
        opt = S.make_optimiser(cfg, [torch.nn.Parameter(torch.zeros(1))])
        assert type(opt) is torch.optim.Adam and opt.defaults["weight_decay"] == 0.0 and opt.defaults["betas"] == (0.9, 0.999)


def _train_steps(opt_cls, steps=5, **kwargs):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    opt = opt_cls(model.parameters(), lr=1e-3, betas=(0.9, 0.999), **kwargs)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randn(4, 6, generator=g)
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def test_adamw_without_decay_is_the_upstream_update():
    for a, b in zip(_train_steps(torch.optim.Adam, weight_decay=0.0), _train_steps(torch.optim.AdamW, weight_decay=0.0)):
        assert torch.equal(a, b)


def test_adamw_decay_is_decoupled_and_follows_the_learning_rate():
    """With a zero gradient AdamW only shrinks the weights by 1 - lr * wd; coupled L2 (Adam) moves them by about lr."""
    p0 = torch.tensor([0.5, -2.0])
    for cls, expected in ((torch.optim.AdamW, p0 * (1 - 1e-3 * 0.1)), (torch.optim.Adam, p0 - 1e-3 * torch.sign(p0))):
        p = torch.nn.Parameter(p0.clone())
        opt = cls([p], lr=1e-3, weight_decay=0.1)
        p.grad = torch.zeros_like(p)
        opt.step()
        assert torch.allclose(p.detach(), expected, rtol=0, atol=1e-6), cls


def test_invalid_optimiser_settings_are_refused(tiny_cfg):
    for bad in ({"optimizer": {"name": "sgd"}}, {"optimizer": {"name": "adamw", "weight_decay": -1.0}}):
        with pytest.raises(ValueError):
            engine.validate_study(OmegaConf.merge(tiny_cfg, bad))


# --- early stopping ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, patience, min_delta, cap, expected",
    [
        ([5, 4, 3, 2, 1], 2, 0.0, 5, {"best_epoch": 5, "stop_epoch": None, "consumed_epochs": 5}),
        ([5, 4, 4, 4, 3], 2, 0.0, 5, {"best_epoch": 2, "stop_epoch": 4, "consumed_epochs": 4}),
        ([5, 4, 4, 4, 3], 3, 0.0, 5, {"best_epoch": 5, "stop_epoch": None, "consumed_epochs": 5}),
        ([5, 4, 3.99, 3.98, 1], 2, 0.01, 5, {"best_epoch": 2, "stop_epoch": 4, "consumed_epochs": 4}),
        ([5, 4, 4, 4, 3], 2, 0.0, 3, {"best_epoch": 2, "stop_epoch": None, "consumed_epochs": 3}),
        ([1, 2, 3], 1, 0.0, 3, {"best_epoch": 1, "stop_epoch": 2, "consumed_epochs": 2}),
        ([5, 4, 4, 4, 4, 4, 4, 3, 3.5], None, 0.0, 9, {"best_epoch": 8, "stop_epoch": None, "consumed_epochs": 9}),
        ([5, 4, 4, 4, 4, 4, 4, 3, 3.5], None, 0.0, 7, {"best_epoch": 2, "stop_epoch": None, "consumed_epochs": 7}),
    ],
)
def test_early_stopping_rule(values, patience, min_delta, cap, expected):
    assert S.early_stopping_outcome([float(v) for v in values], patience, min_delta, cap) == expected


def test_tracker_selects_what_the_rule_selects():
    """The weights kept for every cap are those of the epoch the rule selects, frozen at the stop."""
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 12)
        values = [rng.choice([1.0, 0.9, 0.8, 0.85, 0.7]) for _ in range(n)]
        settings = {"patience": rng.choice([1, 2, 3, 4, None]), "min_delta": rng.choice([0.0, 0.05]), "caps": sorted({rng.randint(1, n), n})}
        tracker = S.EarlyStoppingTracker({k: settings[k] for k in ("patience", "min_delta")}, settings["caps"])
        model = torch.nn.Linear(1, 1, bias=False)
        got = {}
        for epoch, v in enumerate(values, start=1):
            with torch.no_grad():
                model.weight.fill_(epoch)  # the weights identify the epoch
            for cap, selected, stop, state in tracker.update(epoch, v, model):
                got[cap] = (selected, int(state["weight"].item()), stop if stop is not None and stop <= cap else None)
        for cap in settings["caps"]:
            rule = S.early_stopping_outcome(values, settings["patience"], settings["min_delta"], cap)
            assert got[cap] == (rule["best_epoch"], rule["best_epoch"], rule["stop_epoch"]), (values, settings, cap)


def test_early_stopping_settings_are_validated(tiny_cfg):
    base = reg_cfg(tiny_cfg)
    for bad in (
        {"early_stopping": {"rules": {"es": {"patience": 0}}}},
        {"early_stopping": {"rules": {"es": {"min_delta": -0.1}}}},
        {"early_stopping": {"caps": [5]}},
        {"evaluation": {"validate_every": 2}},
        {"snapshots": {"epochs": [9]}},
    ):
        with pytest.raises(ValueError):
            engine.validate_study(OmegaConf.merge(base, bad))


# --- training with snapshots ----------------------------------------------------------------------


@pytest.fixture
def reg_run(tiny_cfg, sets, tmp_path):
    cfg = reg_cfg(tiny_cfg)
    metrics = engine.train_run(cfg, None, 0.0, 0, tmp_path / "run", torch.device("cpu"), sets)
    return cfg, tmp_path / "run", metrics


def _state(path):
    return torch.load(path, weights_only=True)


def test_snapshots_and_records_follow_the_rule(reg_run):
    cfg, run_dir, m = reg_run
    with open(run_dir / "history.csv") as f:
        val = [float(line.split(",")[6]) for line in f.read().splitlines()[1:]]
    assert len(val) == cfg.max_epochs and m["optimizer"] == {"name": "adamw", "weight_decay": 0.01, "betas": [0.9, 0.999], "start_lr": 1e-3, "gamma": cfg.gamma}
    assert set(m["snapshots"]) == {"epoch002", "es_cap002", "es_cap004", "vb_cap002", "vb_cap004"}
    for label, record in m["snapshots"].items():
        assert (run_dir / "snapshots" / f"{label}.pt").exists()
        if record["kind"] == "early_stopping":
            rule = S.early_stopping_outcome(val, record["patience"], 0.0, record["cap"])
            if record["rule"] == "vb":
                assert record["epoch"] == 1 + min(range(record["cap"]), key=val.__getitem__) and record["stop_epoch"] is None
            assert (record["epoch"], record["stop_epoch"], record["consumed_epochs"]) == (rule["best_epoch"], rule["stop_epoch"], rule["consumed_epochs"])
            assert record["optimiser_steps"] == record["consumed_epochs"] * 4 and record["train_seconds"] <= m["train_seconds"]


def test_a_snapshot_is_the_model_of_a_shorter_run(reg_run, tiny_cfg, sets, tmp_path):
    """Training past an epoch does not change the model at that epoch: the fixed-step and early-stopping
    snapshots equal the final weights of runs that stop there."""
    cfg, run_dir, m = reg_run
    for label, record in m["snapshots"].items():
        short = reg_cfg(tiny_cfg, max_epochs=record["epoch"], snapshots={"epochs": []}, early_stopping=None)
        engine.train_run(short, None, 0.0, 0, tmp_path / label, torch.device("cpu"), sets)
        model = build_fno(short)
        from physicsnemo.utils.checkpoint import load_checkpoint

        load_checkpoint(str(tmp_path / label / "checkpoints"), models=model, device="cpu")
        snap = _state(run_dir / "snapshots" / f"{label}.pt")
        for key, value in model.state_dict().items():
            assert torch.equal(value, snap[key]), (label, key)


def test_runs_without_the_new_settings_write_the_old_records(tiny_cfg, sets, tmp_path):
    m = engine.train_run(tiny_cfg, None, 0.0, 0, tmp_path / "old", torch.device("cpu"), sets)
    assert not {"optimizer", "early_stopping", "snapshots", "snapshot_write_seconds"} & set(m)
    assert not (tmp_path / "old" / "snapshots").exists()


def test_resume_refuses_a_changed_weight_decay(tiny_cfg, sets, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "load_datasets", lambda cfg, root, with_train=True: sets)
    cfg = reg_cfg(tiny_cfg)
    study.train_study(cfg, tmp_path)
    assert study.check_resumable(cfg, tmp_path / cfg.output_dir) is False
    with pytest.raises(SystemExit, match="cannot resume"):
        study.check_resumable(OmegaConf.merge(cfg, {"optimizer": {"weight_decay": 0.1}}), tmp_path / cfg.output_dir)


# --- configurations -----------------------------------------------------------------------------------


def test_grid_configurations_train_as_full_cuda_apart_from_the_controls():
    reference = OmegaConf.to_container(load_config("full_cuda"), resolve=True)
    names, dirs = set(), set()
    for name in GRID_CONFIGS:
        cfg = load_config(name)
        engine.validate_study(cfg)
        c = OmegaConf.to_container(cfg, resolve=True)
        for key in ("model", "data", "batch_size", "start_lr", "gamma", "device"):
            assert c[key] == reference[key], (name, key)
        # the same evaluation settings, plus the gate that keeps the test split closed until the selection is final
        assert c["evaluation"] == {**reference["evaluation"], "test_after_selection": True}, name
        assert c["optimizer"]["name"] == "adamw" and c["study"]["physics_weights"] == [0] and c["study"]["seeds"] == [0, 1, 2, 3, 4]
        assert c["snapshots"]["epochs"] == [50] and c["max_epochs"] >= 50 and sorted(c["early_stopping"]["caps"]) == sorted({50, c["max_epochs"]})
        assert c["early_stopping"]["rules"] == {"es": {"patience": 5, "min_delta": 0.0}, "vb": {"patience": None, "min_delta": 0.0}}
        names.add(c["name"])
        dirs.add(c["output_dir"])
    assert len(names) == len(dirs) == len(GRID_CONFIGS)
    existing = {p.name for p in (REPO_ROOT / "results").iterdir() if p.is_dir()}
    assert not {d.split("/")[-1] for d in dirs} & existing
    by_wd = {float(load_config(n).optimizer.weight_decay): n for n in GRID_CONFIGS}
    assert by_wd == {**RS.GRID, **RS.EXTENSION}


# --- selection and analysis protocol ---------------------------------------------------------------------


def _fake_records(audit, val_by_wd, test_offset_by_wd, pino_test=0.030, seeds=RS.SEEDS):
    labels = ("final", "epoch050", "es_cap050", "es_cap051", "vb_cap050", "vb_cap051")
    for wd, study_name in {**RS.GRID, **RS.EXTENSION}.items():
        if wd not in val_by_wd:
            continue
        for stage in ("validation", "test"):
            runs = {}
            for s in seeds:
                runs[f"dataonly_seed{s}"] = {
                    label: {
                        "epoch": 50,
                        "optimiser_steps": 5100,
                        "train_seconds": 190.0,
                        **(
                            {"validation_rel_l2": val_by_wd[wd] + 0.001 * s, "train_rel_l2": 0.008}
                            if stage == "validation"
                            else {
                                "test_rel_l2": 0.038 + test_offset_by_wd[wd] + 0.0005 * s + {"es": 0.0001, "vb": 0.0002}.get(label[:2], 0),
                                "test_rel_l2_by_region": {"other_cells": 0.05, "constant_k_cells": 0.034},
                                "test_solver_residual": {"abs_mean": 2.4, "abs_mean_interface_nodes": 11.0},
                            }
                        ),
                    }
                    for label in labels
                }
            (audit / f"{study_name}_variants_{stage}.json").write_text(json.dumps({"runs": runs}))
    for stage in ("validation", "test"):
        runs = {
            f"consistent_w0.1_seed{s}": {
                "final": {
                    "epoch": 50,
                    "optimiser_steps": 5100,
                    "train_seconds": 193.5,
                    **(
                        {"validation_rel_l2": 0.034, "train_rel_l2": 0.017}
                        if stage == "validation"
                        else {
                            "test_rel_l2": pino_test + 0.0004 * s,
                            "test_rel_l2_by_region": {"other_cells": 0.042, "constant_k_cells": 0.026},
                            "test_solver_residual": {"abs_mean": 0.5, "abs_mean_interface_nodes": 7.7},
                        }
                    ),
                }
            }
            for s in seeds
        }
        (audit / f"pino_variants_{stage}.json").write_text(json.dumps({"runs": runs}))


GRID_VAL = {0.0: 0.045, 0.001: 0.0449, 0.01: 0.0440, 0.1: 0.0445, 1.0: 0.050, 10.0: 0.060}
GRID_TEST = dict.fromkeys(GRID_VAL, 0.0)


def test_selection_uses_validation_only_and_picks_the_lowest_non_zero_decay(tmp_path):
    _fake_records(tmp_path, GRID_VAL, GRID_TEST)
    for wd, name in RS.GRID.items():  # the selection must not need the test files
        (tmp_path / f"{name}_variants_test.json").unlink()
    sel = RS.select(audit=tmp_path)
    assert sel["selected_weight_decay"] == 0.01 and sel["status"] == "final" and sel["test_set_read"] is False
    assert sel["budgets"]["A"] == "epoch050" and sel["budgets"]["ES_A"] == "es_cap050"


def test_a_selection_at_the_upper_boundary_asks_for_the_extension(tmp_path):
    val = {**GRID_VAL, 10.0: 0.040}
    _fake_records(tmp_path, val, GRID_TEST)
    sel = RS.select(audit=tmp_path)
    assert sel["status"] == "extend" and sel["extension_to_train"] == ["reg_cuda_wd1e2"]
    with pytest.raises(SystemExit, match="not final"):
        RS.analyse(sel, audit=tmp_path, pino_studies=("pino",))
    _fake_records(tmp_path, {**val, 100.0: 0.041}, {**GRID_TEST, 100.0: 0.0})
    again = RS.select(audit=tmp_path)
    assert again["status"] == "final" and again["selected_weight_decay"] == 10.0


def test_analysis_computes_the_primary_and_the_holm_family(tmp_path):
    _fake_records(tmp_path, GRID_VAL, GRID_TEST)
    result = RS.analyse(RS.select(audit=tmp_path), audit=tmp_path, pino_studies=("pino",))
    p = result["primary"]
    assert p["comparison"].startswith("budget A: PINO - WD+VB") and p["n_seeds"] == 5
    # WD+VB at budget A: 0.038 + 0.0005 s + 0.0002; PINO: 0.030 + 0.0004 s
    expected = [(0.030 + 0.0004 * s) - (0.0382 + 0.0005 * s) for s in RS.SEEDS]
    assert p["per_seed"] == pytest.approx(expected) and p["reading"] == "setting better"
    assert len(result["secondary_holm"]) == len(RS.SECONDARY)
    assert all(e["holm_p_value"] >= e["t_p_value"] for e in result["secondary_holm"].values())
    assert result["arms"]["A"]["ES"]["test_rel_l2"]["mean"] == pytest.approx(0.038 + 0.001 + 0.0001)
    assert result["arms"]["B"]["WD+VB"]["test_rel_l2"]["mean"] == pytest.approx(0.038 + 0.001 + 0.0002)
    assert "A: PINO - WD+VB" not in result["secondary_holm"] and "B: PINO - WD+VB" in result["secondary_holm"]
    assert set(result["grid"]) == {str(wd) for wd in GRID_VAL} and result["grid"]["0.01"]["selected"]
    assert result["grid"]["0.01"]["mean_validation_rel_l2"] == pytest.approx(0.0440 + 0.002)
    # the synthetic grid has the same test errors for every weight decay, so WD and base coincide; ES does not
    assert result["secondary_holm"]["A: WD - base"]["identical_models"] is True
    assert result["secondary_holm"]["A: ES - base"]["identical_models"] is False


def test_test_evaluation_refuses_to_run_before_the_selection(tmp_path):
    with pytest.raises(SystemExit, match="before any test evaluation"):
        V.evaluate(load_config("reg_smoke"), "test", root=tmp_path, selection=tmp_path / "missing.json")
