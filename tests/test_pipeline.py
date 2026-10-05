"""Loss decomposition, training records, checkpoints and summaries on a tiny synthetic problem."""

import csv
import json

import pytest
import torch
import torch.nn.functional as F

from pino_darcy import engine, study
from pino_darcy import physics as P
from pino_darcy.model import build_fno


@pytest.fixture
def trained(tiny_cfg, sets, tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "load_datasets", lambda cfg, root, with_train=True: sets)
    out_dir = study.train_study(tiny_cfg, tmp_path)
    return tiny_cfg, out_dir, study.evaluate_study(tiny_cfg, tmp_path)


def history(run_dir):
    with open(run_dir / "history.csv") as f:
        return list(csv.DictReader(f))


def test_first_step_loss_is_the_upstream_expression(tiny_cfg, sets):
    """data + 1/240 * weight * mean|residual|, written out as in darcy_physics_informed_fno.py."""
    torch.manual_seed(0)
    model = build_fno(tiny_cfg)
    invar, outvar = sets["train"]["k"][:1], sets["train"]["u"][:1]
    out = model(invar)
    informer = P.build_informer("shipped", torch.device("cpu"))
    pde_out_arr = informer.forward({"u": out, "k": invar[:, 0:1]})["diffusion_u"]
    pde_out_arr = F.pad(pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0)
    loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))
    expected = F.mse_loss(outvar, out) + 1 / 240 * 0.1 * loss_pde

    mine = F.mse_loss(outvar, out) + P.PDE_LOSS_FACTOR * 0.1 * P.pde_loss(P.residual_field(informer, out, invar))
    assert mine.item() == pytest.approx(expected.item(), rel=1e-6)


def test_history_separates_data_and_pde_terms(trained):
    cfg, out_dir, _ = trained
    for name, weight in (("shipped_w0.1_seed0", 0.1), ("consistent_w0.1_seed0", 0.1)):
        for row in history(out_dir / name):
            total = float(row["train_loss_data"]) + weight / 240 * float(row["train_loss_pde"])
            assert float(row["train_loss_total"]) == pytest.approx(total, rel=1e-4)
    for row in history(out_dir / "dataonly_seed0"):
        assert row["train_loss_pde"] == "" and row["train_loss_total"] == row["train_loss_data"]
        assert float(row["validation_pde_l1_shipped"]) > 0 and float(row["validation_pde_l1_consistent"]) > 0
    assert len(history(out_dir / "dataonly_seed0")) == cfg.max_epochs


def test_physics_weight_changes_the_trained_model(trained):
    _, out_dir, summary = trained
    errors = {r["run"]: r["test_rel_l2_mean"] for r in summary["runs"]}
    assert len(set(errors.values())) == 3


def test_metadata_and_summary_are_complete(trained):
    cfg, out_dir, summary = trained
    meta = json.loads((out_dir / "study_metadata.json").read_text())
    for key in ("timestamp_utc", "git_commit", "git_dirty", "upstream", "device", "python_version", "torch_version",
                "physicsnemo", "warp_version", "config", "dataset", "residual_definitions", "physics_device"):
        assert key in meta, key
    assert meta["upstream"]["commit"] == "b45a5c810c741e6b41f8515be24c51121f8fc21f"
    assert meta["config"]["study"]["physics_weights"] == [0, 0.1]
    assert [r["run"] for r in summary["runs"]] == ["dataonly_seed0", "shipped_w0.1_seed0", "consistent_w0.1_seed0"]
    assert set(summary["reference_residual"]["test"]) == {"shipped", "consistent"}
    train = json.loads((out_dir / "shipped_w0.1_seed0" / "train_metrics.json").read_text())
    assert train["physics_weight"] == 0.1 and train["effective_pde_coefficient"] == pytest.approx(0.1 / 240)
    assert train["seed"] == 0 and train["train_seconds"] > 0 and train["parameters"] > 0
    with open(out_dir / "summary.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 and float(rows[1]["test_pde_l1_consistent"]) > 0
    entry = next(e for e in summary["by_setting"] if e["residual"] == "shipped")
    assert entry["n_seeds"] == 1 and entry["test_rel_l2_mean_std"] is None


def test_checkpoint_reproduces_the_stored_errors(trained, sets, tmp_path, monkeypatch):
    cfg, out_dir, _ = trained
    before = (out_dir / "dataonly_seed0" / "eval_metrics.json").read_text()
    rows = study.verify_study(cfg, tmp_path)
    assert max(r["relative_difference"] for r in rows) < 1e-6
    assert (out_dir / "dataonly_seed0" / "eval_metrics.json").read_text() == before


def test_same_seed_repeats_the_run(tiny_cfg, sets, tmp_path):
    device = torch.device("cpu")
    a = engine.train_run(tiny_cfg, "consistent", 0.1, 0, tmp_path / "a", device, sets)
    b = engine.train_run(tiny_cfg, "consistent", 0.1, 0, tmp_path / "b", device, sets)
    assert a["final_train_loss_total"] == b["final_train_loss_total"]


def test_resume_skips_finished_runs(trained, tmp_path, capsys):
    cfg, out_dir, _ = trained
    stamp = (out_dir / "dataonly_seed0" / "train_metrics.json").stat().st_mtime_ns
    study.train_study(cfg, tmp_path, resume=True)
    assert (out_dir / "dataonly_seed0" / "train_metrics.json").stat().st_mtime_ns == stamp
    assert capsys.readouterr().out.count("[skip]") == 3
