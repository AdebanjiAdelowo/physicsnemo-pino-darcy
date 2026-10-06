"""The figures are drawn from a study directory without error."""

from conftest import synthetic_sets

from pino_darcy import engine, study
from pino_darcy.plotting import plot_all


def test_all_figures_are_written(tiny_cfg, tmp_path, monkeypatch):
    sets = synthetic_sets()
    monkeypatch.setattr(engine, "load_datasets", lambda cfg, root, with_train=True: sets)
    out_dir = study.train_study(tiny_cfg, tmp_path)
    study.evaluate_study(tiny_cfg, tmp_path)
    paths = plot_all(out_dir, tmp_path / "figures")
    assert [p.name for p in paths] == [
        f"smoke_{name}.png" for name in ("weight_sweep", "tradeoff", "training_curves", "fields", "residual_fields")
    ]
    assert all(p.stat().st_size > 10_000 for p in paths)
