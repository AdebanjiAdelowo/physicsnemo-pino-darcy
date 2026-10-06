"""Figures of a study directory (``summary.json``, ``fields.npz`` and the run histories)."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
# identity of the residual a model was trained with; markers repeat the distinction
SERIES = {
    "shipped": {"color": "#eb6834", "marker": "s", "label": "trained with the residual as shipped"},
    "consistent": {"color": "#2a78d6", "marker": "o", "label": "trained with the data-consistent residual"},
}
DATA_ONLY = {"color": MUTED, "label": "data only (weight 0)"}
# one hue, light to dark, for the ordered quantity "physics weight"
WEIGHT_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
FIELD_CMAP, ERROR_CMAP, INPUT_CMAP = "viridis", "magma", "Greys"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
)


def load_study(study_dir: Path) -> dict:
    study_dir = Path(study_dir)
    summary = json.loads((study_dir / "summary.json").read_text())
    histories = {}
    for run in summary["runs"]:
        with open(study_dir / run["run"] / "history.csv") as f:
            rows = list(csv.DictReader(f))
        histories[run["run"]] = {key: np.array([float(r[key]) if r[key] != "" else np.nan for r in rows]) for key in rows[0]}
    fields = dict(np.load(study_dir / "fields.npz"))
    return {"summary": summary, "histories": histories, "fields": fields, "name": summary["name"]}


def _settings(summary: dict, residual: str | None) -> list[dict]:
    return sorted((e for e in summary["by_setting"] if e["residual"] == residual), key=lambda e: e["physics_weight"])


def _sweep(ax, summary, key, reference=None):
    """One quantity against the physics weight, a line per training residual, data-only as a level."""
    for residual, style in SERIES.items():
        entries = _settings(summary, residual)
        if not entries:
            continue
        w = [e["physics_weight"] for e in entries]
        mean = np.array([e[f"{key}_mean"] for e in entries])
        std = np.array([e[f"{key}_std"] or 0.0 for e in entries])
        ax.errorbar(w, mean, yerr=std if std.any() else None, color=style["color"], marker=style["marker"], capsize=3,
                    markeredgecolor="white", markeredgewidth=1.2, label=style["label"])
    base = _settings(summary, None)
    if base:
        ax.axhline(base[0][f"{key}_mean"], color=DATA_ONLY["color"], linestyle="--", linewidth=1.4, label=DATA_ONLY["label"])
    if reference is not None:
        ax.axhline(reference, color=INK, linestyle=":", linewidth=1.4, label="reference solution of the dataset")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physics weight")


def plot_weight_sweep(study: dict, path: Path) -> Path:
    summary = study["summary"]
    reference = summary["reference_residual"]["test"]
    panels = [
        ("test_rel_l2_mean", "Relative $L^2$ error (test)", None),
        ("final_train_loss_data", "Data loss, last epoch (training)", None),
        ("test_pde_l1_consistent", "Mean |residual|, data-consistent forcing (test)", reference["consistent"]["l1"]),
        ("test_pde_l1_shipped", "Mean |residual|, forcing as shipped (test)", reference["shipped"]["l1"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    for ax, (key, title, ref) in zip(axes.flat, panels):
        _sweep(ax, summary, key, ref)
        ax.set_title(title, loc="left")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tradeoff(study: dict, path: Path) -> Path:
    """Prediction error against PDE residual; every point is one physics weight."""
    summary = study["summary"]
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    for residual, style in SERIES.items():
        entries = _settings(summary, residual)
        if not entries:
            continue
        x = [e["test_rel_l2_mean_mean"] for e in entries]
        y = [e["test_pde_l1_consistent_mean"] for e in entries]
        ax.plot(x, y, color=style["color"], marker=style["marker"], markeredgecolor="white", markeredgewidth=1.2, label=style["label"])
        offset, align = ((8, 6), "left") if residual == "shipped" else ((-8, -12), "right")
        for e, xi, yi in zip(entries, x, y):
            ax.annotate(f"{e['physics_weight']:g}", (xi, yi), textcoords="offset points", xytext=offset, ha=align, fontsize=9, color=INK)
    base = _settings(summary, None)
    if base:
        ax.plot(base[0]["test_rel_l2_mean_mean"], base[0]["test_pde_l1_consistent_mean"], linestyle="none", marker="D",
                color=DATA_ONLY["color"], markeredgecolor="white", markeredgewidth=1.2, label=DATA_ONLY["label"])
    ax.axhline(summary["reference_residual"]["test"]["consistent"]["l1"], color=INK, linestyle=":", linewidth=1.4,
               label="reference solution of the dataset")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative $L^2$ error (test)")
    ax.set_ylabel("mean |residual|, data-consistent forcing (test)")
    ax.set_title("Prediction error against PDE residual; labels are physics weights", loc="left")
    ax.legend(loc="best")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_training_curves(study: dict, path: Path) -> Path:
    """Validation error and validation residual per epoch, first seed, one row per training residual."""
    summary, histories = study["summary"], study["histories"]
    seed = min(r["seed"] for r in summary["runs"])
    residuals = [r for r in SERIES if _settings(summary, r)]
    fig, axes = plt.subplots(len(residuals), 2, figsize=(10.5, 3.6 * len(residuals)), constrained_layout=True, squeeze=False)
    for row, residual in zip(axes, residuals):
        runs = sorted((r for r in summary["runs"] if r["seed"] == seed and r["residual"] == residual), key=lambda r: r["physics_weight"])
        ramp = WEIGHT_RAMP[len(WEIGHT_RAMP) - len(runs) :] if len(runs) <= len(WEIGHT_RAMP) else None
        columns = [("validation_rel_l2", "Relative $L^2$ error (validation)"), (f"validation_pde_l1_{residual}", "Mean |residual| (validation)")]
        for ax, (key, title) in zip(row, columns):
            base = next((r for r in summary["runs"] if r["seed"] == seed and r["residual"] is None), None)
            if base:
                h = histories[base["run"]]
                ax.plot(h["epoch"], h[key], color=DATA_ONLY["color"], linestyle="--", linewidth=1.4, label="weight 0")
            for i, run in enumerate(runs):
                h = histories[run["run"]]
                ax.plot(h["epoch"], h[key], color=ramp[i] if ramp else None, label=f"weight {run['physics_weight']:g}")
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_title(f"{title}, {SERIES[residual]['label'].replace('trained with the ', '')}", loc="left")
        row[0].legend(ncol=2, fontsize=9)
    fig.savefig(path)
    plt.close(fig)
    return path


def _shown_runs(study: dict) -> list[tuple[str, str]]:
    """Data-only run and, per training residual, the run at the upstream weight (or the nearest)."""
    summary = study["summary"]
    seed = int(study["fields"]["plot_seed"])
    shown = []
    base = next((r for r in summary["runs"] if r["seed"] == seed and r["residual"] is None), None)
    if base:
        shown.append((base["run"], "Data only"))
    for residual in SERIES:
        runs = [r for r in summary["runs"] if r["seed"] == seed and r["residual"] == residual]
        if runs:
            run = min(runs, key=lambda r: abs(np.log10(r["physics_weight"]) + 1))
            shown.append((run["run"], f"{residual.capitalize()} residual, weight {run['physics_weight']:g}"))
    return shown


def _image(ax, data, title, **kwargs):
    im = ax.imshow(data.T, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest", **kwargs)
    ax.set_title(title, loc="left", fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return im


def plot_fields(study: dict, path: Path, sample: int = 0) -> Path:
    """Permeability, reference, prediction and absolute error of one test sample, one row per model."""
    fields = study["fields"]
    shown = _shown_runs(study)
    k, truth = fields["permeability"][sample], fields["truth"][sample]
    errors = [np.abs(fields[f"prediction__{run}"][sample].astype(np.float64) - truth) for run, _ in shown]
    vmax_error = max(e.max() for e in errors)
    fig, axes = plt.subplots(len(shown), 4, figsize=(12.5, 3.05 * len(shown)), constrained_layout=True, squeeze=False)
    for row, (run, label), error in zip(axes, shown, errors):
        im = _image(row[0], k, "Permeability $k'$ (input)", cmap=INPUT_CMAP)
        fig.colorbar(im, ax=row[0], fraction=0.046, pad=0.03)
        im = _image(row[1], truth, "Reference pressure $u'$", cmap=FIELD_CMAP, vmin=truth.min(), vmax=truth.max())
        fig.colorbar(im, ax=row[1], fraction=0.046, pad=0.03)
        im = _image(row[2], fields[f"prediction__{run}"][sample], f"Prediction: {label}", cmap=FIELD_CMAP, vmin=truth.min(), vmax=truth.max())
        fig.colorbar(im, ax=row[2], fraction=0.046, pad=0.03)
        rel = np.linalg.norm(error) / np.linalg.norm(truth)
        im = _image(row[3], error, f"Absolute error (relative $L^2$ {rel:.3f})", cmap=ERROR_CMAP, vmin=0, vmax=vmax_error)
        fig.colorbar(im, ax=row[3], fraction=0.046, pad=0.03)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_residual_fields(study: dict, path: Path, sample: int = 0) -> Path:
    """|residual| of the shown models under both residual definitions, logarithmic colour scale."""
    fields = study["fields"]
    shown = _shown_runs(study)
    norm = LogNorm(vmin=1e-1, vmax=1e3)
    fig, axes = plt.subplots(2, len(shown), figsize=(4.1 * len(shown), 7.6), constrained_layout=True, squeeze=False)
    for row, definition in zip(axes, ("consistent", "shipped")):
        for ax, (run, label) in zip(row, shown):
            data = np.abs(fields[f"residual_{definition}__{run}"][sample].astype(np.float64))
            title = f"{label}\nresidual with {'data-consistent forcing' if definition == 'consistent' else 'forcing as shipped'}"
            im = _image(ax, np.clip(data, norm.vmin, None), title, cmap=ERROR_CMAP, norm=norm)
        fig.colorbar(im, ax=row, fraction=0.025, pad=0.02, label="|residual| (normalised units)")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_data_check(k: np.ndarray, u: np.ndarray, residuals: dict, path: Path) -> Path:
    """One sample of the dataset and the residual of its reference pressure under both definitions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = LogNorm(vmin=1e-2, vmax=1e3)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), constrained_layout=True)
    im = _image(axes[0], k, "Permeability $k'$ (input)", cmap=INPUT_CMAP)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    im = _image(axes[1], u, "Reference pressure $u'$", cmap=FIELD_CMAP)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)
    titles = {"shipped": "|residual| of the reference, forcing as shipped", "consistent": "|residual| of the reference, data-consistent forcing"}
    for ax, name in zip(axes[2:], ("shipped", "consistent")):
        im = _image(ax, np.clip(np.abs(residuals[name]), norm.vmin, None), titles[name], cmap=ERROR_CMAP, norm=norm)
    fig.colorbar(im, ax=axes[2:], fraction=0.025, pad=0.02, label="normalised units")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_all(study_dir: Path, out_dir: Path, prefix: str | None = None) -> list[Path]:
    study = load_study(study_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or study["name"]
    plots = {
        "weight_sweep": plot_weight_sweep,
        "tradeoff": plot_tradeoff,
        "training_curves": plot_training_curves,
        "fields": plot_fields,
        "residual_fields": plot_residual_fields,
    }
    return [plot(study, out_dir / f"{prefix}_{name}.png") for name, plot in plots.items()]
