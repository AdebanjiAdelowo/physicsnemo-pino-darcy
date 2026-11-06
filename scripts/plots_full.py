"""Figures of ``scripts/full_study.py``. Every value comes from the records it passes in, or from the
stored ``fields.npz`` of a published study; nothing is computed from a model here except the exact
residual field of a stored prediction (``reference.solver_residual``)."""

import h5py
import numpy as np

from _bootstrap import ROOT
from pino_darcy import data as D
from pino_darcy import reference as R
from pino_darcy.plotting import DATA_ONLY, ERROR_CMAP, FIELD_CMAP, INK, INPUT_CMAP, MUTED, SERIES, plt

SEEDS = (0, 1, 2)
COLOURS = {
    "data only": DATA_ONLY["color"],
    "shipped": SERIES["shipped"]["color"],
    "consistent": SERIES["consistent"]["color"],
    "consistent_coeff": "#7a4fc9",
    "solver_kcoeff": "#c9a227",
    "solver": "#1a9a6c",
}
MARKERS = {"data only": "D", "shipped": "s", "consistent": "o", "consistent_coeff": "^", "solver_kcoeff": "v", "solver": "P"}


def family_of(label: str) -> str:
    for name in ("consistent_coeff", "solver_kcoeff", "solver", "consistent", "shipped"):
        if name in label:
            return name
    return "data only"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("written", path)


def _t_interval(values):
    from scipy import stats

    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, v.size - 1) * v.std(ddof=1) / np.sqrt(v.size)
    return v.mean(), half


def plot_fields(out):
    """Permeability, reference, prediction, |error| and exact residual for test sample 0, seed 0."""
    rows = [
        ("data only", "full_mps", "dataonly_seed0"),
        ("consistent 0.1", "full_mps", "consistent_w0.1_seed0"),
        ("solver 0.1", "full_mps_operator", "solver_w0.1_seed0"),
    ]
    rows = [r for r in rows if (ROOT / "results" / r[1] / "fields.npz").exists()]
    if not rows:
        return
    with h5py.File(ROOT / "datasets" / "Darcy_241" / "test.hdf5") as f:
        coeff = np.asarray(f["coeff"][0, 0], dtype=np.float64)
    first = np.load(ROOT / "results" / rows[0][1] / "fields.npz")
    k, truth = first["permeability"][0] * D.K_SCALE, first["truth"][0]
    preds = {label: np.load(ROOT / "results" / study / "fields.npz")[f"prediction__{run}"][0] for label, study, run in rows}
    err_max = max(np.abs(p - truth).max() for p in preds.values())
    res = {label: np.abs(R.solver_residual(R.extend_crop(p * D.U_SCALE), coeff))[6:-6, 6:-6] for label, p in preds.items()}
    ref_res = np.abs(R.solver_residual(R.extend_crop(truth.astype(np.float64) * D.U_SCALE), coeff))[6:-6, 6:-6]
    from matplotlib.colors import LogNorm

    norm = LogNorm(1e-4, max(r.max() for r in res.values()))
    fig, axes = plt.subplots(len(rows) + 1, 3, figsize=(10.5, 3.2 * (len(rows) + 1)))
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    im = axes[0, 0].imshow(k, cmap=INPUT_CMAP)
    axes[0, 0].set_title("input Kcoeff (physical)")
    fig.colorbar(im, ax=axes[0, 0], shrink=0.8)
    im = axes[0, 1].imshow(truth, cmap=FIELD_CMAP)
    axes[0, 1].set_title("reference pressure (normalised)")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.8)
    im = axes[0, 2].imshow(ref_res, cmap=ERROR_CMAP, norm=norm)
    axes[0, 2].set_title(f"reference: |exact residual|, mean {ref_res.mean():.1e}")
    fig.colorbar(im, ax=axes[0, 2], shrink=0.8)
    for i, (label, _, _) in enumerate(rows, start=1):
        p = preds[label]
        im = axes[i, 0].imshow(p, cmap=FIELD_CMAP, vmin=truth.min(), vmax=truth.max())
        rel = np.linalg.norm(p - truth) / np.linalg.norm(truth)
        axes[i, 0].set_title(f"{label}: prediction (rel L2 {rel:.4f})")
        fig.colorbar(im, ax=axes[i, 0], shrink=0.8)
        im = axes[i, 1].imshow(np.abs(p - truth), cmap=ERROR_CMAP, vmin=0, vmax=err_max)
        axes[i, 1].set_title(f"{label}: |prediction - reference|")
        fig.colorbar(im, ax=axes[i, 1], shrink=0.8)
        im = axes[i, 2].imshow(res[label], cmap=ERROR_CMAP, norm=norm)
        axes[i, 2].set_title(f"{label}: |exact residual|, mean {res[label].mean():.2f}")
        fig.colorbar(im, ax=axes[i, 2], shrink=0.8)
    fig.suptitle("Test sample 0, seed 0, 50 epochs. Exact residual: generator operator with coeff, physical units (f = 1), 6-node margin", fontsize=10)
    _save(fig, out / "full_fields.png")


def plot_training_curves(records, out):
    groups = [
        ["data only", "shipped 0.1", "consistent 0.1", "solver 0.1", "consistent_coeff 0.1", "solver_kcoeff 0.1"],
        ["N=25 data only", "N=25 solver 0.1", "N=25 solver 0.1 + unlabelled", "N=50 data only", "N=50 solver 0.1", "N=50 solver 0.1 + unlabelled"],
    ]
    groups = [[x for x in g if x in records] for g in groups]
    groups = [g for g in groups if g]
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 3.8 * len(groups)), squeeze=False)
    for row, labels in zip(axes, groups, strict=True):
        for label in labels:
            fam = family_of(label)
            ls = "--" if "N=25" in label else ("-" if "N=50" not in label else "-.")
            ls = ":" if "unlabelled" in label else ls
            for ax, key in ((row[0], "validation_rel_l2"), (row[1], "train_loss_data")):
                curves = [records[label][s]["history"][key] for s in SEEDS]
                steps = records[label][0]["history"]["epoch"] * records[label][0]["steps_per_epoch"]
                c = np.vstack(curves)
                ax.plot(steps, c.mean(0), color=COLOURS[fam], ls=ls, label=label, lw=1.6)
                ax.fill_between(steps, c.min(0), c.max(0), color=COLOURS[fam], alpha=0.15, lw=0)
        row[0].set_ylabel("validation relative L2 (mean, band = min to max over seeds)")
        row[1].set_ylabel("training data loss (MSE, normalised)")
        for ax in row:
            ax.set_yscale("log")
            ax.set_xlabel("optimiser step")
        row[0].legend(fontsize=8)
    _save(fig, out / "full_training_curves.png")


def plot_weight_sweep(records, out):
    if "data only" not in records or not any(f"consistent {w}" in records for w in ("0.001", "1")):
        return
    weights = [0.001, 0.01, 0.1, 1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, ylabel in ((axes[0], "rel_l2", "test relative L2 error"), (axes[1], "exact_residual", "exact residual, mean |r| (physical, f = 1)")):
        m, h = _t_interval([records["data only"][s]["metrics"][key] for s in SEEDS])
        ax.axhspan(m - h, m + h, color=DATA_ONLY["color"], alpha=0.15, lw=0)
        ax.axhline(m, color=DATA_ONLY["color"], lw=1.2, label="data only (mean, 95 % t interval over seeds)")
        for res in ("shipped", "consistent", "solver"):
            ws = [w for w in weights if f"{res} {w:g}" in records]
            if not ws:
                continue
            stats = [_t_interval([records[f"{res} {w:g}"][s]["metrics"][key] for s in SEEDS]) for w in ws]
            ax.errorbar(ws, [x[0] for x in stats], yerr=[x[1] for x in stats], color=COLOURS[res], marker=MARKERS[res], capsize=4, label=res)
            for w in ws:
                ax.scatter([w] * 3, [records[f"{res} {w:g}"][s]["metrics"][key] for s in SEEDS], color=COLOURS[res], s=10, alpha=0.5)
        ax.set_xscale("log")
        if key == "exact_residual":
            ax.set_yscale("log")
        ax.set_xlabel("physics weight")
        ax.set_ylabel(ylabel)
    axes[0].legend(fontsize=8)
    axes[0].set_title("50 epochs, seeds 0 to 2; error bars: 95 % t interval over seeds")
    _save(fig, out / "full_weight_sweep.png")


def plot_intervals(families, out, key="rel_l2", name="full_seed_intervals.png", title="overall test relative L2"):
    rows = [(fam, label, c[key]) for fam, f in families.items() for label, c in f["comparisons"].items() if key in c]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(rows) + 1.4))
    for i, (fam, label, c) in enumerate(rows[::-1]):
        col = COLOURS[family_of(label.split(" - ")[0])]
        rel = lambda x, c=c: 100 * x / c["control_mean"]  # noqa: E731
        ax.errorbar(rel(c["mean"]), i, xerr=[[rel(c["mean"] - c["ci_low"])], [rel(c["ci_high"] - c["mean"])]], color=col, marker="o", capsize=3)
        ax.scatter([rel(x) for x in c["per_seed"]], [i] * len(c["per_seed"]), color=col, s=12, alpha=0.5, zorder=3)
        ax.text(1.01, i, c["reading"], transform=ax.get_yaxis_transform(), fontsize=7.5, va="center", color=MUTED)
    ax.axvline(0, color=INK, lw=0.8)
    ax.axvspan(-5, 5, color=MUTED, alpha=0.08, lw=0)
    ax.set_yticks(range(len(rows)), [f"[{f}] {lab}" for f, lab, _ in rows[::-1]], fontsize=8)
    ax.set_xlabel(f"paired difference in {title}, % of the control mean (dots: seeds; bar: 95 % t interval, df = 2; band: +-5 %)")
    _save(fig, out / name)


def plot_regions(records, out):
    labels = [x for x in ("shipped 0.1", "consistent 0.1", "consistent_coeff 0.1", "solver_kcoeff 0.1", "solver 0.1", "solver 1") if x in records]
    if "data only" not in records or not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, (key, name) in enumerate((("rel_l2_constant_k", "constant-Kcoeff cells"), ("rel_l2_near_jumps", "near jumps"), ("rel_l2", "all cells"))):
        for i, label in enumerate(labels):
            d = [records[label][s]["metrics"][key] - records["data only"][s]["metrics"][key] for s in SEEDS]
            m, h = _t_interval(d)
            x = i + (j - 1) * 0.25
            ax.errorbar(
                x,
                m,
                yerr=h,
                color=COLOURS[family_of(label)],
                marker=("o", "^", "s")[j],
                capsize=3,
                label=name if i == 0 else None,
                mfc="white" if j == 0 else None,
            )
            ax.scatter([x] * 3, d, color=COLOURS[family_of(label)], s=9, alpha=0.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("relative L2 error minus data only (same seed)")
    ax.set_title("Error by region, 50 epochs: circle (open) constant-Kcoeff cells, triangle near jumps, square all; 95 % t interval over seeds")
    ax.legend(fontsize=8)
    _save(fig, out / "full_regions.png")


def plot_operator(records, out):
    labels = [x for x in ("data only", "consistent 0.1", "consistent_coeff 0.1", "solver_kcoeff 0.1", "solver 0.1") if x in records]
    if len(labels) < 5:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    keys = (
        ("rel_l2", "test relative L2"),
        ("rel_l2_near_jumps", "rel. L2 near jumps"),
        ("rel_l2_constant_k", "rel. L2, constant-Kcoeff cells"),
        ("exact_residual", "exact residual (log)"),
    )
    for ax, (key, name) in zip(axes, keys, strict=True):
        for i, label in enumerate(labels):
            v = [records[label][s]["metrics"][key] for s in SEEDS]
            ax.scatter([i] * 3, v, color=COLOURS[family_of(label)], marker=MARKERS[family_of(label)], s=26)
            ax.hlines(np.mean(v), i - 0.25, i + 0.25, color=COLOURS[family_of(label)])
        for s in SEEDS:
            ax.plot(range(len(labels)), [records[label][s]["metrics"][key] for label in labels], color=MUTED, lw=0.6, alpha=0.5)
        ax.set_xticks(range(len(labels)), ["data only", "FD, Kcoeff", "FD, coeff", "solver, Kcoeff", "solver, coeff"], rotation=25, fontsize=8)
        ax.set_title(name)
        if key == "exact_residual":
            ax.set_yscale("log")
    fig.suptitle("Residual formulations at weight 0.1, 50 epochs (lines join one seed; bar: mean)", fontsize=10)
    _save(fig, out / "full_operator.png")


def plot_labels(records, out):
    ns = [25, 50, 102]
    series = {
        "data only": {25: "N=25 data only", 50: "N=50 data only", 102: "data only"},
        "solver 0.1 (labelled fields)": {25: "N=25 solver 0.1", 50: "N=50 solver 0.1", 102: "solver 0.1"},
        "solver 0.1 + unlabelled fields": {25: "N=25 solver 0.1 + unlabelled", 50: "N=50 solver 0.1 + unlabelled"},
    }
    if "N=25 data only" not in records:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    style = {
        "data only": (DATA_ONLY["color"], "D", 0),
        "solver 0.1 (labelled fields)": (COLOURS["solver"], "P", 1),
        "solver 0.1 + unlabelled fields": ("#0d5c3f", "X", 2),
    }
    for name, by_n in series.items():
        col, mk, off = style[name]
        for ax, key in ((axes[0], "rel_l2"), (axes[1], "rel_l2_near_jumps")):
            pts = [(n, by_n[n]) for n in ns if n in by_n and by_n[n] in records]
            if not pts:
                continue
            xs = [n * (1 + 0.03 * (off - 1)) for n, _ in pts]
            st = [_t_interval([records[lab][s]["metrics"][key] for s in SEEDS]) for _, lab in pts]
            ax.errorbar(xs, [x[0] for x in st], yerr=[x[1] for x in st], color=col, marker=mk, capsize=4, label=name)
            for x, (_, lab) in zip(xs, pts, strict=True):
                ax.scatter([x] * 3, [records[lab][s]["metrics"][key] for s in SEEDS], color=col, s=10, alpha=0.5)
    for ax, t in zip(axes, ("test relative L2 error", "relative L2 near jumps"), strict=True):
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.set_xlabel("labelled training samples (5100 optimiser steps in every run)")
        ax.set_ylabel(t)
    axes[0].legend(fontsize=8)
    axes[0].set_title("mean and 95 % t interval over seeds 0 to 2; dots: seeds")
    _save(fig, out / "full_labels.png")


def plot_cost(records, out):
    labels = [
        lab
        for lab in records
        if lab
        in (
            "data only",
            "shipped 0.1",
            "consistent 0.1",
            "consistent_coeff 0.1",
            "solver_kcoeff 0.1",
            "solver 0.1",
            "N=25 solver 0.1 + unlabelled",
            "N=50 solver 0.1 + unlabelled",
        )
    ]
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for i, label in enumerate(labels):
        col = COLOURS[family_of(label)]
        for ax, key in ((axes[0], "optimisation_ms_per_step"), (axes[1], "inference_ms_per_sample")):
            v = [records[label][s]["cost"][key] for s in SEEDS]
            ax.bar(i, np.mean(v), color=col, alpha=0.8)
            ax.scatter([i] * 3, v, color=INK, s=8)
    for ax, t in zip(axes, ("training: ms per optimiser step (FNO on MPS, cell-grid residual on CPU)", "inference: ms per sample, forward pass"), strict=True):
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(t, fontsize=9)
    _save(fig, out / "full_cost.png")


def plot_epochs(epochs, out):
    labels = [lab for lab in ("data only", "shipped 0.1", "consistent 0.1") if lab in epochs and epochs[lab]]
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, key, t in ((axes[0], "rel_l2", "test relative L2"), (axes[1], "exact_residual", "exact residual (log)")):
        for label in labels:
            fam = family_of(label)
            for s in SEEDS:
                e = epochs[label].get(f"seed{s}")
                if e:
                    ax.plot(
                        [15, 50], [e[key]["15_epochs"], e[key]["50_epochs"]], color=COLOURS[fam], marker=MARKERS[fam], lw=1, label=label if s == 0 else None
                    )
        ax.set_xticks([15, 50])
        ax.set_xlabel("epochs (102 steps each; same seed joined)")
        ax.set_title(t)
        if key == "exact_residual":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    _save(fig, out / "full_epochs.png")


def plot_all(records, families, factorial, epochs, out):
    out.mkdir(parents=True, exist_ok=True)
    plot_fields(out)
    plot_training_curves(records, out)
    plot_weight_sweep(records, out)
    plot_intervals(families, out)
    plot_intervals(families, out, key="rel_l2_near_jumps", name="full_seed_intervals_near_jumps.png", title="relative L2 near jumps")
    plot_regions(records, out)
    plot_operator(records, out)
    plot_labels(records, out)
    plot_cost(records, out)
    plot_epochs(epochs, out)
