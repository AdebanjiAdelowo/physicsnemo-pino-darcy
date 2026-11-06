"""Seed study (G1 of docs/EXPERIMENT_GAPS.md): data-only against physics weight 0.1 over seeds 0, 1, 2.

    python scripts/results.py publish runs/local_seeds
    python scripts/seed_study.py

Reads, and does not change, the records of the reduced study (seed 0: ``results/local/``,
``results/audit/local_solver_residual.json``) and of the seed runs (seeds 1 and 2:
``results/local_seeds/``, ``results/audit/local_seeds_solver_residual.json``), and the
checkpoints in ``runs/local/`` and ``runs/local_seeds/``. Writes

- ``results/local_seeds/run_manifest.json``: configuration hashes, checkpoint hashes, dataset
  fingerprints, hardware and training time of all nine runs;
- ``results/audit/seed_study.json``: per-seed metrics, paired per-seed differences with a
  Student-t interval over seeds, per-seed test-sample bootstrap intervals, seed spread;
- ``figures/seed_study.png``.

Two residuals are reported and kept apart. ``cell-grid`` is the residual of the training loss
(``consistent`` definition: smoothed ``Kcoeff``, five-point stencil on the cell grid,
normalised units). ``exact`` is the residual of the discrete problem the reference pressure
solves (``reference.solver_residual`` with the two-valued ``coeff``, physical units), on
which the reference scores 4e-4.
"""

import hashlib
import json
import platform
import subprocess

import numpy as np

from _bootstrap import ROOT
from pino_darcy.metrics import paired_bootstrap, paired_seed_interval
from pino_darcy.provenance import write_json

SEEDS = (0, 1, 2)
SETTINGS = {"data only": "dataonly", "shipped 0.1": "shipped_w0.1", "consistent 0.1": "consistent_w0.1"}
TRAINING_KEYS_EXCLUDED = ("name", "output_dir", "study", "evaluation", "physics_weight")
STUDIES = {
    0: {"runs": ROOT / "runs" / "local", "records": ROOT / "results" / "local", "audit": "local_solver_residual.json"},
    1: {"runs": ROOT / "runs" / "local_seeds", "records": ROOT / "results" / "local_seeds", "audit": "local_seeds_solver_residual.json"},
}
STUDIES[2] = STUDIES[1]
METRICS = {
    "rel_l2": "test relative L2 error",
    "rel_l2_constant_k": "relative L2 error, cells of constant Kcoeff",
    "rel_l2_near_jumps": "relative L2 error, cells near jumps of Kcoeff",
    "exact_residual": "exact residual, mean |r| (coeff, solver operator, physical units)",
    "exact_residual_interface": "exact residual at nodes within two cells of a jump of coeff",
    "exact_residual_other": "exact residual at the other nodes",
    "cellgrid_residual": "cell-grid training residual, mean |r| (consistent definition, Kcoeff, normalised)",
    "cellgrid_residual_constant_k": "cell-grid residual on cells of constant Kcoeff",
    "cellgrid_residual_near_jumps": "cell-grid residual on cells near jumps of Kcoeff",
}
# residuals are compared as ratios (log scale); errors as differences
RATIO_METRICS = {k for k in METRICS if "residual" in k}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def run_metrics(seed: int, prefix: str) -> tuple[dict, list[float]]:
    study = STUDIES[seed]
    run = f"{prefix}_seed{seed}"
    evaluation = load_json(study["records"] / run / "eval_metrics.json")
    audit = load_json(ROOT / "results" / "audit" / study["audit"])["runs"][run]
    cell = evaluation["test_residual_consistent"]
    exact = audit["solver_residual"]
    stored, recomputed = evaluation["test_rel_l2"]["mean"], audit["rel_l2_mean"]
    assert abs(stored - recomputed) <= 1e-5 * stored, f"{run}: stored and recomputed test errors differ"
    metrics = {
        "rel_l2": stored,
        "rel_l2_constant_k": audit["rel_l2_by_region"]["constant_k_cells"],
        "rel_l2_near_jumps": audit["rel_l2_by_region"]["other_cells"],
        "exact_residual": exact["abs_mean"],
        "exact_residual_interface": exact["abs_mean_interface_nodes"],
        "exact_residual_other": exact["abs_mean_other_nodes"],
        "cellgrid_residual": cell["l1"],
        "cellgrid_residual_constant_k": cell["abs_mean_smooth"],
        "cellgrid_residual_near_jumps": cell["abs_mean_interface"],
    }
    return metrics, evaluation["test_rel_l2_per_sample"]


def manifest_entry(seed: int, prefix: str) -> dict:
    study = STUDIES[seed]
    run = f"{prefix}_seed{seed}"
    meta = load_json(study["runs"] / "study_metadata.json")
    train = load_json(study["records"] / run / "train_metrics.json")
    training_config = {k: v for k, v in meta["config"].items() if k not in TRAINING_KEYS_EXCLUDED}
    checkpoints = sorted((study["runs"] / run / "checkpoints").iterdir())
    return {
        "run": run,
        "seed": seed,
        "residual": train["residual"],
        "physics_weight": train["physics_weight"],
        "study_directory": str(study["runs"].relative_to(ROOT)),
        "training_config_sha256": canonical_hash(training_config),
        "run_config_sha256": canonical_hash({**training_config, "residual": train["residual"], "physics_weight": train["physics_weight"], "seed": seed}),
        "git_commit": meta["git_commit"],
        "git_dirty": meta["git_dirty"],
        "trained_at_utc": meta["timestamp_utc"],
        "dataset_files_sha256": meta["dataset"]["files"],
        "dataset_archive_sha256": meta["dataset"]["archive_sha256"],
        "device": meta["device"],
        "physics_device": meta["physics_device"],
        "torch_version": meta["torch_version"],
        "physicsnemo_commit": meta["physicsnemo"]["commit"],
        "platform": meta["platform"],
        "train_seconds": train["train_seconds"],
        "optimisation_ms_per_step": train["optimisation_ms_per_step"],
        "optimiser_steps": train["optimiser_steps"],
        "checkpoints_sha256": {p.name: sha256(p) for p in checkpoints},
    }


def hardware() -> dict:
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    memory = sysctl("hw.memsize")
    return {"cpu": sysctl("machdep.cpu.brand_string"), "memory_gb": int(memory) / 2**30 if memory else None, "platform": platform.platform()}


def compare(per_seed: dict, per_sample: dict, model: str, base: str = "data only") -> dict:
    out = {}
    for key in METRICS:
        a = np.array([per_seed[model][s][key] for s in SEEDS])
        b = np.array([per_seed[base][s][key] for s in SEEDS])
        if key in RATIO_METRICS:
            log = paired_seed_interval(np.log(a / b))
            out[key] = {
                "scale": "ratio model / data only (geometric mean over seeds, interval from log ratios)",
                "per_seed": (a / b).tolist(),
                "ratio": float(np.exp(log["mean"])),
                "ci_low": float(np.exp(log["ci_low"])),
                "ci_high": float(np.exp(log["ci_high"])),
                "t_p_value": log["t_p_value"],
                "seeds_lower": log["seeds_negative"],
            }
        else:
            diff = paired_seed_interval(a - b)
            diff["relative_change"] = float(diff["mean"] / b.mean())
            diff["scale"] = "difference model - data only"
            out[key] = diff
    out["rel_l2_test_sample_bootstrap"] = {
        f"seed{s}": paired_bootstrap(per_sample[model][s], per_sample[base][s]) for s in SEEDS
    }
    return out


def plot(per_seed: dict, comparisons: dict, path) -> None:
    import matplotlib.pyplot as plt

    from pino_darcy.plotting import DATA_ONLY, INK, MUTED, SERIES

    style = {"data only": DATA_ONLY, "shipped 0.1": SERIES["shipped"], "consistent 0.1": SERIES["consistent"]}
    names = list(SETTINGS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, key, label, log in (
        (axes[0], "rel_l2", "test relative L2 error", False),
        (axes[1], "exact_residual", "exact residual, mean |r| (physical units)", True),
    ):
        for s, offset in zip(SEEDS, (-0.12, 0.0, 0.12), strict=True):
            ys = [per_seed[n][s][key] for n in names]
            ax.plot(np.arange(3) + offset, ys, color=MUTED, lw=0.8, alpha=0.6, zorder=1)
            for i, n in enumerate(names):
                ax.scatter(i + offset, ys[i], color=style[n]["color"], marker=style[n].get("marker", "D"), zorder=2, s=34)
        ax.set_xticks(range(3), names)
        ax.set_ylabel(label)
        if log:
            ax.set_yscale("log")
        ax.set_title("per seed (0, 1, 2 left to right; lines join one seed)" if key == "rel_l2" else "exact operator, coeff")

    ax = axes[2]
    for i, n in enumerate(names[1:]):
        c = comparisons[n]["rel_l2"]
        ax.errorbar(i, c["mean"], yerr=[[c["mean"] - c["ci_low"]], [c["ci_high"] - c["mean"]]], color=style[n]["color"], marker=style[n]["marker"], capsize=5)
        ax.scatter([i + 0.15] * 3, c["per_seed"], color=style[n]["color"], marker=style[n]["marker"], s=18, alpha=0.6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(2), names[1:])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("paired error difference to data only")
    ax.set_title("mean over seeds, 95 % t interval (df = 2)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    per_seed, per_sample, manifest = {}, {}, []
    for name, prefix in SETTINGS.items():
        per_seed[name], per_sample[name] = {}, {}
        for s in SEEDS:
            per_seed[name][s], per_sample[name][s] = run_metrics(s, prefix)
            manifest.append(manifest_entry(s, prefix))
    training_hashes = {m["training_config_sha256"] for m in manifest}
    assert len(training_hashes) == 1, "the runs differ in a training setting other than residual, weight and seed"
    write_json(
        ROOT / "results" / "local_seeds" / "run_manifest.json",
        {
            "hardware": hardware(),
            "shared_training_config_sha256": training_hashes.pop(),
            "excluded_from_training_hash": list(TRAINING_KEYS_EXCLUDED),
            "runs": manifest,
        },
    )

    comparisons = {n: compare(per_seed, per_sample, n) for n in list(SETTINGS)[1:]}
    spread = {
        n: {k: {"mean": float(np.mean(v := [per_seed[n][s][k] for s in SEEDS])), "std_over_seeds": float(np.std(v, ddof=1))} for k in METRICS}
        for n in SETTINGS
    }
    reference = load_json(ROOT / "results" / "audit" / "local_seeds_solver_residual.json")["reference"]["solver_residual"]
    result = {
        "design": "G1 of docs/EXPERIMENT_GAPS.md: seeds 0, 1, 2; data only, shipped 0.1, consistent 0.1; local config (15 epochs, 102 training samples)",
        "metrics": METRICS,
        "reference_exact_residual": reference,
        "per_seed": {n: {f"seed{s}": per_seed[n][s] for s in SEEDS} for n in SETTINGS},
        "seed_spread": spread,
        "paired_against_data_only": comparisons,
        "notes": [
            "Seeds are the independent units of the t intervals (df = 2); the t interval for three seeds is 2.48 times its standard error on each side.",
            "The exact two-sided sign test cannot fall below 0.25 with three seeds.",
            "The test-sample bootstrap intervals hold each trained model fixed; they do not include seed variability.",
        ],
    }
    write_json(ROOT / "results" / "audit" / "seed_study.json", result)
    plot(per_seed, comparisons, ROOT / "figures" / "seed_study.png")

    for n in SETTINGS:
        rows = "  ".join(f"s{s} {per_seed[n][s]['rel_l2']:.5f}/{per_seed[n][s]['exact_residual']:.3f}" for s in SEEDS)
        print(f"{n:15s} rel L2 / exact residual: {rows}")
    for n, c in comparisons.items():
        e, r = c["rel_l2"], c["exact_residual"]
        print(
            f"{n:15s} delta rel L2 {e['mean']:+.5f} [{e['ci_low']:+.5f}, {e['ci_high']:+.5f}] ({e['relative_change']:+.1%}), "
            f"exact residual x{r['ratio']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )


if __name__ == "__main__":
    main()
