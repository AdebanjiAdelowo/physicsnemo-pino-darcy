"""Analysis of the 50-epoch studies: G2 (FULL_BUDGET_PLAN.md) and G3 to G6 (OPERATOR_AND_LABELS_PLAN.md).

    python scripts/full_study.py

Reads, and does not change, the published records ``results/<study>/`` and the audit files
``results/audit/<study>_solver_residual.json`` and ``<study>_generalisation.json`` of the studies
``full_mps``, ``full_mps_sweep``, ``full_mps_operator``, ``full_mps_solver_w1``, ``labels25``,
``labels50``, ``labels25_unlabelled``, ``labels50_unlabelled``, and the 15-epoch records of the
seed study, plus ``full_mps_more`` (G2 stage 3, seeds 3 and 4). A study that has not been published is reported as pending and left out. Writes
``results/audit/full_study.json`` and the ``figures/full_*.png`` figures.

Statistics follow the plans: the training seed is the unit of replication; a setting is compared
with its control of the same seed on the same 256 test samples; Student-t 95 % interval over
seeds; Holm correction within each family of comparisons; the pre-stated reading of an interval
(improvement, degradation, no material effect within +-5 % of the control mean, inconclusive).
Test-sample bootstrap intervals are reported per seed for completeness and are not evidence about
a setting.
"""

import csv
import json

import numpy as np

from _bootstrap import ROOT
from pino_darcy.metrics import paired_bootstrap, paired_seed_interval
from pino_darcy.provenance import collect_metadata, write_json

RESULTS = ROOT / "results"
AUDIT = RESULTS / "audit"
FIGURES = ROOT / "figures"
SEEDS = (0, 1, 2)
STAGE3, STAGE3_SEEDS = "full_mps_more", (3, 4)
MARGIN = 0.05  # "no material effect" band, fraction of the control mean (both plans)

# label -> (study, run prefix); the run is f"{prefix}_seed{seed}"
SETTINGS = {
    "data only": ("full_mps", "dataonly"),
    "shipped 0.1": ("full_mps", "shipped_w0.1"),
    "consistent 0.1": ("full_mps", "consistent_w0.1"),
    **{f"{r} {w}": ("full_mps_sweep", f"{r}_w{w}") for r in ("shipped", "consistent") for w in ("0.001", "0.01", "1")},
    "consistent_coeff 0.1": ("full_mps_operator", "consistent_coeff_w0.1"),
    "solver_kcoeff 0.1": ("full_mps_operator", "solver_kcoeff_w0.1"),
    "solver 0.1": ("full_mps_operator", "solver_w0.1"),
    "solver 1": ("full_mps_solver_w1", "solver_w1"),
    **{f"N={n} data only": (f"labels{n}", "dataonly") for n in (25, 50)},
    **{f"N={n} solver 0.1": (f"labels{n}", "solver_w0.1") for n in (25, 50)},
    **{f"N={n} solver 0.1 + unlabelled": (f"labels{n}_unlabelled", "solver_w0.1") for n in (25, 50)},
}
FIFTEEN_EPOCHS = {"data only": "dataonly", "shipped 0.1": "shipped_w0.1", "consistent 0.1": "consistent_w0.1"}

# families of comparisons: (setting, control); Holm within a family, on the overall error
FAMILIES = {
    "G2 primary": [("consistent 0.1", "data only")],
    "G2 secondary": [("shipped 0.1", "data only")] + [(f"{r} {w}", "data only") for r in ("shipped", "consistent") for w in ("0.001", "0.01", "1")],
    "G4 primary": [("solver 0.1", "data only")],
    "G3/G4 secondary": [
        ("consistent_coeff 0.1", "data only"),
        ("solver_kcoeff 0.1", "data only"),
        ("solver 1", "data only"),
        ("consistent_coeff 0.1", "consistent 0.1"),
        ("solver 0.1", "consistent 0.1"),
        ("solver_kcoeff 0.1", "consistent 0.1"),
        ("solver 0.1", "solver_kcoeff 0.1"),
        ("solver 0.1", "consistent_coeff 0.1"),
        ("solver 1", "solver 0.1"),
    ],
    "G5/G6 primary": [(f"N={n} solver 0.1{u}", f"N={n} data only") for n in (25, 50) for u in ("", " + unlabelled")],
    "G5/G6 secondary": [(f"N={n} solver 0.1 + unlabelled", f"N={n} solver 0.1") for n in (25, 50)]
    + [("data only", "N=50 data only"), ("N=50 data only", "N=25 data only")],
}
METRICS = {
    "rel_l2": "test relative L2 error",
    "rel_l2_constant_k": "relative L2 error on cells of locally constant Kcoeff",
    "rel_l2_near_jumps": "relative L2 error on the other cells (near jumps)",
    "exact_residual": "mean |r| under the generator's operator with coeff (physical units, f = 1)",
    "exact_residual_interface": "same, nodes within two cells of a jump of coeff",
    "exact_residual_other": "same, the other nodes",
    "cellgrid_residual": "mean |r| of the cell-grid consistent residual (Kcoeff, normalised units)",
    "train_rel_l2": "relative L2 error on the labelled training samples",
}
RATIO = {k for k in METRICS if "residual" in k}


def load(path):
    return json.loads(path.read_text())


def available(study: str) -> bool:
    return (RESULTS / study / "summary.json").exists() and (AUDIT / f"{study}_solver_residual.json").exists()


def history(study: str, run: str) -> dict:
    with open(RESULTS / study / run / "history.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def run_record(study: str, run: str) -> dict:
    ev = load(RESULTS / study / run / "eval_metrics.json")
    tr = load(RESULTS / study / run / "train_metrics.json")
    audit = load(AUDIT / f"{study}_solver_residual.json")["runs"][run]
    stored, recomputed = ev["test_rel_l2"]["mean"], audit["rel_l2_mean"]
    assert abs(stored - recomputed) <= 1e-5 * stored, f"{study}/{run}: stored and recomputed test errors differ"
    gen_path = AUDIT / f"{study}_generalisation.json"
    gen = load(gen_path)["runs"].get(run) if gen_path.exists() else None
    h = history(study, run)
    steps_per_epoch = tr["optimiser_steps"] / h["epoch"][-1]
    return {
        "metrics": {
            "rel_l2": stored,
            "rel_l2_constant_k": audit["rel_l2_by_region"]["constant_k_cells"],
            "rel_l2_near_jumps": audit["rel_l2_by_region"]["other_cells"],
            "exact_residual": audit["solver_residual"]["abs_mean"],
            "exact_residual_interface": audit["solver_residual"]["abs_mean_interface_nodes"],
            "exact_residual_other": audit["solver_residual"]["abs_mean_other_nodes"],
            "cellgrid_residual": ev["test_residual_consistent"]["l1"],
            "train_rel_l2": gen["train"] if gen else None,
        },
        "per_sample": ev["test_rel_l2_per_sample"],
        "test_sample_std": ev["test_rel_l2"]["std"],
        "cost": {
            "optimiser_steps": tr["optimiser_steps"],
            "train_seconds": tr["train_seconds"],
            "optimisation_ms_per_step": tr["optimisation_ms_per_step"],
            "inference_ms_per_sample": ev["inference_ms_per_sample"],
        },
        "convergence": {
            "final_validation_rel_l2": float(h["validation_rel_l2"][-1]),
            "min_validation_rel_l2": float(np.nanmin(h["validation_rel_l2"])),
            "epoch_of_min": int(h["epoch"][int(np.nanargmin(h["validation_rel_l2"]))]),
            # relative change of the validation error over the last ~10 % of the steps
            "last_tenth_relative_change": float(
                h["validation_rel_l2"][-1] / h["validation_rel_l2"][np.searchsorted(h["epoch"] * steps_per_epoch, 0.9 * tr["optimiser_steps"])] - 1
            ),
        },
        "history": h,
        "steps_per_epoch": steps_per_epoch,
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def reading(ci_low: float, ci_high: float, control_mean: float) -> str:
    """The pre-stated reading of an interval of an error difference."""
    if ci_high < 0:
        return "improvement"
    if ci_low > 0:
        return "degradation"
    if -MARGIN * control_mean <= ci_low and ci_high <= MARGIN * control_mean:
        return "no material effect"
    return "inconclusive"


def compare(records: dict, setting: str, control: str, seeds=SEEDS) -> dict:
    out = {}
    for key in METRICS:
        a = [records[setting][s]["metrics"][key] for s in seeds]
        b = [records[control][s]["metrics"][key] for s in seeds]
        if None in a or None in b:
            continue
        a, b = np.array(a), np.array(b)
        if key in RATIO:
            log = paired_seed_interval(np.log(a / b))
            out[key] = {
                "scale": "ratio setting / control, geometric mean over seeds",
                "per_seed": (a / b).tolist(),
                "ratio": float(np.exp(log["mean"])),
                "ci_low": float(np.exp(log["ci_low"])),
                "ci_high": float(np.exp(log["ci_high"])),
                "t_p_value": log["t_p_value"],
                "seeds_lower": log["seeds_negative"],
            }
        else:
            d = paired_seed_interval(a - b)
            d["control_mean"] = float(b.mean())
            d["relative_change"] = float(d["mean"] / b.mean())
            d["ci_relative"] = [d["ci_low"] / b.mean(), d["ci_high"] / b.mean()]
            # standardised paired effect (d_z); with three to five seeds a very rough number
            d["effect_size_dz"] = float(d["mean"] / d["std"]) if d["std"] else None
            # the same difference in units of the control's seed-to-seed standard deviation
            sd = float(np.std(b, ddof=1))
            d["in_control_seed_sd"] = float(d["mean"] / sd) if sd > 0 else None
            d["reading"] = reading(d["ci_low"], d["ci_high"], b.mean())
            out[key] = d
    out["rel_l2_test_sample_bootstrap"] = {f"seed{s}": paired_bootstrap(records[setting][s]["per_sample"], records[control][s]["per_sample"]) for s in seeds}
    return out


def fifteen_epoch_record(label: str, seed: int) -> dict | None:
    study = "local" if seed == 0 else "local_seeds"
    run = f"{FIFTEEN_EPOCHS[label]}_seed{seed}"
    path = RESULTS / study / run / "eval_metrics.json"
    if not path.exists():
        return None
    audit = load(AUDIT / f"{study}_solver_residual.json")["runs"][run]
    return {
        "rel_l2": load(path)["test_rel_l2"]["mean"],
        "rel_l2_near_jumps": audit["rel_l2_by_region"]["other_cells"],
        "exact_residual": audit["solver_residual"]["abs_mean"],
    }


def main() -> None:
    studies = sorted({s for s, _ in SETTINGS.values()})
    status = {s: ("available" if available(s) else "pending") for s in studies}
    records = {}
    for label, (study, prefix) in SETTINGS.items():
        if status[study] == "available":
            records[label] = {s: run_record(study, f"{prefix}_seed{s}") for s in SEEDS}

    families = {}
    for family, pairs in FAMILIES.items():
        done = [(a, b) for a, b in pairs if a in records and b in records]
        comparisons = {f"{a} - {b}": compare(records, a, b) for a, b in done}
        p = [c["rel_l2"]["t_p_value"] for c in comparisons.values()]
        for c, adj in zip(comparisons.values(), holm(p) if p else [], strict=True):
            c["rel_l2"]["holm_p_value"] = adj
            c["rel_l2"]["holm_family_size"] = len(p)
        families[family] = {"comparisons": comparisons, "pending": [f"{a} - {b}" for a, b in pairs if (a, b) not in done]}

    # G2 stage 3 (FULL_BUDGET_PLAN.md): seeds 3 and 4 of the stage-1 settings, reported after n = 3
    stage3 = None
    if "consistent 0.1" in records and available(STAGE3):
        seeds5 = SEEDS + STAGE3_SEEDS
        five = {label: {**records[label], **{s: run_record(STAGE3, f"{SETTINGS[label][1]}_seed{s}") for s in STAGE3_SEEDS}} for label in FIFTEEN_EPOCHS}
        stage3 = {
            "seeds": list(seeds5),
            "comparisons": {f"{a} - {b}": compare(five, a, b, seeds5) for a, b in (FAMILIES["G2 primary"] + FAMILIES["G2 secondary"][:1])},
        }
        stage3["per_setting_mean"] = {
            label: {k: float(np.mean([five[label][s]["metrics"][k] for s in seeds5])) for k in ("rel_l2", "rel_l2_near_jumps")} for label in five
        }

    # 2 x 2: operator (cell grid, generator) x coefficient (Kcoeff, coeff); contrasts per seed
    factorial = None
    cells = {
        ("cell", "Kcoeff"): "consistent 0.1",
        ("cell", "coeff"): "consistent_coeff 0.1",
        ("solver", "Kcoeff"): "solver_kcoeff 0.1",
        ("solver", "coeff"): "solver 0.1",
    }
    if all(v in records for v in cells.values()):
        factorial = {}
        for key in ("rel_l2", "rel_l2_near_jumps", "rel_l2_constant_k"):
            v = {c: np.array([records[name][s]["metrics"][key] for s in SEEDS]) for c, name in cells.items()}
            contrasts = {
                "coefficient effect at the cell-grid operator (coeff - Kcoeff)": v[("cell", "coeff")] - v[("cell", "Kcoeff")],
                "coefficient effect at the generator operator (coeff - Kcoeff)": v[("solver", "coeff")] - v[("solver", "Kcoeff")],
                "operator effect with Kcoeff (generator - cell grid)": v[("solver", "Kcoeff")] - v[("cell", "Kcoeff")],
                "operator effect with coeff (generator - cell grid)": v[("solver", "coeff")] - v[("cell", "coeff")],
                "interaction": v[("solver", "coeff")] - v[("solver", "Kcoeff")] - v[("cell", "coeff")] + v[("cell", "Kcoeff")],
            }
            factorial[key] = {name: paired_seed_interval(d) for name, d in contrasts.items()}

    epochs = {}
    for label in FIFTEEN_EPOCHS:
        if label not in records:
            continue
        rows = {}
        for s in SEEDS:
            old = fifteen_epoch_record(label, s)
            if old:
                new = records[label][s]["metrics"]
                rows[f"seed{s}"] = {k: {"15_epochs": old[k], "50_epochs": new[k]} for k in old}
        epochs[label] = rows
    # the primary difference at 15 and at 50 epochs, same seeds
    if "consistent 0.1" in records and all(fifteen_epoch_record(x, s) for x in FIFTEEN_EPOCHS for s in SEEDS):
        for label in ("consistent 0.1", "shipped 0.1"):
            d15 = [fifteen_epoch_record(label, s)["rel_l2"] - fifteen_epoch_record("data only", s)["rel_l2"] for s in SEEDS]
            epochs[f"{label} - data only, 15 epochs"] = paired_seed_interval(d15)

    per_setting = {
        label: {
            "study": SETTINGS[label][0],
            "per_seed": {f"seed{s}": r[s]["metrics"] for s in SEEDS},
            "mean": {k: float(np.mean(v)) for k in METRICS if None not in (v := [r[s]["metrics"][k] for s in SEEDS])},
            "std_over_seeds": {k: float(np.std(v, ddof=1)) for k in METRICS if None not in (v := [r[s]["metrics"][k] for s in SEEDS])},
            "mean_test_sample_std": float(np.mean([r[s]["test_sample_std"] for s in SEEDS])),
            "cost": {k: float(np.mean([r[s]["cost"][k] for s in SEEDS])) for k in records[label][0]["cost"]},
            "convergence": {f"seed{s}": r[s]["convergence"] for s in SEEDS},
        }
        for label, r in records.items()
    }
    result = {
        "environment": collect_metadata(__import__("torch").device("cpu")),
        "plans": ["docs/FULL_BUDGET_PLAN.md", "docs/OPERATOR_AND_LABELS_PLAN.md"],
        "device": "FNO on Apple MPS, cell-grid residual on CPU (not CUDA)",
        "studies": status,
        "metrics": METRICS,
        "per_setting": per_setting,
        "families": families,
        "factorial_2x2": factorial,
        "fifteen_vs_fifty_epochs": epochs,
        "g2_stage3_n5": stage3 if stage3 else ("pending" if "consistent 0.1" in records else None),
        "notes": [
            "Seeds are the independent units. With three seeds (df = 2) the 95 % t interval is 4.30 standard errors "
            "(2.48 standard deviations of the paired differences) on each side; with five seeds (stage 3, df = 4) 2.78 standard errors.",
            "std_over_seeds is variability across independently trained models; mean_test_sample_std is the spread of the "
            "per-sample error over the 256 test samples for one model. They are different quantities.",
            "Holm adjustment within each family, on the overall error only. Only the G2 and G4 primary and the G5/G6 primary family are confirmatory.",
            "effect_size_dz = mean / sd of the paired differences over three seeds; unstable, reported for completeness.",
            "The test-sample bootstrap intervals hold the trained models fixed and do not include seed variability.",
        ],
    }
    write_json(AUDIT / "full_study.json", result)
    print("studies:", status)
    for family, f in families.items():
        for name, c in f["comparisons"].items():
            e = c["rel_l2"]
            print(
                f"[{family}] {name}: {e['mean']:+.5f} [{e['ci_low']:+.5f}, {e['ci_high']:+.5f}] ({e['relative_change']:+.1%}), "
                f"p={e['t_p_value']:.3f}, Holm {e['holm_p_value']:.3f} -> {e['reading']}"
            )
    if stage3:
        for name, c in stage3["comparisons"].items():
            e = c["rel_l2"]
            print(
                f"[G2 stage 3, n = 5] {name}: {e['mean']:+.5f} [{e['ci_low']:+.5f}, {e['ci_high']:+.5f}] "
                f"({e['relative_change']:+.1%}), p={e['t_p_value']:.3f} -> {e['reading']}"
            )
    import plots_full

    plots_full.plot_all(records, families, factorial, epochs, FIGURES)


if __name__ == "__main__":
    main()
