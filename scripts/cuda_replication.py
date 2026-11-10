"""Analysis of the CUDA replication of the G2 primary comparison (docs/CUDA_REPLICATION_PLAN.md).

    python scripts/cuda_replication.py
    python scripts/cuda_replication.py --results-root <dir> --out <file> --figure ""   # e.g. inside the Kaggle session

Reads the published CUDA studies ``full_cuda`` (seeds 0 to 2), ``full_cuda_more`` (seeds 3, 4) and,
if present, ``full_cuda_sweep``, their audit files ``audit/<study>_solver_residual.json`` (and
``_generalisation.json`` when present), and the MPS studies ``full_mps``, ``full_mps_more``,
``full_mps_sweep`` for comparison. Nothing is written into the studies. Writes
``results/audit/cuda_replication.json`` and ``figures/cuda_replication.png`` by default.

Statistics are those of ``scripts/full_study.py`` (the same functions): paired by seed, Student-t
95 % interval over seeds, relative to the mean data-only error. The primary criterion is fixed in
the plan: the upper endpoint of the five-seed relative interval of ``consistent`` 0.1 minus data
only must be below -5 %.
"""

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ROOT
from full_study import compare, run_record
from pino_darcy.metrics import paired_seed_interval
from pino_darcy.provenance import collect_metadata, write_json

PLAN = "docs/CUDA_REPLICATION_PLAN.md"
THRESHOLD = -0.05  # the upper endpoint of the relative interval must lie below this
PRIMARY_SEEDS, MORE_SEEDS = (0, 1, 2), (3, 4)
STAGE1 = {"data only": "dataonly", "shipped 0.1": "shipped_w0.1", "consistent 0.1": "consistent_w0.1"}
SWEEP = {f"{r} {w}": f"{r}_w{w}" for r in ("shipped", "consistent") for w in ("0.001", "0.01", "1")}
PRIMARY = ("consistent 0.1", "data only")
SECONDARY = ("shipped 0.1", "data only")
DESCRIPTIVE = ("rel_l2", "rel_l2_near_jumps", "rel_l2_constant_k", "exact_residual", "cellgrid_residual", "train_rel_l2")


def criterion(ci_relative_high: float | None, n_seeds: int, required: int = 5) -> str:
    """The pre-registered verdict from the upper end of the relative interval (a fraction, not a percentage)."""
    if ci_relative_high is None or n_seeds < required:
        return "not evaluable"
    if ci_relative_high < THRESHOLD:
        return "replicated on CUDA"
    if ci_relative_high < 0:
        return "not replicated by the criterion; the direction agrees"
    return "not replicated"


def available(results: Path, study: str) -> bool:
    return (results / study / "summary.json").exists() and (results / "audit" / f"{study}_solver_residual.json").exists()


def load_device(results: Path, prefix: str) -> tuple[dict, dict]:
    """Records ``{label: {seed: run_record}}`` of the stage-1 settings (seeds 0 to 4) and the sweep (seeds 0 to 2)."""
    stages = {"primary": prefix, "more": f"{prefix}_more", "sweep": f"{prefix}_sweep"}
    status = {stage: (study, "available" if available(results, study) else "missing") for stage, study in stages.items()}
    records = {}
    for label, run in STAGE1.items():
        records[label] = {}
        for stage, seeds in (("primary", PRIMARY_SEEDS), ("more", MORE_SEEDS)):
            if status[stage][1] == "available":
                for s in seeds:
                    records[label][s] = run_record(stages[stage], f"{run}_seed{s}", results)
    if status["sweep"][1] == "available":
        for label, run in SWEEP.items():
            records[label] = {s: run_record(stages["sweep"], f"{run}_seed{s}", results) for s in PRIMARY_SEEDS}
        for s in PRIMARY_SEEDS:  # the sweep's control is the data-only run of the same seed in stage 1
            if s not in records["data only"]:
                status["sweep"] = (status["sweep"][0], "available, but stage 1 missing")
    return records, status


def seeds_of(records: dict, *labels: str) -> tuple:
    common = set.intersection(*(set(records.get(label, {})) for label in labels))
    return tuple(sorted(common))


def comparison(records: dict, setting: str, control: str, seeds: tuple) -> dict | None:
    if not seeds:
        return None
    out = compare(records, setting, control, seeds)
    out["seeds"] = list(seeds)
    return out


def device_differences(cuda: dict, mps: dict) -> dict:
    """CUDA minus MPS, paired by seed: per setting, and for the paired effects of the primary and secondary comparison."""
    out = {"per_setting": {}, "effects": {}}
    for label in cuda:
        seeds = tuple(s for s in sorted(cuda[label]) if s in mps.get(label, {}))
        if len(seeds) < 2:
            continue
        out["per_setting"][label] = {}
        for key in DESCRIPTIVE:
            a = [cuda[label][s]["metrics"][key] for s in seeds]
            b = [mps[label][s]["metrics"][key] for s in seeds]
            if None in a or None in b:
                continue
            d = paired_seed_interval(np.array(a) - np.array(b))
            d.update(
                {"seeds": list(seeds), "cuda_mean": float(np.mean(a)), "mps_mean": float(np.mean(b)), "relative_to_mps": float(np.mean(a) / np.mean(b) - 1)}
            )
            out["per_setting"][label][key] = d
    for setting, control in (PRIMARY, SECONDARY):
        seeds = tuple(s for s in seeds_of(cuda, setting, control) if s in seeds_of(mps, setting, control))
        if len(seeds) < 2:
            continue
        eff = {}
        for name, rec in (("cuda", cuda), ("mps", mps)):
            eff[name] = np.array([rec[setting][s]["metrics"]["rel_l2"] - rec[control][s]["metrics"]["rel_l2"] for s in seeds])
        d = paired_seed_interval(eff["cuda"] - eff["mps"])
        d.update({"seeds": list(seeds), "cuda_effect_per_seed": eff["cuda"].tolist(), "mps_effect_per_seed": eff["mps"].tolist()})
        out["effects"][f"{setting} - {control}"] = d
    return out


def summarise(records: dict) -> dict:
    return {
        label: {
            "seeds": sorted(r),
            "per_seed": {f"seed{s}": r[s]["metrics"] for s in sorted(r)},
            "mean": {k: float(np.mean(v)) for k in DESCRIPTIVE if r and None not in (v := [r[s]["metrics"][k] for s in r])},
            "std_over_seeds": {k: float(np.std(v, ddof=1)) for k in DESCRIPTIVE if len(r) > 1 and None not in (v := [r[s]["metrics"][k] for s in r])},
            "cost": {k: float(np.mean([r[s]["cost"][k] for s in r])) for k in ("train_seconds", "optimisation_ms_per_step")} if r else {},
        }
        for label, r in records.items()
        if r
    }


def analyse(results: Path, cuda_prefix: str = "full_cuda", mps_prefix: str = "full_mps") -> dict:
    cuda, cuda_status = load_device(results, cuda_prefix)
    mps, mps_status = load_device(results, mps_prefix)
    five = seeds_of(cuda, *PRIMARY)
    primary = comparison(cuda, *PRIMARY, five)
    rel = primary["rel_l2"] if primary else None
    verdict = criterion(rel["ci_relative"][1] if rel else None, len(five))
    result = {
        "plan": PLAN,
        "criterion": "upper endpoint of the five-seed 95 % t interval of (consistent 0.1 - data only) / mean data-only error below -5 %",
        "studies": {"cuda": cuda_status, "mps": mps_status},
        "primary": {
            "comparison": "consistent 0.1 - data only",
            "seeds": list(five),
            "verdict": verdict,
            "relative_change": rel["relative_change"] if rel else None,
            "ci_relative": rel["ci_relative"] if rel else None,
            "detail": primary,
        },
        "primary_seeds_0_to_2": comparison(cuda, *PRIMARY, tuple(s for s in five if s in PRIMARY_SEEDS)),
        "secondary": {
            "shipped 0.1 - data only": comparison(cuda, *SECONDARY, seeds_of(cuda, *SECONDARY)),
            **{
                f"{label} - data only": comparison(cuda, label, "data only", tuple(s for s in seeds_of(cuda, label, "data only") if s in PRIMARY_SEEDS))
                for label in SWEEP
                if label in cuda
            },
        },
        "mps_reference": {
            "consistent 0.1 - data only": comparison(mps, *PRIMARY, seeds_of(mps, *PRIMARY)),
            "shipped 0.1 - data only": comparison(mps, *SECONDARY, seeds_of(mps, *SECONDARY)),
        },
        "per_setting": {"cuda": summarise(cuda), "mps": summarise(mps)},
        "cuda_minus_mps": device_differences(cuda, mps),
        "notes": [
            "Seeds are the independent units; the test-sample bootstrap intervals in 'detail' hold one pair of models fixed.",
            "CUDA and MPS are not expected to agree numerically; on CUDA the finite-difference residual runs on the GPU, on MPS it ran on the CPU.",
            "Relative intervals divide the interval of the mean difference by the mean data-only error of the same device and seeds.",
            "cuda_minus_mps is descriptive; no equivalence of the devices is claimed from it.",
        ],
    }
    return result


def plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colours = {"cuda": "#2a6fdb", "mps": "#555555"}
    for device, offset in (("cuda", -0.12), ("mps", 0.12)):
        per = result["per_setting"][device]
        for i, label in enumerate(STAGE1):
            if label not in per:
                continue
            vals = [v["rel_l2"] for v in per[label]["per_seed"].values()]
            axes[0].scatter([i + offset] * len(vals), vals, color=colours[device], s=18, label=device.upper() if i == 0 else None)
            axes[0].hlines(np.mean(vals), i + offset - 0.08, i + offset + 0.08, color=colours[device])
    axes[0].set_xticks(range(len(STAGE1)), list(STAGE1))
    axes[0].set_ylabel("test relative L2 error")
    axes[0].set_title("per seed (dots) and mean (bar)", fontsize=9)
    axes[0].legend(fontsize=8)
    rows = [("CUDA", result["primary"]["detail"]), ("MPS", result["mps_reference"]["consistent 0.1 - data only"])]
    for y, (name, c) in enumerate(rows):
        if not c:
            continue
        e = c["rel_l2"]
        lo, hi = (100 * x for x in e["ci_relative"])
        axes[1].errorbar(
            100 * e["relative_change"],
            y,
            xerr=[[100 * e["relative_change"] - lo], [hi - 100 * e["relative_change"]]],
            fmt="o",
            capsize=4,
            color=colours[name.lower()],
        )
        axes[1].scatter([100 * d / e["control_mean"] for d in e["per_seed"]], [y + 0.15] * len(e["per_seed"]), s=10, color=colours[name.lower()], alpha=0.6)
    axes[1].axvline(100 * THRESHOLD, color="#b03030", ls="--", lw=1, label="criterion: upper end below -5 %")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[1].set_ylim(-0.6, len(rows) - 0.4)
    axes[1].set_xlabel("consistent 0.1 - data only, % of the data-only error")
    axes[1].set_title(f"five seeds, 95 % t interval; verdict: {result['primary']['verdict']}", fontsize=9)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "audit" / "cuda_replication.json")
    parser.add_argument("--figure", default=str(ROOT / "figures" / "cuda_replication.png"), help='"" for no figure')
    parser.add_argument("--cuda-prefix", default="full_cuda", help="study prefix of the replication (default full_cuda)")
    parser.add_argument("--mps-prefix", default="full_mps")
    args = parser.parse_args()
    if args.cuda_prefix.startswith(args.mps_prefix) and args.out.resolve() == (ROOT / "results" / "audit" / "cuda_replication.json").resolve():
        raise SystemExit("a dry run on the MPS studies must write elsewhere: pass --out")
    result = analyse(args.results_root, args.cuda_prefix, args.mps_prefix)
    result["environment"] = collect_metadata(__import__("torch").device("cpu"))
    write_json(args.out, result)
    p = result["primary"]
    if p["relative_change"] is None:
        print(f"primary: {p['verdict']} (CUDA studies: {result['studies']['cuda']})")
    else:
        lo, hi = p["ci_relative"]
        print(
            f"primary, seeds {p['seeds']}: consistent 0.1 - data only {100 * p['relative_change']:+.1f} % [{100 * lo:+.1f}, {100 * hi:+.1f}] -> {p['verdict']}"
        )
    for name, c in result["cuda_minus_mps"]["effects"].items():
        print(f"CUDA - MPS, effect {name}: {c['mean']:+.5f} [{c['ci_low']:+.5f}, {c['ci_high']:+.5f}] over seeds {c['seeds']}")
    if args.figure and result["primary"]["detail"]:
        plot(result, Path(args.figure))
    print("written", args.out)


if __name__ == "__main__":
    main()
