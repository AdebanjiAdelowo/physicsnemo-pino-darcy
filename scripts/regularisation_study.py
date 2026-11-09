"""Selection and analysis of the regularisation controls (docs/REGULARISATION_PLAN.md).

    python scripts/regularisation_study.py select     # validation only; writes results/audit/regularisation_selection.json
    python scripts/regularisation_study.py analyse    # needs a final selection; writes results/audit/regularisation_study.json

``select`` reads only ``<study>_variants_validation.json`` of the weight-decay grid and fixes, from
the mean validation error over the training seeds, the weight decay of the "weight decay" arms. If
the selected value is the largest of the grid and the grid can still be extended, the selection
is marked ``extend`` and the analysis refuses to run until the extension is trained and selected.

``analyse`` reads the test files of the grid studies and of the physics-informed comparison runs
(``consistent_w0.1`` of ``full_cuda`` and ``full_cuda_more``), builds the arms of the two budgets
and computes paired differences over seeds with 95 % t intervals, the pre-registered primary
comparison and the Holm-adjusted secondary family. Nothing is trained or evaluated here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import ROOT
from full_study import holm
from pino_darcy.config import load_config
from pino_darcy.metrics import paired_seed_interval
from pino_darcy.provenance import collect_metadata, write_json

PLAN = "docs/REGULARISATION_PLAN.md"
MARGIN = 0.05  # reading rule: a difference within +-5 % of the comparison mean is "no material difference"
GRID = {0.0: "reg_cuda_wd0", 0.001: "reg_cuda_wd1e-3", 0.01: "reg_cuda_wd1e-2", 0.1: "reg_cuda_wd1e-1", 1.0: "reg_cuda_wd1e0", 10.0: "reg_cuda_wd1e1"}
EXTENSION = {100.0: "reg_cuda_wd1e2"}  # trained only if the boundary rule asks for it; one extension at most
PINO_STUDIES = ("full_cuda", "full_cuda_more")
PINO_RUN = "consistent_w0.1"
SEEDS = (0, 1, 2, 3, 4)
AUDIT = ROOT / "results" / "audit"
CONTROLS = ("WD", "ES", "VB", "WD+ES", "WD+VB")
PRIMARY = ("PINO", "WD+VB")  # budget A; the plan states why WD+VB and not WD+ES
SECONDARY = (
    [("A", c, "base") for c in CONTROLS]
    + [("A", "PINO", c) for c in ("base",) + CONTROLS if (c != PRIMARY[1])]
    + [("B", c, "base") for c in CONTROLS]
    + [("B", "PINO", c) for c in ("base",) + CONTROLS]
)


def labels(study: str) -> dict:
    """Variant labels of the budgets, from the configuration of a grid study.

    A is the first snapshot epoch (equal steps), B the final epoch (matched compute, official); every
    other snapshot epoch between them is a sensitivity budget ``B<epoch>`` (amendment: B51), analysed
    with the same rules and reported separately from the Holm family.
    """
    cfg = load_config(study)
    snapshots = sorted(int(e) for e in cfg.snapshots.epochs)
    a, b = snapshots[0], int(cfg.max_epochs)
    out = {"A_epochs": a, "B_epochs": b, "sensitivity": [f"B{e}" for e in snapshots[1:] if a < e < b]}
    for budget, epoch, fixed in [("A", a, f"epoch{a:03d}"), ("B", b, "final")] + [(f"B{e}", e, f"epoch{e:03d}") for e in snapshots[1:] if a < e < b]:
        out[budget] = fixed
        out[f"ES_{budget}"] = f"es_cap{epoch:03d}"
        out[f"VB_{budget}"] = f"vb_cap{epoch:03d}"
    return out


def checkpoint_selections(record: dict, lab: dict, seeds=SEEDS) -> dict:
    """Validation-only record of what every rule selected in every run (epoch, stop, compute consumed)."""
    out = {}
    for budget in ["A", "B", *lab["sensitivity"]]:
        for rule in ("ES", "VB"):
            label = lab[f"{rule}_{budget}"]
            runs = {s: record["runs"][f"dataonly_seed{s}"][label] for s in seeds}
            out[label] = {
                "budget": budget,
                "selected_epoch": {str(s): r["epoch"] for s, r in runs.items()},
                "stop_epoch": {str(s): r.get("stop_epoch") for s, r in runs.items()},
                "consumed_epochs": {str(s): r.get("consumed_epochs") for s, r in runs.items()},
                "mean_validation_rel_l2": float(np.mean([r["validation_rel_l2"] for r in runs.values()])),
            }
    return out


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def mean_validation(record: dict, label: str, seeds=SEEDS) -> float:
    return float(np.mean([record["runs"][f"dataonly_seed{s}"][label]["validation_rel_l2"] for s in seeds]))


def select(grid: dict = GRID, extension: dict = EXTENSION, audit: Path = AUDIT, seeds=SEEDS) -> dict:
    """The weight decay of the "weight decay" arms, from validation errors only."""
    lab = labels(next(iter(grid.values())))
    tested = dict(grid)
    for wd, study in extension.items():
        if (audit / f"{study}_variants_validation.json").exists():
            tested[wd] = study
    missing = [s for s in tested.values() if not (audit / f"{s}_variants_validation.json").exists()]
    if missing:
        raise SystemExit(f"validation files missing for {missing}")
    records = {wd: read(audit / f"{s}_variants_validation.json") for wd, s in tested.items()}
    val = {wd: mean_validation(records[wd], lab["A"], seeds) for wd in tested}
    nonzero = {wd: v for wd, v in val.items() if wd > 0}
    chosen = min(nonzero, key=nonzero.get)
    untested = [wd for wd in extension if wd not in tested and wd > max(tested)]
    status = "extend" if chosen == max(tested) and untested else "final"
    return {
        "plan": PLAN,
        "rule": "weight decay of the WD arms: the non-zero grid value with the lowest mean validation relative L2 over the seeds, "
        "of the fixed-step model (budget A); the same value is used for budget B and for WD+ES. If it is the largest value "
        "tested and the grid can be extended, the extension is trained and the selection repeated once.",
        "seeds": list(seeds),
        "budgets": lab,
        "mean_validation_rel_l2_budget_A": {str(wd): v for wd, v in sorted(val.items())},
        "studies": {str(wd): s for wd, s in sorted(tested.items())},
        "selected_weight_decay": chosen,
        "selected_study": tested[chosen],
        "status": status,
        "extension_to_train": [extension[wd] for wd in untested] if status == "extend" else [],
        "checkpoint_selections": {str(wd): checkpoint_selections(records[wd], lab, seeds) for wd in sorted(tested)},
        "mean_validation_rel_l2_by_budget": {
            str(wd): {b: mean_validation(records[wd], lab[b], seeds) for b in ["A", "B", *lab["sensitivity"]]} for wd in sorted(tested)
        },
        "test_set_read": False,
    }


def arm_records(selection: dict, audit: Path = AUDIT, pino_studies=PINO_STUDIES, pino_run=PINO_RUN, seeds=SEEDS) -> dict:
    """``{budget: {arm: {seed: record}}}`` with validation and test numbers merged per variant."""
    lab = selection["budgets"]

    def merged(study: str, run: str, label: str) -> dict:
        v = read(audit / f"{study}_variants_validation.json")["runs"][run][label]
        t = read(audit / f"{study}_variants_test.json")["runs"][run][label]
        return {**v, **{k: x for k, x in t.items() if k.startswith("test")}}

    base, wd = selection["studies"]["0.0"], selection["selected_study"]
    pino = {}
    for s in seeds:
        for study in pino_studies:
            if (audit / f"{study}_variants_test.json").exists() and f"{pino_run}_seed{s}" in read(audit / f"{study}_variants_test.json")["runs"]:
                pino[s] = merged(study, f"{pino_run}_seed{s}", "final")
    if sorted(pino) != list(seeds):
        raise SystemExit(f"physics-informed comparison runs missing for seeds {sorted(set(seeds) - set(pino))}")
    out = {}
    for budget in ["A", "B", *lab["sensitivity"]]:
        fixed, es, vb = lab[budget], lab[f"ES_{budget}"], lab[f"VB_{budget}"]
        out[budget] = {
            "base": {s: merged(base, f"dataonly_seed{s}", fixed) for s in seeds},
            "WD": {s: merged(wd, f"dataonly_seed{s}", fixed) for s in seeds},
            "ES": {s: merged(base, f"dataonly_seed{s}", es) for s in seeds},
            "VB": {s: merged(base, f"dataonly_seed{s}", vb) for s in seeds},
            "WD+ES": {s: merged(wd, f"dataonly_seed{s}", es) for s in seeds},
            "WD+VB": {s: merged(wd, f"dataonly_seed{s}", vb) for s in seeds},
            "PINO": pino,
        }
    return out


def describe(records: dict, seeds=SEEDS) -> dict:
    keys = {
        "test_rel_l2": lambda r: r["test_rel_l2"],
        "validation_rel_l2": lambda r: r["validation_rel_l2"],
        "train_rel_l2": lambda r: r["train_rel_l2"],
        "test_rel_l2_near_jumps": lambda r: r["test_rel_l2_by_region"]["other_cells"],
        "test_rel_l2_constant_k": lambda r: r["test_rel_l2_by_region"]["constant_k_cells"],
        "test_solver_residual": lambda r: r["test_solver_residual"]["abs_mean"],
        "test_solver_residual_interface": lambda r: r["test_solver_residual"]["abs_mean_interface_nodes"],
        "train_seconds": lambda r: r["train_seconds"],
        "optimiser_steps": lambda r: r["optimiser_steps"],
        "selected_epoch": lambda r: r.get("epoch"),
    }
    out = {}
    for key, get in keys.items():
        values = [get(records[s]) for s in seeds]
        out[key] = {"per_seed": values, "mean": None if None in values else float(np.mean(values))}
    return out


def paired(setting: dict, control: dict, seeds=SEEDS, key=lambda r: r["test_rel_l2"]) -> dict:
    a = np.array([key(setting[s]) for s in seeds])
    b = np.array([key(control[s]) for s in seeds])
    d = paired_seed_interval(a - b)
    d["control_mean"] = float(b.mean())
    # both arms are the same models in every seed (e.g. a rule that selects the final epoch): the interval is the point 0
    d["identical_models"] = bool(np.all(a == b))
    d["relative_change"] = float(d["mean"] / b.mean())
    d["ci_relative"] = [d["ci_low"] / b.mean(), d["ci_high"] / b.mean()]
    if d["ci_high"] < 0:
        d["reading"] = "setting better"
    elif d["ci_low"] > 0:
        d["reading"] = "control better"
    elif -MARGIN * b.mean() <= d["ci_low"] and d["ci_high"] <= MARGIN * b.mean():
        d["reading"] = "no material difference (within 5 %)"
    else:
        d["reading"] = "inconclusive"
    return d


def grid_table(selection: dict, audit: Path = AUDIT, seeds=SEEDS) -> dict:
    """Validation and test error of every weight decay at budget A, side by side (after the selection), to show the selection's optimism."""
    label = selection["budgets"]["A"]
    out = {}
    for wd, study in selection["studies"].items():
        val = read(audit / f"{study}_variants_validation.json")["runs"]
        test = read(audit / f"{study}_variants_test.json")["runs"]
        out[wd] = {
            "study": study,
            "mean_validation_rel_l2": float(np.mean([val[f"dataonly_seed{s}"][label]["validation_rel_l2"] for s in seeds])),
            "mean_test_rel_l2": float(np.mean([test[f"dataonly_seed{s}"][label]["test_rel_l2"] for s in seeds])),
            "selected": study == selection["selected_study"],
        }
    return out


def analyse(selection: dict, audit: Path = AUDIT, **kwargs) -> dict:
    if selection.get("status") != "final":
        raise SystemExit("the selection is not final: train and select the grid extension first")
    seeds = tuple(selection["seeds"])
    arms = arm_records(selection, audit, seeds=seeds, **kwargs)
    primary = paired(arms["A"][PRIMARY[0]], arms["A"][PRIMARY[1]], seeds)
    secondary = {f"{b}: {x} - {y}": paired(arms[b][x], arms[b][y], seeds) for b, x, y in SECONDARY}
    for (name, entry), p in zip(secondary.items(), holm([e["t_p_value"] for e in secondary.values()])):
        entry["holm_p_value"] = p
    sensitivity = {
        budget: {f"{budget}: {x} - {y}": paired(arms[budget][x], arms[budget][y], seeds) for b, x, y in SECONDARY if b == "B"}
        for budget in selection["budgets"]["sensitivity"]
    }
    regional = {
        f"{b}: PINO - {y}, {metric}": paired(arms[b]["PINO"], arms[b][y], seeds, key=get)
        for b in ("A", "B")
        for y in ("base", PRIMARY[1])
        for metric, get in (
            ("near jumps", lambda r: r["test_rel_l2_by_region"]["other_cells"]),
            ("solver residual", lambda r: r["test_solver_residual"]["abs_mean"]),
        )
    }
    return {
        "plan": PLAN,
        "selection": selection,
        "primary": {"comparison": f"budget A: {PRIMARY[0]} - {PRIMARY[1]}, test relative L2", **primary},
        "secondary_holm": secondary,
        "sensitivity_unadjusted": sensitivity,  # B51: same comparisons as budget B, descriptive; B (52 epochs) is the official result
        "descriptive_regions_and_residual": regional,
        "arms": {b: {arm: describe(rec, seeds) for arm, rec in arms[b].items()} for b in arms},
        "grid": grid_table(selection, audit, seeds),
        "notes": [
            "Seeds are the independent units; intervals are Student-t over seeds, relative to the comparison mean.",
            "A non-significant difference is not evidence of equivalence; "
            "'no material difference' requires the whole interval within 5 % of the comparison mean.",
            "The physics-informed runs are not retrained: they are consistent_w0.1 of full_cuda and full_cuda_more (5100 steps).",
            "train_seconds of an early-stopping arm is the compute consumed until its stop (or cap), not the allocated budget.",
        ],
        "environment": collect_metadata(torch.device("cpu")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("select", "analyse"))
    parser.add_argument("--selection", type=Path, default=AUDIT / "regularisation_selection.json")
    parser.add_argument("--out", type=Path, default=AUDIT / "regularisation_study.json")
    args = parser.parse_args()
    if args.stage == "select":
        if args.selection.exists():
            previous = read(args.selection)
            if previous.get("status") != "extend":
                raise SystemExit(f"{args.selection} is final; the selection is made once")
            # repeated once, after the extension it asked for was trained and evaluated on validation
            result = select()
            result["previous_selection"] = previous
            if result["status"] != "final":
                raise SystemExit("the extension does not settle the selection; the grid is extended once only")
        else:
            result = select()
        write_json(args.selection, result)
        print(json.dumps({k: result[k] for k in ("mean_validation_rel_l2_budget_A", "selected_weight_decay", "status")}, indent=1))
    else:
        selection = read(args.selection)
        result = analyse(selection)
        result["selection_file_sha256"] = hashlib.sha256(args.selection.read_bytes()).hexdigest()
        write_json(args.out, result)
        p = result["primary"]
        lo, hi = (100 * x for x in p["ci_relative"])
        print(f"primary, budget A, {PRIMARY[0]} - {PRIMARY[1]}: {100 * p['relative_change']:+.1f} % [{lo:+.1f}, {hi:+.1f}] -> {p['reading']}")


if __name__ == "__main__":
    main()
