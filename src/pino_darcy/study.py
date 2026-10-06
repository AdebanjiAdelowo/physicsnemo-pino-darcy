"""The loop over residual definitions, physics weights and seeds, and the summary tables."""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from . import engine
from .provenance import collect_metadata, write_json

SUMMARY_FIELDS = [
    "run",
    "residual",
    "physics_weight",
    "seed",
    "parameters",
    "final_train_loss_data",
    "final_train_loss_pde",
    "final_train_loss_total",
    "validation_mse",
    "test_mse",
    "test_rel_l2_mean",
    "test_rel_l2_std",
    "test_rel_l2_median",
    "test_rel_l2_max",
    "test_pde_l1_shipped",
    "test_pde_l1_consistent",
    "test_pde_rms_consistent",
    "test_pde_signed_mean_consistent",
    "test_pde_smooth_consistent",
    "test_pde_interface_consistent",
    "train_seconds",
    "optimisation_seconds",
    "optimisation_ms_per_step",
    "inference_ms_per_sample",
    "peak_train_memory_mb",
    "peak_inference_memory_mb",
]
AGGREGATED = [
    "test_rel_l2_mean",
    "test_mse",
    "final_train_loss_data",
    "test_pde_l1_shipped",
    "test_pde_l1_consistent",
    "test_pde_smooth_consistent",
    "test_pde_interface_consistent",
    "train_seconds",
    "optimisation_ms_per_step",
    "inference_ms_per_sample",
    "peak_train_memory_mb",
]


def train_study(cfg: DictConfig, root: Path, resume: bool = False) -> Path:
    """Train every run of the study. ``resume`` skips runs that already finished."""
    out_dir, device, sets = engine.prepare_study(cfg, root, write_metadata=not resume)
    for residual, weight, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(residual, weight, seed)
        if resume and (run_dir / "train_metrics.json").exists():
            print(f"[skip] {run_dir.name} already trained", flush=True)
            continue
        m = engine.train_run(cfg, residual, weight, seed, run_dir, device, sets)
        pde = "none" if m["final_train_loss_pde"] is None else f"{m['final_train_loss_pde']:.3e}"
        print(
            f"[train] {run_dir.name}: data loss {m['final_train_loss_data']:.3e}, PDE loss {pde}, "
            f"{m['train_seconds']:.1f} s on {device}",
            flush=True,
        )
    return out_dir


def summary_row(train: dict, evaluation: dict) -> dict:
    row = {**train, **{k: v for k, v in evaluation.items() if not isinstance(v, (dict, list))}}
    for stat, value in evaluation["test_rel_l2"].items():
        row[f"test_rel_l2_{stat}"] = value
    row["test_pde_l1_shipped"] = evaluation["test_residual_shipped"]["l1"]
    consistent = evaluation["test_residual_consistent"]
    row["test_pde_l1_consistent"] = consistent["l1"]
    row["test_pde_rms_consistent"] = consistent["rms"]
    row["test_pde_signed_mean_consistent"] = consistent["signed_mean"]
    row["test_pde_smooth_consistent"] = consistent["abs_mean_smooth"]
    row["test_pde_interface_consistent"] = consistent["abs_mean_interface"]
    return {k: row[k] for k in SUMMARY_FIELDS}


def evaluate_study(cfg: DictConfig, root: Path) -> dict:
    """Evaluate every run from its checkpoint and write the summary files and plotting fields."""
    out_dir, device, sets = engine.prepare_study(cfg, root, write_metadata=False, with_train=False)
    n_plot = cfg.evaluation.plot_samples
    plot_seed = cfg.study.seeds[0]
    fields = {"permeability": sets["test"]["k"][:n_plot, 0].numpy(), "truth": sets["test"]["u"][:n_plot, 0].numpy()}
    rows = []
    for residual, weight, seed in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(residual, weight, seed)
        evaluation, run_fields = engine.evaluate_run(cfg, run_dir, device, sets)
        rows.append(summary_row(json.loads((run_dir / "train_metrics.json").read_text()), evaluation))
        if seed == plot_seed:
            for key, value in run_fields.items():
                # residual fields are only plotted: half precision keeps the tracked file small
                fields[f"{key}__{run_dir.name}"] = value.astype(np.float16) if key.startswith("residual") else value
        print(
            f"[eval] {run_dir.name}: test relative L2 {rows[-1]['test_rel_l2_mean']:.4e}, "
            f"PDE L1 shipped {rows[-1]['test_pde_l1_shipped']:.3e}, consistent {rows[-1]['test_pde_l1_consistent']:.3e}",
            flush=True,
        )

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": cfg.name,
        "evaluated_with": collect_metadata(device),
        "device": str(device),
        "evaluation_device_note": "inference times and memory are those of the evaluating device",
        "units": "normalised variables k' = k / 4.49996, u' = u / 3.88433e-3; relative L2 is unit-free",
        "reference_residual": engine.reference_residuals(sets, device),
        "by_setting": aggregate(rows),
        "runs": rows,
    }
    write_json(out_dir / "summary.json", summary)
    np.savez_compressed(out_dir / "fields.npz", plot_seed=plot_seed, **fields)
    return summary


def verify_study(cfg: DictConfig, root: Path) -> list[dict]:
    """Recompute the test error of every run from its checkpoint and compare with the stored value.

    Nothing is written. Used after results were produced on another machine: the stored
    timings and memory figures belong to that machine and must not be replaced.
    """
    out_dir, device, sets = engine.prepare_study(cfg, root, write_metadata=False, with_train=False)
    rows = []
    for run in engine.study_runs(cfg):
        run_dir = out_dir / engine.run_name(*run)
        stored = json.loads((run_dir / "eval_metrics.json").read_text())["test_rel_l2"]["mean"]
        recomputed = engine.evaluate_run(cfg, run_dir, device, sets, write=False)[0]["test_rel_l2"]["mean"]
        rows.append({"run": run_dir.name, "stored": stored, "recomputed": recomputed, "relative_difference": abs(recomputed - stored) / stored})
        print(f"[verify] {run_dir.name}: stored {stored:.6e}, recomputed on {device} {recomputed:.6e}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean and sample standard deviation over seeds for every (residual, weight)."""
    out = []
    settings = sorted({(r["residual"] or "", r["physics_weight"]) for r in rows})
    for residual, weight in settings:
        group = [r for r in rows if (r["residual"] or "") == residual and r["physics_weight"] == weight]
        entry = {"residual": residual or None, "physics_weight": weight, "n_seeds": len(group), "parameters": group[0]["parameters"]}
        for key in AGGREGATED:
            values = [r[key] for r in group if r[key] is not None]
            entry[f"{key}_mean"] = statistics.fmean(values) if values else None
            entry[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        out.append(entry)
    return out
