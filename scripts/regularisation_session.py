"""Steps of a regularisation-study session (kaggle/run_regularisation.ipynb), kept here so they can be tested.

    python scripts/regularisation_session.py env-check                     # versions equal to those of full_cuda
    python scripts/regularisation_session.py restore-inputs --input-root /kaggle/input
    python scripts/regularisation_session.py pilot --out /kaggle/working/regularisation_pilot.json
    python scripts/regularisation_session.py check-pilot                   # tracked pilot and configured budget B agree
    python scripts/regularisation_session.py archive reg_cuda_wd0 --out /kaggle/working
    python scripts/regularisation_session.py restore-runs --input-root /kaggle/input --manifest results/audit/regularisation_archives.sha256
    python scripts/regularisation_session.py check-selection               # tracked, final, validation-only selection
    python scripts/regularisation_session.py check-gate                    # before training: every test evaluation is refused

Every restore verifies SHA-256 hashes before and after extraction and never overwrites a file.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

from _bootstrap import ROOT
from pino_darcy import selection as S
from pino_darcy.config import load_config

INPUTS = Path("kaggle/regularisation_inputs.sha256")
PILOT = Path("results/audit/regularisation_pilot.json")
ARCHIVES = Path("results/audit/regularisation_archives.sha256")
REFERENCE_STUDY = "full_cuda"  # trained the physics-informed checkpoints; the environment must match it
PILOT_EPOCHS = 3
COMPARABLE = 1e-3  # largest relative difference of the pilot histories from the full_cuda histories


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, str, str]]:
    """``(kind, sha256, name)`` lines; ``#`` starts a comment."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            kind, digest, name = line.split(maxsplit=2)
            rows.append((kind, digest, name))
    return rows


# --- environment ------------------------------------------------------------------------------


def installed_versions() -> dict:
    import numpy
    import sympy
    import torch
    import warp

    from pino_darcy.provenance import physicsnemo_source

    return {
        "python_version": ".".join(sys.version.split()[0].split(".")[:2]),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": numpy.__version__,
        "warp_version": warp.__version__,
        "sympy_version": sympy.__version__,
        "physicsnemo_commit": physicsnemo_source()["commit"],
    }


def reference_versions(root: Path = ROOT) -> dict:
    meta = json.loads((root / "results" / REFERENCE_STUDY / "study_metadata.json").read_text())
    return {
        "python_version": ".".join(meta["python_version"].split(".")[:2]),
        "torch_version": meta["torch_version"],
        "cuda_version": meta["cuda_version"],
        "numpy_version": meta["numpy_version"],
        "warp_version": meta["warp_version"],
        "sympy_version": meta["sympy_version"],
        "physicsnemo_commit": meta["physicsnemo"]["commit"],
    }


def env_check(installed: dict, reference: dict) -> list[str]:
    return [f"{k}: installed {installed.get(k)!r}, {REFERENCE_STUDY} {v!r}" for k, v in reference.items() if installed.get(k) != v]


# --- restore -----------------------------------------------------------------------------------


def find(input_root: Path, name: str) -> Path:
    hits = sorted(p for p in Path(input_root).rglob(name) if p.is_file())
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one {name} under {input_root}, found {len(hits)}")
    return hits[0]


def extract_new(archive: Path, target: Path) -> int:
    """Extract the members that do not exist yet; existing files are kept and checked by the caller."""
    with zipfile.ZipFile(archive) as z:
        for m in z.namelist():
            if Path(m).is_absolute() or ".." in Path(m).parts:
                raise SystemExit(f"{archive}: unsafe member {m}")
        new = [m for m in z.namelist() if not m.endswith("/") and not (target / m).exists()]
        z.extractall(target, new)
    return len(new)


def restore_inputs(input_root: Path, root: Path = ROOT, manifest: Path | None = None) -> dict:
    """Restore and verify the physics-informed checkpoints and records of full_cuda and full_cuda_more."""
    rows = read_manifest(manifest or root / INPUTS)
    report = {"archives": {}, "checkpoints_verified": 0}
    for kind, digest, name in rows:
        if kind != "archive":
            continue
        path = find(input_root, name)
        if sha256(path) != digest:
            raise SystemExit(f"{path}: SHA-256 differs from {INPUTS}")
        # eval bundles hold runs/<study>/...; record archives hold <study>/...
        target = root if name.endswith("-eval-bundle.zip") else root / "runs"
        target.mkdir(parents=True, exist_ok=True)
        report["archives"][name] = {"sha256": digest, "extracted": extract_new(path, target)}
    for kind, digest, name in rows:
        if kind == "checkpoint":
            if sha256(root / name) != digest:
                raise SystemExit(f"{name}: SHA-256 differs from {INPUTS} after restoring")
            report["checkpoints_verified"] += 1
    report["metadata"] = check_input_metadata(root)
    return report


def check_input_metadata(root: Path = ROOT) -> dict:
    """The restored studies are the tracked ones: same metadata, configuration and run records."""
    out = {}
    for study in ("full_cuda", "full_cuda_more"):
        restored = json.loads((root / "runs" / study / "study_metadata.json").read_text())
        tracked = json.loads((root / "results" / study / "study_metadata.json").read_text())
        if restored != tracked:
            raise SystemExit(f"runs/{study}/study_metadata.json differs from the tracked results/{study}")
        if restored["device"] != "cuda" or restored["git_dirty"] is not False:
            raise SystemExit(f"{study}: not a clean CUDA study")
        cfg = load_config(study)
        from omegaconf import OmegaConf

        if restored["config"] != OmegaConf.to_container(cfg, resolve=True):
            raise SystemExit(f"{study}: the configuration of the restored study differs from configs/{study}.yaml")
        for seed in cfg.study.seeds:
            run = f"consistent_w0.1_seed{seed}"
            m = json.loads((root / "runs" / study / run / "train_metrics.json").read_text())
            if (m["residual"], m["physics_weight"], m["seed"], m["optimiser_steps"]) != ("consistent", 0.1, seed, 5100):
                raise SystemExit(f"{study}/{run}: unexpected training record")
            out[f"{study}/{run}"] = {"commit": restored["git_commit"], "steps": m["optimiser_steps"], "train_seconds": m["train_seconds"]}
    return out


# --- pilot -----------------------------------------------------------------------------------------


def epoch_times(history: Path) -> list[float]:
    elapsed = [float(r["elapsed_seconds"]) for r in csv.DictReader(open(history))]
    return [b - a for a, b in zip([0.0] + elapsed, elapsed)]


def budget_epochs(t_pino: float, t_data: float, steps_epochs: int = 50) -> int:
    """The pre-registered rule: B = ceil(50 * t_pino / t_data) epochs."""
    return math.ceil(steps_epochs * t_pino / t_data - 1e-12)


def history_difference(pilot: Path, reference: Path, epochs: int) -> dict:
    a = list(csv.DictReader(open(pilot)))[:epochs]
    b = list(csv.DictReader(open(reference)))[:epochs]
    out = {}
    for key in ("train_loss_data", "validation_rel_l2"):
        out[key] = max(abs(float(x[key]) - float(y[key])) / abs(float(y[key])) for x, y in zip(a, b))
    return out


PILOT_RUNS = {
    # label: (config, run directory name, overrides); seed 0, budget-A snapshot and caps scaled to the pilot
    "data_wd0": ("reg_cuda_wd0", "dataonly_seed0", ["name=reg_pilot_wd0"]),
    "data_wd1e-2": ("reg_cuda_wd1e-2", "dataonly_seed0", ["name=reg_pilot_wd1e-2"]),
    "pino": ("full_cuda", "consistent_w0.1_seed0", ["name=reg_pilot_pino", "study.residuals=[consistent]", "study.physics_weights=[0.1]"]),
}


def pilot(root: Path = ROOT, epochs: int = PILOT_EPOCHS, extra: list[str] | None = None, compare: bool = True) -> dict:
    result = {"rule": "B = ceil(50 * t_pino / t_data); t = mean time of epochs 2 to 3 of seed 0 (elapsed_seconds, validation included); "
              "t_data from reg_cuda_wd1e-2, t_pino from consistent 0.1 as in full_cuda", "epochs": epochs, "runs": {}}
    for label, (config, run, overrides) in PILOT_RUNS.items():
        name = next(o for o in overrides if o.startswith("name=")).split("=", 1)[1]
        if (root / "runs" / name).exists():
            raise SystemExit(f"runs/{name} exists; the pilot starts from scratch")
        args = [*overrides, f"max_epochs={epochs}", "study.seeds=[0]", *(extra or [])]
        if config.startswith("reg_"):
            args += [f"snapshots.epochs=[{epochs - 1}]", f"early_stopping.caps=[{epochs - 1},{epochs}]"]
        subprocess.run([sys.executable, str(root / "scripts" / "train.py"), "--config", config, "--no-evaluate", *args], cwd=root, check=True)
        history = root / "runs" / name / run / "history.csv"
        times = epoch_times(history)
        entry = {
            "config": config,
            "overrides": args,
            "epoch_seconds": times,
            "seconds_per_epoch": sum(times[1:]) / len(times[1:]),
            "train_metrics": json.loads((root / "runs" / name / run / "train_metrics.json").read_text()),
        }
        if compare:
            entry["relative_difference_from_full_cuda"] = history_difference(history, root / "results" / "full_cuda" / run / "history.csv", epochs)
        result["runs"][label] = entry
    t_data, t_pino = result["runs"]["data_wd1e-2"]["seconds_per_epoch"], result["runs"]["pino"]["seconds_per_epoch"]
    result.update(t_data=t_data, t_pino=t_pino, ratio=t_pino / t_data, budget_B_epochs=budget_epochs(t_pino, t_data))
    if compare:
        worst = max(v for label in ("data_wd0", "pino") for v in result["runs"][label]["relative_difference_from_full_cuda"].values())
        result["comparability"] = {"largest_relative_difference": worst, "threshold": COMPARABLE, "comparable": worst <= COMPARABLE}
    from pino_darcy.provenance import collect_metadata
    import torch

    result["environment"] = collect_metadata(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    result["versions"] = installed_versions()
    return result


def check_pilot(root: Path = ROOT) -> dict:
    path = root / PILOT
    if not path.exists():
        raise SystemExit(f"{PILOT} is not tracked: run the pilot session, commit its result and set budget B first")
    p = json.loads(path.read_text())
    configured = int(load_config("reg_cuda").max_epochs)
    if p["budget_B_epochs"] != configured:
        raise SystemExit(f"budget B: pilot {p['budget_B_epochs']} epochs, configs/reg_cuda.yaml {configured}")
    if not p.get("comparability", {}).get("comparable"):
        raise SystemExit("the pilot did not find the new runs comparable with full_cuda; see the plan before training")
    return p


# --- study archives ----------------------------------------------------------------------------------


def archive(study: str, out: Path, root: Path = ROOT) -> list[str]:
    """Records (without test results) and checkpoints with snapshots of a trained study; hashes appended to the manifest."""
    run_dir = root / "runs" / study
    py = [sys.executable, str(root / "scripts" / "results.py")]
    subprocess.run([*py, "package", str(run_dir), "--records-only"], check=True)
    subprocess.run([*py, "bundle", str(run_dir)], check=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    pairs = (
        (run_dir.with_suffix(".zip"), f"physicsnemo-pino-darcy-{study}-records.zip"),
        (run_dir.parent / f"{study}_eval_bundle.zip", f"physicsnemo-pino-darcy-{study}-bundle.zip"),
    )
    for src, name in pairs:
        target = out / name
        if target.exists():
            raise SystemExit(f"{target} exists; archives are not overwritten")
        target.write_bytes(src.read_bytes())
        lines.append(f"archive {sha256(target)} {name}")
    with open(out / ARCHIVES.name, "a") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def restore_runs(input_root: Path, manifest: Path, root: Path = ROOT) -> dict:
    """Restore the trained controls of an earlier session, verifying every archive against the tracked manifest."""
    report = {}
    for kind, digest, name in read_manifest(manifest):
        path = find(input_root, name)
        if sha256(path) != digest:
            raise SystemExit(f"{path}: SHA-256 differs from {manifest}")
        target = root if name.endswith("-bundle.zip") else root / "runs"
        report[name] = extract_new(path, target)
    return report


def check_selection(root: Path = ROOT) -> dict:
    selection = S.require_final_selection(root / S.SELECTION_FILE)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", S.SELECTION_FILE], cwd=root, capture_output=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", S.SELECTION_FILE], cwd=root, capture_output=True, text=True).stdout.strip()
    if not tracked or dirty:
        raise SystemExit(f"{S.SELECTION_FILE} must be committed, unchanged, before the test session")
    return selection


GATED_STUDIES = ("reg_cuda_wd0", "reg_cuda_wd1e-3", "reg_cuda_wd1e-2", "reg_cuda_wd1e-1", "reg_cuda_wd1e0", "reg_cuda_wd1e1", "reg_cuda_wd1e2")


def check_gate(root: Path = ROOT) -> list[str]:
    """Before any selection exists: every regularisation study has the gate on and a test evaluation is refused."""
    if (root / S.SELECTION_FILE).exists():
        raise SystemExit(f"{S.SELECTION_FILE} exists already; check-gate is for the session before the selection")
    for study in GATED_STUDIES:
        cfg = load_config(study)
        if not cfg.evaluation.get("test_after_selection"):
            raise SystemExit(f"{study}: the test gate is off")
        try:
            S.guard_test_evaluation(cfg, root)
        except SystemExit:
            continue
        raise SystemExit(f"{study}: a test evaluation would not be refused")
    return list(GATED_STUDIES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env-check")
    p = sub.add_parser("restore-inputs")
    p.add_argument("--input-root", type=Path, required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=PILOT_EPOCHS)
    p.add_argument("--no-compare", action="store_true", help="smoke use only: skip the comparison with full_cuda")
    p.add_argument("overrides", nargs="*", help="smoke use only: extra Hydra overrides of every pilot run")
    sub.add_parser("check-pilot")
    p = sub.add_parser("archive")
    p.add_argument("study")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("restore-runs")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("check-selection")
    sub.add_parser("check-gate")
    args = parser.parse_args()

    if args.command == "env-check":
        installed, reference = installed_versions(), reference_versions()
        print(json.dumps({"installed": installed, REFERENCE_STUDY: reference}, indent=1))
        if problems := env_check(installed, reference):
            raise SystemExit("environment differs from the one that trained the physics-informed checkpoints:\n" + "\n".join(problems))
        print("ENVIRONMENT MATCHES", REFERENCE_STUDY)
    elif args.command == "restore-inputs":
        report = restore_inputs(args.input_root)
        print(json.dumps(report, indent=1))
        print(f"RESTORED AND VERIFIED: {len(report['archives'])} archives, {report['checkpoints_verified']} checkpoint files")
    elif args.command == "pilot":
        if args.out.exists():
            raise SystemExit(f"{args.out} exists")
        result = pilot(epochs=args.epochs, extra=args.overrides, compare=not args.no_compare)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        t_data, t_pino, ratio, b = (result[k] for k in ("t_data", "t_pino", "ratio", "budget_B_epochs"))
        print(f"t_data {t_data:.3f} s/epoch, t_pino {t_pino:.3f} s/epoch, ratio {ratio:.4f} -> budget B = {b} epochs")
        if "comparability" in result:
            print("comparability with full_cuda:", result["comparability"])
    elif args.command == "check-pilot":
        p = check_pilot()
        print(f"budget B = {p['budget_B_epochs']} epochs (ratio {p['ratio']:.4f}); configs/reg_cuda.yaml agrees")
    elif args.command == "archive":
        print("\n".join(archive(args.study, args.out)))
    elif args.command == "restore-runs":
        print(json.dumps(restore_runs(args.input_root, args.manifest), indent=1))
    elif args.command == "check-gate":
        print(f"TEST GATE ACTIVE for {len(check_gate())} studies: any test evaluation is refused until the selection is final")
    else:
        selection = check_selection()
        print(f"selection final, committed: weight decay {selection['selected_weight_decay']} ({selection['selected_study']})")


if __name__ == "__main__":
    main()
