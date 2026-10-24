"""The tracked seed study agrees with the per-run records it was computed from."""

import json

import numpy as np
import pytest

from pino_darcy.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"
SETTINGS = {"data only": "dataonly", "shipped 0.1": "shipped_w0.1", "consistent 0.1": "consistent_w0.1"}


def load(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def study():
    return load(RESULTS / "audit" / "seed_study.json")


def records(seed):
    return RESULTS / ("local" if seed == 0 else "local_seeds")


def test_seed_zero_is_the_existing_reduced_study(study):
    audit = load(RESULTS / "audit" / "local_solver_residual.json")["runs"]
    for name, prefix in SETTINGS.items():
        entry, run = study["per_seed"][name]["seed0"], audit[f"{prefix}_seed0"]
        assert entry["exact_residual"] == run["solver_residual"]["abs_mean"]
        assert entry["rel_l2_near_jumps"] == run["rel_l2_by_region"]["other_cells"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_per_seed_errors_are_the_means_of_the_stored_per_sample_errors(study, seed):
    for name, prefix in SETTINGS.items():
        evaluation = load(records(seed) / f"{prefix}_seed{seed}" / "eval_metrics.json")
        per_sample = np.array(evaluation["test_rel_l2_per_sample"])
        assert per_sample.size == 256
        assert study["per_seed"][name][f"seed{seed}"]["rel_l2"] == pytest.approx(per_sample.mean(), rel=1e-6)


def test_paired_differences_recompute_from_the_per_seed_values(study):
    for name in ("shipped 0.1", "consistent 0.1"):
        c = study["paired_against_data_only"][name]["rel_l2"]
        d = [study["per_seed"][name][f"seed{s}"]["rel_l2"] - study["per_seed"]["data only"][f"seed{s}"]["rel_l2"] for s in range(3)]
        assert c["per_seed"] == pytest.approx(d)
        half = 4.302653 * np.std(d, ddof=1) / np.sqrt(3)
        assert (c["ci_low"], c["ci_high"]) == pytest.approx((np.mean(d) - half, np.mean(d) + half), rel=1e-5)


def test_runs_differ_only_in_residual_weight_and_seed():
    manifest = load(RESULTS / "local_seeds" / "run_manifest.json")
    runs = manifest["runs"]
    assert len(runs) == 9
    assert {r["training_config_sha256"] for r in runs} == {manifest["shared_training_config_sha256"]}
    assert len({r["run_config_sha256"] for r in runs}) == 9
    assert len({json.dumps(r["dataset_files_sha256"], sort_keys=True) for r in runs}) == 1
    assert len({r["torch_version"] for r in runs}) == 1 and len({r["physicsnemo_commit"] for r in runs}) == 1
    assert all(r["git_dirty"] is False for r in runs)
