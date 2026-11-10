"""The CUDA replication analysis: the pre-registered criterion, and the computation on the tracked MPS records.

The MPS studies stand in for the CUDA ones (``cuda_prefix="full_mps"``), which must reproduce the MPS
numbers exactly and give zero device differences. The tracked CUDA studies (``full_cuda``, ``full_cuda_more``,
trained on Kaggle at 416bfe6) must agree with their per-run records, with an independent recomputation of the
primary statistic and with ``results/audit/cuda_replication.json``. Offline, CPU, no checkpoints.
"""

import json
import sys

import pytest

from pino_darcy.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import cuda_replication as C  # noqa: E402

RESULTS = REPO_ROOT / "results"


@pytest.mark.parametrize(
    ("high", "n", "verdict"),
    [
        (-0.196, 5, "replicated on CUDA"),
        (-0.0501, 5, "replicated on CUDA"),
        (-0.05, 5, "not replicated by the criterion; the direction agrees"),
        (-0.01, 5, "not replicated by the criterion; the direction agrees"),
        (0.0, 5, "not replicated"),
        (0.03, 5, "not replicated"),
        (-0.30, 4, "not evaluable"),
        (-0.30, 3, "not evaluable"),
        (None, 0, "not evaluable"),
    ],
)
def test_criterion_is_the_pre_registered_rule(high, n, verdict):
    assert C.criterion(high, n) == verdict


def test_plan_states_the_criterion_and_threshold():
    plan = (REPO_ROOT / C.PLAN).read_text()
    assert "upper endpoint of the relative 95 % interval is below -5 %" in plan
    assert C.THRESHOLD == -0.05 and "t_{0.975,\\,4}" in plan


@pytest.fixture(scope="module")
def stand_in():
    return C.analyse(RESULTS, cuda_prefix="full_mps", mps_prefix="full_mps")


def test_stand_in_reproduces_the_five_seed_mps_result(stand_in):
    tracked = json.loads((RESULTS / "audit" / "full_study.json").read_text())["g2_stage3_n5"]["comparisons"]["consistent 0.1 - data only"]["rel_l2"]
    p = stand_in["primary"]
    assert p["seeds"] == [0, 1, 2, 3, 4]
    assert p["relative_change"] == pytest.approx(tracked["relative_change"], rel=1e-12)
    assert p["ci_relative"] == pytest.approx(tracked["ci_relative"], rel=1e-12)
    assert p["verdict"] == "replicated on CUDA"
    assert stand_in["primary_seeds_0_to_2"]["seeds"] == [0, 1, 2]


def test_stand_in_has_no_device_difference(stand_in):
    for c in stand_in["cuda_minus_mps"]["effects"].values():
        assert c["mean"] == 0 and len(c["seeds"]) == 5
    for metrics in stand_in["cuda_minus_mps"]["per_setting"].values():
        assert all(m["mean"] == 0 for m in metrics.values())
    assert set(stand_in["secondary"]) == {"shipped 0.1 - data only"} | {
        f"{r} {w} - data only" for r in ("shipped", "consistent") for w in ("0.001", "0.01", "1")
    }


def test_missing_or_partial_cuda_results_are_not_evaluable(tmp_path):
    empty = C.analyse(tmp_path)  # no study at all
    assert empty["primary"]["verdict"] == "not evaluable" and empty["primary"]["detail"] is None
    # only seeds 0 to 2 (stage 3 missing): never evaluated on three seeds
    (tmp_path / "audit").mkdir()
    (tmp_path / "full_mps").symlink_to(RESULTS / "full_mps")
    (tmp_path / "audit" / "full_mps_solver_residual.json").symlink_to(RESULTS / "audit" / "full_mps_solver_residual.json")
    partial = C.analyse(tmp_path, cuda_prefix="full_mps", mps_prefix="full_mps")
    assert partial["primary"]["seeds"] == [0, 1, 2] and partial["primary"]["verdict"] == "not evaluable"


def test_study_status_counts_finished_runs_and_missing_checkpoints(tmp_path):
    import study_status

    st = study_status.status("full_cuda", root=tmp_path)
    assert (st["runs"], len(st["finished"]), len(st["to_train"])) == (9, 0, 9) and st["device"] == "cuda"
    finished = tmp_path / "runs" / "full_cuda" / "dataonly_seed0"
    (finished / "checkpoints").mkdir(parents=True)
    (finished / "train_metrics.json").write_text("{}")
    assert study_status.status("full_cuda", root=tmp_path)["missing_checkpoint"] == ["dataonly_seed0"]
    (finished / "checkpoints" / "FNO.0.50.mdlus").write_bytes(b"")
    st = study_status.status("full_cuda", root=tmp_path)
    assert st["missing_checkpoint"] == [] and [r["run"] for r in st["finished"]] == ["dataonly_seed0"] and len(st["to_train"]) == 8


def test_publish_never_replaces_an_existing_study(tmp_path, monkeypatch):
    import results

    monkeypatch.setattr(results, "ROOT", tmp_path)
    existing = tmp_path / "results" / "full_cuda"
    existing.mkdir(parents=True)
    (existing / "summary.json").write_text('{"kept": true}')
    source = tmp_path / "runs" / "full_cuda"
    source.mkdir(parents=True)
    (source / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        results.publish(source, force=False)
    assert json.loads((existing / "summary.json").read_text()) == {"kept": True}


CUDA_STUDIES = {"full_cuda": (0, 1, 2), "full_cuda_more": (3, 4)}
T_975_DF4 = 2.7764451051977987  # Student t, 0.975 quantile, 4 degrees of freedom (the plan states 2.776)


def _cuda_error(seed: int, run: str) -> float:
    study = "full_cuda" if seed in CUDA_STUDIES["full_cuda"] else "full_cuda_more"
    return json.loads((RESULTS / study / f"{run}_seed{seed}" / "eval_metrics.json").read_text())["test_rel_l2"]["mean"]


def test_tracked_cuda_runs_are_complete_and_from_the_pinned_commit():
    for study, seeds in CUDA_STUDIES.items():
        meta = json.loads((RESULTS / study / "study_metadata.json").read_text())
        assert meta["device"] == "cuda" and meta["git_dirty"] is False
        assert meta["git_commit"] == "416bfe6fc35527390e6cc72e49545d8826e05cdf"
        assert meta["physicsnemo"]["commit"] == meta["upstream"]["commit"] == "b45a5c810c741e6b41f8515be24c51121f8fc21f"
        runs = sorted(p.name for p in (RESULTS / study).iterdir() if p.is_dir())
        assert runs == sorted(f"{r}_seed{s}" for r in ("dataonly", "shipped_w0.1", "consistent_w0.1") for s in seeds)
        for run in runs:
            train = json.loads((RESULTS / study / run / "train_metrics.json").read_text())
            history = (RESULTS / study / run / "history.csv").read_text().strip().splitlines()
            assert train["optimiser_steps"] == 5100 and len(history) == 1 + 50
            assert (RESULTS / study / run / "eval_metrics.json").exists()


def test_cuda_primary_statistic_recomputed_from_the_run_records():
    """Independent of scripts/cuda_replication.py: the formula of docs/CUDA_REPLICATION_PLAN.md."""
    seeds = range(5)
    control = [_cuda_error(s, "dataonly") for s in seeds]
    d = [_cuda_error(s, "consistent_w0.1") - c for s, c in zip(seeds, control)]
    mean = sum(d) / 5
    sd = (sum((x - mean) ** 2 for x in d) / 4) ** 0.5
    half = T_975_DF4 * sd / 5**0.5
    base = sum(control) / 5
    rel, high = mean / base, (mean + half) / base
    assert all(x < 0 for x in d)
    assert round(rel * 100, 1) == -21.5 and round((mean - half) / base * 100, 1) == -23.5 and round(high * 100, 1) == -19.6
    assert C.criterion(high, 5) == "replicated on CUDA"
    tracked = json.loads((RESULTS / "audit" / "cuda_replication.json").read_text())["primary"]
    assert tracked["verdict"] == "replicated on CUDA" and tracked["seeds"] == [0, 1, 2, 3, 4]
    assert tracked["relative_change"] == pytest.approx(rel, rel=1e-12)
    assert tracked["ci_relative"] == pytest.approx([(mean - half) / base, high], rel=1e-9)


def test_tracked_cuda_replication_is_what_the_analysis_gives():
    tracked = json.loads((RESULTS / "audit" / "cuda_replication.json").read_text())
    fresh = json.loads(json.dumps(C.analyse(RESULTS)))
    for key in ("criterion", "studies", "primary", "primary_seeds_0_to_2", "secondary", "mps_reference", "per_setting", "cuda_minus_mps"):
        assert _close(fresh[key], tracked[key]), key


def _close(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == pytest.approx(b, rel=1e-12, abs=1e-15)
    return a == b
