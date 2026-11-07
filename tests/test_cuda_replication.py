"""The CUDA replication analysis: the pre-registered criterion, and the computation on the tracked MPS records.

No CUDA result exists yet; the MPS studies stand in for the CUDA ones (``cuda_prefix="full_mps"``), which
must reproduce the MPS numbers exactly and give zero device differences. Offline, CPU, no checkpoints.
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
    empty = C.analyse(RESULTS)  # no full_cuda* study is tracked
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
