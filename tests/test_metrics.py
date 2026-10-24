import numpy as np
import pytest
import torch

from pino_darcy.metrics import relative_l2, summarise


def test_relative_l2_values_and_scale_invariance():
    target = torch.tensor([[[3.0, 4.0]], [[1.0, 0.0]]])
    pred = target.clone()
    pred[0, 0, 0] += 0.5
    rel = relative_l2(pred, target)
    assert rel.tolist() == pytest.approx([0.1, 0.0])
    # the error is the same for normalised and physical pressure
    assert relative_l2(pred * 3.88433e-3, target * 3.88433e-3).tolist() == pytest.approx(rel.tolist())


def test_summarise():
    s = summarise(torch.tensor([1.0, 2.0, 3.0, 6.0]))
    assert s["mean"] == pytest.approx(3.0) and s["max"] == 6.0 and s["median"] == 2.0
    assert summarise(torch.tensor([2.0]))["std"] == 0.0


def test_paired_seed_interval_matches_the_t_distribution():
    from pino_darcy.metrics import paired_seed_interval

    s = paired_seed_interval([-0.001, 0.0005, -0.0002])
    assert s["mean"] == pytest.approx(-0.00023333333, rel=1e-6)
    assert s["std"] == pytest.approx(0.000750555, rel=1e-5)
    # t(0.975, 2) = 4.302653
    half = 4.302653 * 0.000750555 / 3**0.5
    assert (s["ci_low"], s["ci_high"]) == pytest.approx((s["mean"] - half, s["mean"] + half), rel=1e-5)
    assert s["seeds_negative"] == 2 and s["sign_test_p_value"] == pytest.approx(1.0)
    assert paired_seed_interval([-1.0, -2.0, -3.0])["sign_test_p_value"] == pytest.approx(0.25)
    assert paired_seed_interval([0.5])["ci_low"] is None


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    from pino_darcy.metrics import paired_bootstrap

    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 0.1, 200), rng.normal(1.0, 0.1, 200)
    first, second = paired_bootstrap(a, b), paired_bootstrap(a, b)
    assert first == second
    assert first["ci_low"] < first["mean"] < first["ci_high"]
    assert paired_bootstrap(a - 1.0, a)["fraction_improved"] == 1.0
