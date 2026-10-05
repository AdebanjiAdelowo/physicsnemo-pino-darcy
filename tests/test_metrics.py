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
