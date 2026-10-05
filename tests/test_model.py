import torch

from pino_darcy.config import load_config
from pino_darcy.model import build_fno, count_parameters


def test_official_model_shape_and_size():
    model = build_fno(load_config("official"))
    assert count_parameters(model) == 2_365_169
    with torch.no_grad():
        assert model(torch.rand(1, 1, 64, 64)).shape == (1, 1, 64, 64)


def test_seed_fixes_the_initial_weights():
    cfg = load_config("smoke")
    torch.manual_seed(3)
    a = build_fno(cfg)
    torch.manual_seed(3)
    b = build_fno(cfg)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
