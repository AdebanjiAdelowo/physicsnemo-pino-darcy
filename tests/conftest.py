import pytest
import torch
import warp as wp
from omegaconf import OmegaConf

from pino_darcy.config import load_config

wp.config.quiet = True


def synthetic_sets(n_train=4, n_eval=3, size=24, seed=0):
    """Two-valued permeabilities and smooth pressures on a small grid, in the layout of ``load_datasets``."""
    g = torch.Generator().manual_seed(seed)

    def split(n):
        k = torch.where(torch.rand(n, 1, size, size, generator=g) > 0.5, 2.667, 0.667)
        x = torch.linspace(0, 1, size)
        u = torch.sin(torch.pi * x)[:, None] * torch.sin(torch.pi * x)[None, :] * (1 + torch.rand(n, 1, 1, 1, generator=g))
        return {"k": k, "u": u.float()}

    return {"train": split(n_train), "validation": split(n_eval), "test": split(n_eval), "manifest": {"synthetic": True}}


@pytest.fixture
def tiny_cfg():
    cfg = load_config(
        "smoke",
        ["device=cpu", "max_epochs=2", "model.fno.latent_channels=4", "model.fno.num_fno_modes=4", "model.fno.padding=4"],
    )
    OmegaConf.set_struct(cfg, False)
    return cfg


@pytest.fixture
def sets():
    return synthetic_sets()
