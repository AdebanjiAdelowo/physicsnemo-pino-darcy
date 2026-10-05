"""The PDE residual of the upstream PhysicsInformer and the two residual definitions."""

import math

import pytest
import torch

from pino_darcy import physics as P
from pino_darcy.data import K_SCALE, U_SCALE


def manufactured(n, h):
    """Smooth u and k on a cell-centred grid, and -div(k grad u) evaluated analytically."""
    x = (torch.arange(n, dtype=torch.float64) + 0.5) * h
    X, Y = torch.meshgrid(x, x, indexing="ij")
    u = torch.sin(math.pi * X) * torch.sin(2 * math.pi * Y)
    k = 1 + 0.5 * torch.cos(math.pi * X) * Y
    ux = math.pi * torch.cos(math.pi * X) * torch.sin(2 * math.pi * Y)
    uy = 2 * math.pi * torch.sin(math.pi * X) * torch.cos(2 * math.pi * Y)
    kx, ky = -0.5 * math.pi * torch.sin(math.pi * X) * Y, 0.5 * torch.cos(math.pi * X)
    operator = -(kx * ux + ky * uy - 5 * math.pi**2 * k * u)
    return u[None, None].float(), k[None, None].float(), operator


@pytest.mark.parametrize("name", ["shipped", "consistent"])
def test_residual_matches_the_analytic_operator(name):
    spec = P.RESIDUALS[name]
    u, k, operator = manufactured(96, spec["fd_dx"])
    r = P.residual_field(P.build_informer(name, torch.device("cpu")), u, k)
    inner = (slice(P.BORDER, -P.BORDER),) * 2
    error = (r[0, 0][inner].double() - (operator[inner] - spec["forcing"])).abs().max()
    # second-order differences plus single-precision rounding of the second derivative
    assert error < 2e-3 * operator.abs().max()


def test_border_is_zero_and_loss_is_the_mean_absolute_value():
    u, k, _ = manufactured(48, 1 / 240)
    r = P.residual_field(P.build_informer("shipped", torch.device("cpu")), u, k)
    assert r.shape == u.shape
    edge = torch.ones_like(r, dtype=torch.bool)
    edge[..., P.BORDER : -P.BORDER, P.BORDER : -P.BORDER] = False
    assert torch.all(r[edge] == 0) and torch.any(r[~edge] != 0)
    assert P.pde_loss(r).item() == pytest.approx(r.abs().mean().item())


def test_forcing_definitions():
    shipped, consistent = P.RESIDUALS["shipped"], P.RESIDUALS["consistent"]
    assert shipped["forcing"] == 1.0 * 4.49996e00 * 3.88433e-03 and shipped["fd_dx"] == 1 / 240
    # -div(k grad u) = 1 with k = K_SCALE k', u = U_SCALE u'  =>  -div(k' grad u') = 1 / (K_SCALE U_SCALE)
    assert consistent["forcing"] == pytest.approx(1 / (K_SCALE * U_SCALE))
    assert shipped["forcing"] * consistent["forcing"] == pytest.approx(1.0)
    assert consistent["fd_dx"] == 1 / 241


def test_equation_is_the_expanded_form_and_the_forcing_enters_as_a_constant():
    darcy = P.Diffusion(T="u", time=False, dim=2, D="k", Q=0.0)
    assert str(darcy.equations["diffusion_u"]) == "-k*u__x__x - k*u__y__y - k__x*u__x - k__y*u__y"
    u, k, _ = manufactured(48, 1 / 240)

    def residual(forcing):
        pde = P.Diffusion(T="u", time=False, dim=2, D="k", Q=forcing)
        informer = P.PhysicsInformer(["diffusion_u"], pde, "finite_difference", fd_dx=1 / 240, device=torch.device("cpu"))
        return P.residual_field(informer, u, k)[0, 0, P.BORDER : -P.BORDER, P.BORDER : -P.BORDER]

    assert torch.allclose(residual(0.0) - residual(2.5), torch.full((44, 44), 2.5), atol=1e-4)


def test_batches_larger_than_one_are_rejected():
    u, k, _ = manufactured(32, 1 / 240)
    with pytest.raises(ValueError):
        P.residual_field(P.build_informer("shipped", torch.device("cpu")), u.repeat(2, 1, 1, 1), k.repeat(2, 1, 1, 1))


def test_gradient_reaches_the_field():
    u, k, _ = manufactured(32, 1 / 240)
    u.requires_grad_(True)
    P.pde_loss(P.residual_field(P.build_informer("consistent", torch.device("cpu")), u, k)).backward()
    assert torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0


def test_physics_device():
    assert P.physics_device(torch.device("mps")).type == "cpu"
    assert P.physics_device(torch.device("cpu")).type == "cpu"
    assert P.physics_device(torch.device("cuda")).type == "cuda"


def test_interface_mask_and_statistics():
    k = torch.ones(1, 1, 16, 16)
    k[..., 8:] = 3.0
    mask = P.interface_mask(k)[0, 0, 2:-2, 2:-2]
    assert mask[:, 5:7].all() and not mask[:, :5].any() and not mask[:, 7:].any()
    residual = torch.zeros(1, 1, 16, 16)
    residual[..., 2:-2, 2:-2] = torch.where(P.interface_mask(k)[..., 2:-2, 2:-2], 4.0, -1.0)
    stats = P.residual_statistics(residual, k)
    assert stats["abs_mean_interface"] == pytest.approx(4.0) and stats["abs_mean_smooth"] == pytest.approx(1.0)
    assert stats["interface_fraction"] == pytest.approx(2 / 12)
    assert stats["l1"] == pytest.approx(residual.abs().mean().item())
    assert stats["signed_mean"] == pytest.approx((4 * 2 - 10) / 12)
    assert stats["rms"] == pytest.approx(math.sqrt((16 * 2 + 10) / 12))
