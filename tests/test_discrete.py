"""Verification of the differentiable solver-consistent residual (G4, ``pino_darcy.discrete``)."""

import math

import numpy as np
import pytest
import torch

from pino_darcy import discrete as Dz
from pino_darcy import physics as P
from pino_darcy import reference as R
from pino_darcy.data import K_SCALE, U_SCALE
from test_reference import two_valued

f64 = torch.float64


def t(x):
    return torch.as_tensor(np.asarray(x), dtype=f64)


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_the_numpy_residual_of_the_reference_module(seed):
    """Same numbers as ``reference.solver_residual`` (itself checked against the dataset) for arbitrary fields."""
    K = 41
    coeff = two_valued(K, seed)
    u = np.random.default_rng(seed).normal(size=(2, K, K))
    a = Dz.node_coefficient(t(coeff))
    expected = R.solver_residual(u, coeff, forcing=1.0)
    full = Dz.solver_residual(t(u), a, 1.0, margin=0, extend=False).numpy()
    assert np.abs(full - expected).max() < 1e-9 * np.abs(expected).max()
    cropped = Dz.solver_residual(t(u[:, :-1, :-1]), a, 1.0, margin=0).numpy()
    assert np.abs(cropped - R.solver_residual(R.extend_crop(u[:, :-1, :-1]), coeff)).max() < 1e-9 * np.abs(expected).max()


def test_node_operator_is_the_sparse_matrix_of_the_solver():
    K = 23
    a = Dz.node_coefficient(t(two_valued(K, 3)))
    p = torch.zeros(K, K, dtype=f64)
    p[1:-1, 1:-1] = torch.randn(K - 2, K - 2, dtype=f64, generator=torch.Generator().manual_seed(0))
    A = R.node_matrix(a.numpy())
    assert np.allclose(Dz.node_operator(p, a).numpy().ravel(), A @ p[1:-1, 1:-1].numpy().ravel(), rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_solution_has_zero_residual_at_jumps(seed):
    """Discontinuous coefficient: the solver's own output, given as cells and as the upstream crop."""
    K = 61
    coeff = two_valued(K, seed)
    u = R.solve_reference(coeff)
    a = Dz.node_coefficient(t(coeff))
    mask = R.interface_nodes(coeff)
    r = Dz.solver_residual(t(u), a, 1.0, margin=0, extend=False).numpy()
    assert mask.any() and np.abs(r).max() < 1e-8
    # cropped and extended: the extrapolated edge leaves an error that decays away from it (non-local
    # inverse spline); beyond the margin it is below 1e-4 of the forcing, at jump nodes as well.
    # (On the 241 grid, results/audit/reference_check.json measures at most 7e-3 beyond 4 nodes.)
    r = Dz.solver_residual(t(u[:-1, :-1]), a, 1.0).numpy()
    inner = (slice(Dz.MARGIN, -Dz.MARGIN),) * 2
    assert mask[inner].any() and np.abs(r[inner]).max() < 1e-4
    assert np.all(r[: Dz.MARGIN] == 0) and np.all(r[:, -Dz.MARGIN :] == 0)


def manufactured(K):
    """p = sin(pi x) sin(pi y) (zero on the boundary) and a = 1 + 0.5 x y + sin(y) on the nodes,
    with -div(a grad p) analytically."""
    x = torch.linspace(0, 1, K, dtype=f64)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    s, c = torch.sin, torch.cos
    p = s(math.pi * X) * s(math.pi * Y)
    a = 1 + 0.5 * X * Y + s(Y)
    px, py = math.pi * c(math.pi * X) * s(math.pi * Y), math.pi * s(math.pi * X) * c(math.pi * Y)
    ax, ay = 0.5 * Y, 0.5 * X + c(Y)
    exact = -(ax * px + ay * py - 2 * math.pi**2 * a * p)
    return p, a, exact[1:-1, 1:-1]


def test_manufactured_solution_converges_at_second_order():
    errors = []
    for K in (33, 65, 129):
        p, a, exact = manufactured(K)
        errors.append((Dz.node_operator(p, a) - exact).abs().max().item())
    rates = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(1.9 < r < 2.1 for r in rates), (errors, rates)


def test_constant_coefficient_is_the_five_point_laplacian():
    K = 20
    p = torch.randn(K, K, dtype=f64, generator=torch.Generator().manual_seed(1))
    a = torch.full((K, K), 2.5, dtype=f64)
    lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * p[1:-1, 1:-1]) * (K - 1) ** 2
    assert torch.allclose(Dz.node_operator(p, a), -2.5 * lap, rtol=1e-12, atol=1e-8)


def test_operator_is_conservative_symmetric_and_positive():
    """Discrete divergence theorem and symmetry of the flux form, for a discontinuous coefficient."""
    K = 30
    a = Dz.node_coefficient(t(two_valued(K, 4)))
    g = torch.Generator().manual_seed(2)

    def interior():
        p = torch.zeros(K, K, dtype=f64)
        p[1:-1, 1:-1] = torch.randn(K - 2, K - 2, dtype=f64, generator=g)
        return p

    p, q = interior(), interior()
    Ap, Aq = Dz.node_operator(p, a), Dz.node_operator(q, a)
    assert torch.isclose((Ap * q[1:-1, 1:-1]).sum(), (Aq * p[1:-1, 1:-1]).sum(), rtol=1e-10)
    assert (Ap * p[1:-1, 1:-1]).sum() > 0
    # a field that is 1 on the interior: the sum of the operator is the flux through the boundary faces only
    one = torch.zeros(K, K, dtype=f64)
    one[1:-1, 1:-1] = 1
    boundary_flux = (K - 1) ** 2 * (
        ((a[1, 1:-1] + a[0, 1:-1]) / 2).sum() + ((a[-2, 1:-1] + a[-1, 1:-1]) / 2).sum()
        + ((a[1:-1, 1] + a[1:-1, 0]) / 2).sum() + ((a[1:-1, -2] + a[1:-1, -1]) / 2).sum()
    )
    assert torch.isclose(Dz.node_operator(one, a).sum(), boundary_flux, rtol=1e-12)


def test_gradcheck_with_respect_to_the_pressure():
    K = 14
    a = Dz.node_coefficient(t(two_valued(K, 5)))
    u = torch.randn(1, 1, K - 1, K - 1, dtype=f64, generator=torch.Generator().manual_seed(3), requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: Dz.solver_residual(v, a, 1.0, margin=2), (u,))


def test_directional_derivative_of_the_training_loss():
    """The L1 training loss: autograd gradient against a central finite difference along a random direction."""
    K = 41
    a = Dz.node_coefficient(t(two_valued(K, 6))) / K_SCALE
    g = torch.Generator().manual_seed(4)
    u = torch.randn(1, 1, K - 1, K - 1, dtype=f64, generator=g, requires_grad=True)
    v = torch.randn(1, 1, K - 1, K - 1, dtype=f64, generator=g)

    def loss(w):
        return P.pde_loss(Dz.solver_residual(w, a, P.RESIDUALS["consistent"]["forcing"]))

    loss(u).backward()
    eps = 1e-7
    with torch.no_grad():
        fd = (loss(u + eps * v) - loss(u - eps * v)) / (2 * eps)
    assert torch.isclose((u.grad * v).sum(), fd, rtol=1e-5)


def test_normalised_residual_is_the_physical_one_over_the_scale_product():
    K = 31
    coeff = two_valued(K, 7)
    u = np.random.default_rng(7).normal(size=(K - 1, K - 1)) * U_SCALE
    physical = Dz.solver_residual(t(u), Dz.node_coefficient(t(coeff)), 1.0)
    normalised = Dz.solver_residual(t(u / U_SCALE), Dz.node_coefficient(t(coeff / K_SCALE)), 1 / (K_SCALE * U_SCALE))
    assert torch.allclose(normalised, physical / (K_SCALE * U_SCALE), rtol=1e-9, atol=1e-6)


def test_single_precision_rounding_of_the_reference_is_small():
    """In float32 (the training precision) the residual of an exact solution stays far below the forcing."""
    K = 121
    coeff = two_valued(K, 8)
    u = R.solve_reference(coeff / K_SCALE, 1 / (K_SCALE * U_SCALE))
    a = Dz.node_coefficient(t(coeff / K_SCALE)).float()
    r = Dz.solver_residual(torch.as_tensor(u[:-1, :-1], dtype=torch.float32), a, 1 / (K_SCALE * U_SCALE))
    assert r.abs().mean().item() < 1e-2 * (1 / (K_SCALE * U_SCALE))


def test_physics_term_selects_operator_and_coefficient():
    K = 25
    coeff = torch.stack([t(two_valued(K, s)) for s in (0, 1)]).float()[:, None] / K_SCALE
    smooth = coeff * 0.5 + 0.5
    split = {"k": smooth[..., :-1, :-1], "coeff_full": coeff, "kcoeff_full": smooth}
    u = torch.randn(1, 1, K - 1, K - 1, generator=torch.Generator().manual_seed(9))
    cpu = torch.device("cpu")
    # cell-grid definitions with the model input are the previous function, unchanged
    for name in ("shipped", "consistent"):
        expected = P.residual_field(P.build_informer(name, cpu), u, split["k"][1:2])
        assert torch.equal(P.PhysicsTerm(name, cpu, split).residual(u, split["k"][1:2], 1), expected)
    # G3: same cell-grid residual, but with the two-valued coefficient of sample 1
    got = P.PhysicsTerm("consistent_coeff", cpu, split).residual(u, split["k"][1:2], 1)
    assert torch.equal(got, P.residual_field(P.build_informer("consistent", cpu), u, coeff[1:2, :, :-1, :-1]))
    # G4: solver operator with coeff, and the control with Kcoeff
    forcing = P.RESIDUALS["consistent"]["forcing"]
    for name, field in (("solver", coeff), ("solver_kcoeff", smooth)):
        got = P.PhysicsTerm(name, cpu, split).residual(u, split["k"][1:2], 1)
        expected = Dz.solver_residual(u.double(), Dz.node_coefficient(field[1:2].double()), forcing)
        assert got.shape == (1, 1, K - 2, K - 2)
        assert torch.allclose(got.double(), expected, rtol=1e-4, atol=1e-3)
    with pytest.raises(ValueError):
        P.PhysicsTerm("solver", cpu, {"k": split["k"]})
