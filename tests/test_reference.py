"""The re-implementation of the FNO data-generation solver and its discrete residual."""

import numpy as np
import pytest

from pino_darcy import physics as P
from pino_darcy import reference as R
from pino_darcy.data import K_SCALE, U_SCALE


def two_valued(K, seed=0):
    """Piecewise-constant 3/12 coefficient with straight and curved jumps, as in the dataset."""
    rng = np.random.default_rng(seed)
    x = R.cell_centres(K)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phase = rng.uniform(0, 2 * np.pi, 3)
    field = np.sin(3 * X + phase[0]) + np.cos(4 * Y + phase[1]) + 0.5 * np.sin(5 * X * Y + phase[2])
    return np.where(field > 0, 12.0, 3.0)


def test_spline_and_extension_are_exact_for_cubics():
    K = 17
    xc, xn = R.cell_centres(K), R.nodes(K)
    cubic = lambda t: 1 - 2 * t + 3 * t**2 - 4 * t**3  # noqa: E731
    assert np.allclose(R.spline_matrix(xc, xn) @ cubic(xc), cubic(xn), atol=1e-12)
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    u = cubic(X) * cubic(Y)
    assert np.allclose(R.extend_crop(u[:-1, :-1]), u, atol=1e-10)


def test_node_matrix_is_symmetric_positive_definite():
    a = R.interpolate(two_valued(12), R.spline_matrix(R.cell_centres(12), R.nodes(12)))
    A = R.node_matrix(a).toarray()
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A).min() > 0


@pytest.mark.parametrize("seed", [0, 1])
def test_the_solution_has_zero_solver_residual_at_jumps_too(seed):
    coeff = two_valued(41, seed)
    u = R.solve_reference(coeff)
    r = np.abs(R.solver_residual(u, coeff))
    mask = R.interface_nodes(coeff)
    assert mask.any() and (~mask).any()
    assert r.max() < 1e-8


def test_cell_grid_residual_of_the_solution_is_large_only_at_jumps():
    """The finite-difference residual of the training loss does not vanish for the solver's output
    near a jump, whatever the forcing: the operators differ there, not the data."""
    K = 41
    coeff = two_valued(K)
    u = R.solve_reference(coeff)
    h = 1 / K
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2
    c = coeff[1:-1, 1:-1]
    constant = (coeff[2:, 1:-1] == c) & (coeff[:-2, 1:-1] == c) & (coeff[1:-1, 2:] == c) & (coeff[1:-1, :-2] == c)
    r = np.abs(-c * lap - 1)
    far = constant & ~R.interface_nodes(coeff)  # at least two cells from a jump
    assert np.median(r[far]) < 0.05
    assert r[~constant].mean() > 1.0


def test_normalised_forcing_is_the_reciprocal_of_the_scale_product():
    """-div(k grad u) = 1 with k = K_SCALE k', u = U_SCALE u' is -div(k' grad u') = 1 / (K_SCALE U_SCALE)."""
    coeff = two_valued(25)
    u = R.solve_reference(coeff, 1.0)
    u_scaled = R.solve_reference(coeff / K_SCALE, 1 / (K_SCALE * U_SCALE))
    assert np.allclose(u_scaled, u / U_SCALE, rtol=1e-10, atol=1e-12)
    assert P.RESIDUALS["consistent"]["forcing"] == pytest.approx(1 / (K_SCALE * U_SCALE))
    # the forcing of the upstream script is the product, which solves a different problem
    u_shipped = R.solve_reference(coeff / K_SCALE, P.RESIDUALS["shipped"]["forcing"])
    assert np.allclose(u_shipped * (1 / (K_SCALE * U_SCALE)) / P.RESIDUALS["shipped"]["forcing"], u / U_SCALE)
    assert u_shipped.max() < 1e-3 * (u / U_SCALE).max()


def test_interface_nodes_follow_the_jumps():
    coeff = np.full((20, 20), 3.0)
    assert not R.interface_nodes(coeff).any()
    coeff[:, 10:] = 12.0
    mask = R.interface_nodes(coeff)
    assert mask.shape == (18, 18)
    # node j (interior index j - 1) is flagged when cells j-2 .. j+2 straddle the jump at cell 10
    assert np.array_equal(np.flatnonzero(mask[0]) + 1, np.arange(8, 12))
    assert np.array_equal(R.interface_nodes(np.stack([coeff, coeff])), np.stack([mask, mask]))
