"""The discrete problem that the reference pressure of ``Darcy_241`` solves.

The FNO data-generation script ``data_generation/darcy/solve_gwf.m`` (Li et al., 2021)
takes the coefficient on the K x K cell centres ``(i + 1/2) / K``, interpolates it with a
not-a-knot cubic spline to the K x K nodes ``i / (K - 1)``, solves

    -div(a grad p) = f   on the interior nodes,   p = 0 on the boundary nodes,

with the conservative five-point scheme (arithmetic face means, spacing 1 / (K - 1)), and
interpolates ``p`` back to the cell centres with the same spline. This module is a NumPy
re-implementation of that procedure, written for this audit, and the residual of a
cell-centred field under the same discrete operator (``solver_residual``).
``scripts/verify_reference.py`` checks it against the dataset.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.interpolate import make_interp_spline
from scipy.ndimage import maximum_filter, minimum_filter


def cell_centres(K: int) -> np.ndarray:
    return (np.arange(K) + 0.5) / K


def nodes(K: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, K)


def spline_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Matrix of 1-D not-a-knot cubic spline interpolation from ``source`` to ``target`` points.

    ``interp2(..., 'spline')`` of MATLAB on a grid is this interpolation along each axis.
    """
    return make_interp_spline(source, np.eye(len(source)), k=3, bc_type="not-a-knot", axis=0)(target)


def interpolate(field: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 1-D interpolation matrix along both axes of a ``[..., K, K]`` array."""
    return matrix @ field @ matrix.T


def _faces(a: np.ndarray) -> tuple:
    """Arithmetic face means of the node coefficient at the interior nodes: north, south, west, east."""
    c = a[..., 1:-1, 1:-1]
    return (
        (c + a[..., :-2, 1:-1]) / 2,
        (c + a[..., 2:, 1:-1]) / 2,
        (c + a[..., 1:-1, :-2]) / 2,
        (c + a[..., 1:-1, 2:]) / 2,
    )


def node_matrix(a: np.ndarray) -> sp.csr_matrix:
    """Sparse matrix of the scheme on the interior nodes, Dirichlet zero on the boundary nodes."""
    K = a.shape[0]
    n = K - 2
    wn, ws, ww, we = (w.ravel() for w in _faces(a))
    idx = np.arange(n * n).reshape(n, n)
    rows = [idx.ravel()]
    cols = [idx.ravel()]
    vals = [wn + ws + ww + we]
    for w, shift in ((ws, (1, 0)), (we, (0, 1))):
        w = w.reshape(n, n)[: n - shift[0], : n - shift[1]].ravel()
        here = idx[: n - shift[0], : n - shift[1]].ravel()
        there = idx[shift[0] :, shift[1] :].ravel()
        rows += [here, there]
        cols += [there, here]
        vals += [-w, -w]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return A * (K - 1) ** 2


def solve_reference(coeff: np.ndarray, forcing: float = 1.0) -> np.ndarray:
    """Cell-centred solution of ``-div(coeff grad u) = forcing`` as computed by ``solve_gwf.m``."""
    K = coeff.shape[0]
    to_nodes = spline_matrix(cell_centres(K), nodes(K))
    a = interpolate(np.asarray(coeff, dtype=np.float64), to_nodes)
    p = np.zeros((K, K))
    p[1:-1, 1:-1] = spl.spsolve(node_matrix(a).tocsc(), np.full((K - 2) ** 2, float(forcing))).reshape(K - 2, K - 2)
    return interpolate(p, spline_matrix(nodes(K), cell_centres(K)))


def extend_crop(u: np.ndarray) -> np.ndarray:
    """Extend ``[..., K - 1, K - 1]`` fields (the upstream crop drops the last row and column) to
    ``[..., K, K]`` by cubic extrapolation. Only for ``solver_residual`` away from that edge: the
    inverse spline is non-local, so nodes within a margin of the extrapolated edge are not exact
    (``scripts/verify_reference.py`` measures the margin).
    """
    u = np.asarray(u, dtype=np.float64)
    row = 4 * u[..., -1:, :] - 6 * u[..., -2:-1, :] + 4 * u[..., -3:-2, :] - u[..., -4:-3, :]
    u = np.concatenate([u, row], axis=-2)
    col = 4 * u[..., :, -1:] - 6 * u[..., :, -2:-1] + 4 * u[..., :, -3:-2] - u[..., :, -4:-3]
    return np.concatenate([u, col], axis=-1)


def solver_residual(u: np.ndarray, coeff: np.ndarray, forcing: float = 1.0) -> np.ndarray:
    """Residual of cell-centred fields ``u`` (``[..., K, K]``) under the discrete operator of ``solve_gwf.m``.

    ``u`` is mapped to the nodes by inverting the node-to-cell spline, and
    ``-div_h(a grad_h p) - forcing`` is evaluated on the interior nodes. Shape ``[..., K - 2, K - 2]``,
    in the units of ``forcing``. The reference pressure gives zero up to rounding.
    """
    K = coeff.shape[-1]
    to_nodes = spline_matrix(cell_centres(K), nodes(K))
    from_nodes_inverse = np.linalg.inv(spline_matrix(nodes(K), cell_centres(K)))
    a = interpolate(np.asarray(coeff, dtype=np.float64), to_nodes)
    p = interpolate(np.asarray(u, dtype=np.float64), from_nodes_inverse)
    wn, ws, ww, we = _faces(a)
    c = p[..., 1:-1, 1:-1]
    flux = wn * (c - p[..., :-2, 1:-1]) + ws * (c - p[..., 2:, 1:-1]) + ww * (c - p[..., 1:-1, :-2]) + we * (c - p[..., 1:-1, 2:])
    return (K - 1) ** 2 * flux - forcing


def interface_nodes(coeff: np.ndarray) -> np.ndarray:
    """Interior nodes near a jump of ``coeff``. Shape ``[..., K-2, K-2]``.

    Node ``i / (K - 1)`` lies within half a cell of the centre of cell ``i``; a node counts as an
    interface node when ``coeff`` is not constant on the 5 x 5 cells around that cell (the spline
    coefficient varies over about two cells on each side of a jump).
    """
    size = (1,) * (coeff.ndim - 2) + (5, 5)
    spread = maximum_filter(coeff, size=size, mode="nearest") != minimum_filter(coeff, size=size, mode="nearest")
    return spread[..., 1:-1, 1:-1]
