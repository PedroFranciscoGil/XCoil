"""Uniform periodic cubic B-spline library.

A closed coil coordinate is a uniform cubic B-spline on the periodic parameter s in [0, 1)
with N control points and knots at i/N. Cubic B-splines have local support (4 nonzero basis
functions at any s) and C^2 continuity; the third derivative is piecewise constant.

Curve evaluation is `value = B0 @ P` with precomputed basis matrices B0..B3 (value and first
three s-derivatives) for a fixed quadrature grid, so the hot path is dense matmuls. Periodicity
is handled by wrapping the 4 local control-point indices modulo N.

Analytic derivatives match finite differences to < 1e-9 relative in FP64
(tests/test_bspline.py).
"""
from __future__ import annotations

import jax.numpy as jnp

# local offsets of the 4 control points influencing a segment: k-1, k, k+1, k+2
_OFFS = jnp.array([-1, 0, 1, 2])


def basis_weights(t):
    """Uniform cubic B-spline weights and their t-derivatives on local t in [0, 1).

    Returns (w, w_t, w_tt, w_ttt), each shape (..., 4), for the four control points
    [P_{k-1}, P_k, P_{k+1}, P_{k+2}]. Value = sum_j w_j P_j (derivatives wrt local t).
    """
    t = jnp.asarray(t)
    o = jnp.ones_like(t)
    t2 = t * t
    t3 = t2 * t
    w = jnp.stack([(1 - 3 * t + 3 * t2 - t3),
                   (4 - 6 * t2 + 3 * t3),
                   (1 + 3 * t + 3 * t2 - 3 * t3),
                   t3], axis=-1) / 6.0
    w_t = jnp.stack([(-3 + 6 * t - 3 * t2),
                     (-12 * t + 9 * t2),
                     (3 + 6 * t - 9 * t2),
                     (3 * t2)], axis=-1) / 6.0
    w_tt = jnp.stack([(6 - 6 * t),
                      (-12 + 18 * t),
                      (6 - 18 * t),
                      (6 * t)], axis=-1) / 6.0
    w_ttt = jnp.stack([-6 * o, 18 * o, -18 * o, 6 * o], axis=-1) / 6.0
    return w, w_t, w_tt, w_ttt


def basis_matrices(s, N):
    """Precompute basis matrices B0..B3.

    s: (M,) evaluation parameters in [0, 1). Returns [B0, B1, B2, B3], each (M, N), with the
    chain-rule scale for s-derivatives folded in (u = s*N, dt/ds = N):
        value      = B0 @ P
        d/ds       = B1 @ P
        d^2/ds^2   = B2 @ P
        d^3/ds^3   = B3 @ P
    """
    s = jnp.asarray(s)
    M = s.shape[0]
    u = s * N
    k = jnp.floor(u).astype(jnp.int64)
    t = u - k
    w, w_t, w_tt, w_ttt = basis_weights(t)                 # (M, 4)
    idx = jnp.mod(k[:, None] + _OFFS[None, :], N)          # (M, 4) periodic wrap
    rows = jnp.repeat(jnp.arange(M), 4)
    cols = idx.reshape(-1)

    def mat(weights, scale):
        vals = (weights * scale).reshape(-1)
        return jnp.zeros((M, N), dtype=weights.dtype).at[rows, cols].add(vals)

    return [mat(w, 1.0), mat(w_t, N), mat(w_tt, N * N), mat(w_ttt, N * N * N)]


def evaluate(P, s, N):
    """Evaluate a periodic cubic B-spline curve and its first three s-derivatives.

    P: (N, dim) control points. s: (M,). Returns (value, d1, d2, d3), each (M, dim).
    """
    B0, B1, B2, B3 = basis_matrices(s, N)
    return B0 @ P, B1 @ P, B2 @ P, B3 @ P
