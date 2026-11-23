"""Periodic cubic B-spline derivatives vs finite differences < 1e-9 (FP64).

Also checks periodic closure (the curve and its derivatives are C^2-periodic) and that the
precomputed basis matrices reproduce a direct partition-of-unity evaluation.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

import xcoil  # noqa: F401  (enables x64)
from xcoil.geometry import periodic_bspline as pb


def _interior_points(N, M, margin=5e-3, seed=0):
    """Random s away from knots (3rd derivative is discontinuous at knots, and the FD stencil
    must stay within a single cubic segment)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 1, size=8 * M)
    knot_dist = np.min(np.abs((s[:, None] * N) - np.round(s[:, None] * N)), axis=1) / N
    s = s[knot_dist > margin][:M]
    assert s.shape[0] == M, "not enough interior points; increase pool"
    return jnp.asarray(np.sort(s))


def _fd4(f, s, h):
    """4th-order central difference; EXACT (to roundoff) for a cubic when the stencil stays
    within one segment, since f^(4)=f^(5)=0 there."""
    return (-f(s + 2 * h) + 8 * f(s + h) - 8 * f(s - h) + f(s - 2 * h)) / (12 * h)


def test_derivatives_vs_fd():
    N, dim, M = 16, 3, 64
    rng = np.random.default_rng(1)
    P = jnp.asarray(rng.standard_normal((N, dim)))
    s = _interior_points(N, M)

    val, d1, d2, d3 = pb.evaluate(P, s, N)

    def value(sq):
        return pb.evaluate(P, sq, N)[0]

    def d1_fn(sq):
        return pb.evaluate(P, sq, N)[1]

    def d2_fn(sq):
        return pb.evaluate(P, sq, N)[2]

    h = 1e-4  # stencil s +- 2h stays inside a segment (points are >5e-3/N from knots)
    fd1 = _fd4(value, s, h)
    fd2 = _fd4(d1_fn, s, h)
    fd3 = _fd4(d2_fn, s, h)

    def rel(a, b):
        return float(jnp.max(jnp.abs(a - b) / jnp.maximum(1.0, jnp.abs(b))))

    e1, e2, e3 = rel(d1, fd1), rel(d2, fd2), rel(d3, fd3)
    print(f"[bspline] rel err  d1={e1:.2e}  d2={e2:.2e}  d3={e3:.2e}  (tolerance 1e-9)")
    assert e1 < 1e-9 and e2 < 1e-9 and e3 < 1e-9, (e1, e2, e3)


def test_partition_of_unity():
    # cubic B-spline basis sums to 1 everywhere (constant curve reproduces the control value)
    N = 20
    s = jnp.linspace(0.0, 1.0, 200, endpoint=False)
    B0 = pb.basis_matrices(s, N)[0]
    row_sums = np.asarray(jnp.sum(B0, axis=1))
    err = float(np.max(np.abs(row_sums - 1.0)))
    print(f"[bspline] partition-of-unity max|sum-1| = {err:.2e}")
    assert err < 1e-12, err


def _one_sided_at_knot(P, i, N, side):
    """Exact value/d1/d2/d3 at knot i/N from the segment on the given side ('left'|'right')."""
    t = 1.0 if side == "left" else 0.0
    k = (i - 1) if side == "left" else i          # segment index owning this side of the knot
    w, w_t, w_tt, w_ttt = pb.basis_weights(jnp.asarray(t))
    idx = jnp.mod(k + pb._OFFS, N)
    Pk = P[idx]                                   # (4, dim)
    val = w @ Pk
    d1 = (N) * (w_t @ Pk)
    d2 = (N * N) * (w_tt @ Pk)
    d3 = (N ** 3) * (w_ttt @ Pk)
    return val, d1, d2, d3


def test_periodic_closure():
    # exact C^2 continuity across every knot (incl. the s=0/1 seam): value, d1, d2 match from
    # both sides to machine precision; d3 is expected to be discontinuous (piecewise constant).
    N, dim = 16, 3
    rng = np.random.default_rng(2)
    P = jnp.asarray(rng.standard_normal((N, dim)))
    worst = [0.0, 0.0, 0.0]
    d3_jump = 0.0
    for i in range(N):                            # i=0 is the periodic seam
        L = _one_sided_at_knot(P, i, N, "left")
        R = _one_sided_at_knot(P, i, N, "right")
        for order in range(3):
            worst[order] = max(worst[order], float(jnp.max(jnp.abs(L[order] - R[order]))))
        d3_jump = max(d3_jump, float(jnp.max(jnp.abs(L[3] - R[3]))))
    print(f"[bspline] C^2 closure max gap  value={worst[0]:.2e}  d1={worst[1]:.2e}  d2={worst[2]:.2e}")
    print(f"[bspline] d3 jump at knots = {d3_jump:.2e} (expected > 0: piecewise-constant d3)")
    assert worst[0] < 1e-12 and worst[1] < 1e-12 and worst[2] < 1e-12, worst
    assert d3_jump > 1e-6, "d3 should be discontinuous at knots"


if __name__ == "__main__":
    test_derivatives_vs_fd()
    test_partition_of_unity()
    test_periodic_closure()
    print("[bspline] passed")
