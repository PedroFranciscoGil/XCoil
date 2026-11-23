"""Exact per-coil curvature of the coil Lagrangian (problem_h.exact_blockdiag, HConfig.lagrangian_hessian).

The Gauss-Newton matrix misses the second-order flux term S = sum_q r_q d2 r_q, which at the LP-QA
solution is ~30x larger than GN along the near-null directions of J^T J; undamped Newton steps
then overshoot and cycle (P6 stationary-reference diagnosis). exact_blockdiag adds, per coil,
2 c_f S (closed form for the flux residual, autodiff for the regularizer residuals) and the
z-weighted constraint Hessians. These tests check each piece against autodiff.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, seeds, problem_h as ph
from xcoil.homotopy.config import for_device
from xcoil.constraints import pointwise as pw
from xcoil.field import jacobian as jac, residuals as rs


@pytest.fixture(scope="module")
def lpqa():
    cfg = for_device("lpqa", w_msc_final=1e-3)          # nonzero MSC residual exercises that path
    b = bd.load(cfg.surface)
    seed = seeds.build(b, 4, ph.limits_from(cfg), cfg.basis, cfg.order, cfg.nq_opt, cfg.grid_dist)
    T = ph.build_tables(b, cfg, seed, grid=(16, 32))
    x = jnp.asarray(seed.x) + 1e-2 * jnp.asarray(np.random.default_rng(0).normal(size=seed.x.size))
    return cfg, T, x


@pytest.mark.bigmem
def test_flux_S_blocks_vs_autodiff(lpqa):
    cfg, T, x = lpqa
    X, n, w = ph.surface(T, 0.6); cq = rs.surface_coeffs(w, T.B_ref)
    rbar = jnp.asarray(np.random.default_rng(1).normal(size=X.shape[0]))
    S = jax.jit(lambda x: jac.residual_hessian_blocks(x, T.cb, T.sym, X, n, cq, T.I0, rbar))(x)
    H = jax.jit(jax.hessian(lambda y: rbar @ jac.residual_and_jacobian(y, T.cb, T.sym, X, n, cq, T.I0)[0]))(x)
    Dc = x.size // 4; sc = float(jnp.max(jnp.abs(H)))
    for c in range(4):
        blk = slice(c * Dc, (c + 1) * Dc)
        assert float(jnp.max(jnp.abs(H[blk, blk] - S[c]))) <= 1e-12 * sc
        for c2 in range(4):                              # exactly block-diagonal over coils
            if c2 != c:
                assert float(jnp.max(jnp.abs(H[blk, c2 * Dc:(c2 + 1) * Dc]))) == 0.0


@pytest.mark.bigmem
def test_exact_blockdiag_directional(lpqa):
    """v' H v of the unclipped model vs eager forward-over-forward of 2 c_f rbar.r + z.g along
    single-coil directions. Coil-coil multipliers are kept at barrier size mu/s (their cross-coil
    couplings are dropped by design); all other rows get O(1) multipliers."""
    cfg, T, x = lpqa
    p = (0.7, 18.2)
    z = 1e-6 / jnp.maximum(-ph.cons(T, x, p), 1e-12)      # barrier-size multipliers mu / s, s = -g
    bl = pw.blocks(4, 128); rng = np.random.default_rng(2)
    for k in ("L", "msc", "cs", "curv"):
        z = z.at[bl[k]].set(jnp.asarray(rng.uniform(0, 1, z[bl[k]].size)))
    r = ph.resid(T, x, p)
    Hm = np.asarray(jax.jit(lambda y: ph.exact_blockdiag(T, y, p, z, r, clip=False))(x))
    F = lambda y: 2.0 / cfg.f_ref * jnp.dot(r, ph.resid(T, y, p)) + jnp.dot(z, ph.cons(T, y, p))
    Dc = x.size // 4
    for c in range(4):
        v = np.zeros(x.size); v[c * Dc:(c + 1) * Dc] = rng.normal(size=Dc); v = jnp.asarray(v / np.linalg.norm(v))
        # eager on purpose: jitted forward-over-forward of the curvature rows is numerically
        # unreliable in XLA (returns ~1e13 for O(1e4) values); forward-over-reverse (used by the
        # model) is not affected
        ref = float(jax.jvp(lambda y: jax.jvp(F, (y,), (v,))[1], (x,), (v,))[1])
        mod = float(v @ Hm @ v)
        assert abs(ref - mod) <= 1e-4 * abs(ref), (c, ref, mod)


@pytest.mark.bigmem
def test_exact_blockdiag_psd(lpqa):
    cfg, T, x = lpqa
    p = (0.4, 16.0)
    z = 1e-6 / jnp.maximum(-ph.cons(T, x, p), 1e-12)
    H = jax.jit(lambda y: ph.exact_blockdiag(T, y, p, z, ph.resid(T, y, p)))(x)
    assert float(jnp.min(jnp.linalg.eigvalsh(H))) >= -1e-9 * float(jnp.max(jnp.abs(H)))
