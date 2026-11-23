"""Mixed precision (FP64 geometry and separations, FP32 kernels, FP64 accumulation) against the
FP64 reference on the W7-X production problem (fixed surface, t = 1), 3D-offset start:
flux residual |r|^2 to 1e-7 relative, flux Jacobian to 1e-3, distance constraints g to 1e-5
absolute (normalized rows), constraint Jacobian to 1e-3 relative; the default path (kdt=None)
is bitwise the FP64 code."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, problem_h as ph, seeds as seeds_mod
from xcoil.homotopy.config import for_device
from xcoil.homotopy import surface_path as sp
from xcoil.field import mixed as mx
from xcoil.constraints import pointwise as pw
from xcoil.fixed.starts import generate

LIM = dict(kappa_lim=2.5, msc_lim=1.5, dcc=0.3, dcs=0.35)


@pytest.fixture(scope="module")
def w7x():
    cfg = for_device("w7x", **LIM); b = bd.load(cfg.surface); lim = ph.limits_from(cfg)
    seed = seeds_mod.build(b, cfg.n_coils, lim, cfg.basis, cfg.order, cfg.nq_opt, cfg.grid_dist)
    T = ph.build_tables(b, cfg, seed)
    s = [s for s in generate(b, cfg.n_coils, cfg.order, 38.0, lim.dcs) if s.family == "offset_3d" and s.label == "L85"][0]
    p = (jnp.asarray(1.0), jnp.asarray(38.0))
    return T, jnp.asarray(s.x), p


def _rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def test_flux_residual_and_jacobian(w7x):
    T, x, p = w7x
    surf = ph.surface(T, p[0])
    r64, J64 = ph.resid_jac(T, x, p)
    rm, Jm = mx.stacked(x, p[0], T.cb, T.sym, surf, T.B_ref, T.I0, T.cfg, T.L_ref)
    rf = mx.stacked_residual(x, p[0], T.cb, T.sym, surf, T.B_ref, T.I0, T.cfg, T.L_ref)
    assert Jm.dtype == jnp.float64 and rf.dtype == jnp.float64
    assert abs(float(rf @ rf - r64 @ r64)) / float(r64 @ r64) < 1e-7
    assert abs(float(rm @ rm - r64 @ r64)) / float(r64 @ r64) < 1e-7
    assert _rel(Jm, J64) < 1e-3


def test_constraints(w7x):
    T, x, p = w7x
    g64, A64 = ph.cons_jac(T, x, p)
    gm, Am = ph.cons_jac(T, x, p, jnp.float32)
    assert gm.dtype == jnp.float64 and Am.dtype == jnp.float64
    assert float(jnp.max(jnp.abs(gm - g64))) < 1e-5
    assert _rel(Am, A64) < 1e-3
    g0, A0 = ph.cons_jac(T, x, p, None)                  # default path unchanged
    assert bool(jnp.all(g0 == g64)) and bool(jnp.all(A0 == A64))


@pytest.mark.parametrize("kdt", [None, jnp.float32])
def test_cc_active_rows_exact(w7x, kdt):
    """Row-restricted coil-coil Jacobian: selected rows equal the full block's rows, and the AL
    terms A^T h and A_a^T A_a (h > 0 rows only) are unchanged when all h > 0 rows are selected."""
    from xcoil.constraints import pointwise as pw
    from xcoil.homotopy import surface_path as sp
    from xcoil.curves import fourier as fr
    T, x, p = w7x
    C, nq = T.sym.C, T.cb.B0.shape[0]
    S = sp.eval_xyz(T.dist, p[0]); cap = ph.cap_at(T, p)
    g = ph.cons(T, x, p, kdt)
    cc = pw.blocks(C, nq)["cc"]
    t = jnp.quantile(g[cc], 0.8)                         # a band of active rows: the top 20 % of the cc rows
    h = jnp.maximum(0.0, g - t)
    hcc = h[cc]; n_act = int(jnp.sum(hcc > 0))
    assert 0 < n_act < C * nq
    K = max(n_act, 8)
    _, idx = jax.lax.top_k(hcc, K)
    A_full = pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, cap, kdt)
    A_part = pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, cap, kdt, cc_rows=(idx // nq, idx % nq))
    rows = cc.start + idx
    tol = 1e-12 if kdt is None else 1e-5                 # FP32 kernel: different reduction order
    err = float(jnp.max(jnp.abs(A_part[rows] - A_full[rows]))) / float(jnp.max(jnp.abs(A_full)))
    print(f"kdt={kdt} n_act={n_act} row err {err:.1e}")
    assert err <= tol
    act = (h > 0)[:, None]
    gr_f, gr_p = A_full.T @ h, A_part.T @ h
    H_f, H_p = (A_full * act).T @ (A_full * act), (A_part * act).T @ (A_part * act)
    assert float(jnp.max(jnp.abs(gr_p - gr_f))) <= tol * float(jnp.max(jnp.abs(gr_f)))
    assert float(jnp.max(jnp.abs(H_p - H_f))) <= tol * float(jnp.max(jnp.abs(H_f)))


def test_mixed_field_on(w7x):
    """Mixed-precision field (monitor) vs FP64 Biot-Savart on the flux grid."""
    from xcoil.field import residuals as rs
    T, x, p = w7x
    X, n, w = ph.surface(T, p[0])
    B64 = rs.field_on(x, T.cb, T.sym, X, T.I0)
    Bm = mx.field_on(x, T.cb, T.sym, X, T.I0)
    assert _rel(Bm, B64) < 1e-6
    q64 = jnp.abs(jnp.sum(B64 * n, 1)) / jnp.linalg.norm(B64, axis=1)
    qm = jnp.abs(jnp.sum(Bm * n, 1)) / jnp.linalg.norm(Bm, axis=1)
    assert float(jnp.max(jnp.abs(qm - q64))) < 1e-6
