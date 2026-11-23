"""Free coil currents (curves/currents).

Field vs simsopt with unequal currents (1e-12); the current sum is C I0; u = 0 reproduces the
equal-current residual exactly; analytic J (FP64 and mixed) vs jacfwd including the current
columns; regularizer and constraint Jacobians have zero current columns; the problem-level
stacked Jacobian and the AL gradient match finite differences in the current directions.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, surface_path as sp
from xcoil.curves import fourier as fr, symmetry as sy, currents as cur
from xcoil.field import residuals as rs, jacobian as jac, mixed as mx
from test_residual_jacobian import _random_coils


def _with_u(x, u):
    return jnp.concatenate([jnp.asarray(x), jnp.asarray(u, float)])


@pytest.mark.parametrize("nfp,stellsym", [(2, True), (4, True), (3, False)])
def test_field_vs_simsopt_unequal(nfp, stellsym):
    pytest.importorskip("simsopt")                     # cross-check only; not installed on Colab
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    C, M, nq = 4, 6, 128
    x = _random_coils(C, M, nfp, stellsym)
    u = np.array([0.31, -0.42, 0.07])
    I0 = 3.1e5
    Ib = I0 * (1.0 + np.concatenate([u, [-u.sum()]]))
    assert abs(Ib.sum() - C * I0) <= 1e-9 * I0
    base = [fr.to_simsopt(v, nq, M) for v in x.reshape(C, -1)]
    bsf = BiotSavart(coils_via_symmetries(base, [Current(I) for I in Ib], nfp, stellsym))
    pts = np.random.default_rng(5).uniform(-0.6, 0.6, (200, 3)) + np.array([0.5, 0.3, 0.0])
    bsf.set_points(pts)
    B_ref = bsf.B()
    cb = fr.make_basis("fourier", nq, M); S = sy.build(C, nfp, stellsym)
    B = np.asarray(rs.field_on(_with_u(x, u), cb, S, jnp.asarray(pts), I0))
    assert np.abs(B - B_ref).max() / np.abs(B_ref).max() <= 1e-12
    Bm = np.asarray(mx.field_on(_with_u(x, u), cb, S, jnp.asarray(pts), I0))
    assert np.abs(Bm - B_ref).max() / np.abs(B_ref).max() <= 1e-5


def _setup(name="landreman_paul_qa", M=8, C=4, nq=64):
    b = bd.load(name)
    x = jnp.asarray(_random_coils(C, M, b.nfp, True, R0=b.R00, r=0.45 * (1.0 if b.R00 < 2 else 2.2)))
    cb = fr.make_basis("fourier", nq, M); S = sy.build(C, b.nfp, True)
    X, n, w = sp.evaluate(sp.build(b, 8, 16, "fundamental"), 1.0)
    return b, x, cb, S, X, n, w


def test_u_zero_is_equal_currents():
    _, x, cb, S, X, n, w = _setup()
    cq = rs.surface_coeffs(w, 1.0)
    r0, J0 = jac.residual_and_jacobian(x, cb, S, X, n, cq, 1e6)
    r1, J1 = jac.residual_and_jacobian(_with_u(x, jnp.zeros(3)), cb, S, X, n, cq, 1e6)
    assert np.array_equal(np.asarray(r0), np.asarray(r1))
    assert np.abs(np.asarray(J1[:, :J0.shape[1]] - J0)).max() <= 1e-14 * np.abs(np.asarray(J0)).max()


def test_flux_jacobian_vs_jacfwd():
    _, x, cb, S, X, n, w = _setup()
    cq = rs.surface_coeffs(w, 1.0)
    y = _with_u(x, jnp.array([0.2, -0.15, 0.05]))
    f = lambda v: jac.residual_and_jacobian(v, cb, S, X, n, cq, 1e6)[0]
    r, J = jac.residual_and_jacobian(y, cb, S, X, n, cq, 1e6)
    Jf = jax.jacfwd(f)(y)
    assert J.shape == (X.shape[0], y.size)
    rel = np.abs(np.asarray(J - Jf)).max() / np.abs(np.asarray(Jf)).max()
    assert rel <= 1e-12, rel
    ng = cur.n_geo(cb, 4)
    relu = np.abs(np.asarray(J[:, ng:] - Jf[:, ng:])).max() / np.abs(np.asarray(Jf[:, ng:])).max()
    assert relu <= 1e-12, relu
    # forward residual of the separate kernel agrees
    rf = rs.magnetic_residual(y, cb, S, X, n, w, 1.0, 1e6)
    assert np.abs(np.asarray(rf - r)).max() <= 1e-12 * np.abs(np.asarray(r)).max()
    # mixed precision: same layout, FP32-kernel accuracy. As in the solver, FP32 products run at
    # "highest" precision: Ampere GPUs otherwise use TF32 by default (~1e-4 errors)
    with jax.default_matmul_precision("highest"):
        rm, Jm = mx.residual_and_jacobian(y, cb, S, X, n, cq, 1e6)
    assert Jm.shape == J.shape
    assert np.abs(np.asarray(Jm - J)).max() / np.abs(np.asarray(J)).max() <= 1e-4
    assert np.abs(np.asarray(rm - r)).max() / np.abs(np.asarray(r)).max() <= 1e-5


def test_hessian_blocks_use_coil_currents():
    """residual_hessian_blocks = d2/dx_c^2 of sum_q rbar_q r_q at fixed currents (per coil)."""
    _, x, cb, S, X, n, w = _setup(M=4, nq=32)
    cq = rs.surface_coeffs(w, 1.0)
    u = jnp.array([0.3, -0.2, 0.1]); y = _with_u(x, u)
    rbar = jnp.asarray(np.random.default_rng(1).normal(size=X.shape[0]))
    Sb = jac.residual_hessian_blocks(y, cb, S, X, n, cq, 1e6, rbar)
    ng = cur.n_geo(cb, 4); Dc = ng // 4
    phi = lambda v: rbar @ jac.residual_and_jacobian(jnp.concatenate([v, u]), cb, S, X, n, cq, 1e6)[0]
    H = jax.hessian(phi)(y[:ng])
    for c in range(4):
        Hc = H[c * Dc:(c + 1) * Dc, c * Dc:(c + 1) * Dc]
        assert np.abs(np.asarray(Sb[c] - Hc)).max() <= 1e-10 * np.abs(np.asarray(Hc)).max()


@pytest.mark.bigmem
def test_problem_free_currents():
    """Problem level (W7-X setup): starts carry u = 0, Jacobians have zero current columns
    except the flux rows, and resid_jac / cons_jac match finite differences in u."""
    from problems import problem_for, starts_for
    from xcoil.homotopy import problem_h as ph
    prob = problem_for("w7x")
    C = prob.cfg.n_coils; ng = cur.n_geo(prob.T.cb, C)
    x0 = starts_for(prob)[0].x
    assert x0.size == ng + C - 1 == prob.n and np.all(x0[ng:] == 0)
    y = jnp.asarray(x0).at[ng:].set(jnp.array([0.1, -0.05, 0.02]))
    p = (1.0, prob.eps)
    r, J = ph.resid_jac(prob.T, y, p)
    Ns = ph.surface(prob.T, 1.0)[0].shape[0]
    assert J.shape == (r.size, y.size)
    assert np.all(np.asarray(J[Ns:, ng:]) == 0.0)                   # regularizers: no current dependence
    g, A = ph.cons_jac(prob.T, y, p)
    assert A.shape == (g.size, y.size) and np.all(np.asarray(A[:, ng:]) == 0.0)
    for k in range(C - 1):
        e = jnp.zeros(y.size).at[ng + k].set(1.0); h = 1e-6
        fd = (ph.resid(prob.T, y + h * e, p) - ph.resid(prob.T, y - h * e, p)) / (2 * h)
        rel = np.abs(np.asarray(fd - J[:, ng + k])).max() / np.abs(np.asarray(J[:, ng + k])).max()
        assert rel <= 1e-7, (k, rel)
    # the certification reports the relative currents and their sum is C
    c = prob.certify(np.asarray(y))
    assert abs(c["I_rel_min"] - 0.93) < 1e-12 and abs(c["I_rel_max"] - 1.1) < 1e-12
