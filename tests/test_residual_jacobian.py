"""Residual and analytic Jacobian.

B vs closed-form on-axis circle (1e-14); vs simsopt BiotSavart (1e-12); analytic J vs jacfwd
(1e-12 relative, incl. symmetry folding); fundamental-domain f_B == full-torus f_B (1e-12);
regularizer Jacobian vs jacfwd; peak memory at production P measured (compiled program).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, surface_path as sp
from xcoil.homotopy.config import HConfig
from xcoil.curves import fourier as fr, symmetry as sy
from xcoil.field import residuals as rs, jacobian as jac


def _random_coils(C, M, nfp, stellsym, R0=1.0, r=0.45, amp=0.03, seed=0):
    """Planar-ish loops around the torus with random Fourier perturbations."""
    rng = np.random.default_rng(seed)
    K = 2 * M + 1; xs = []
    phis = sy.base_toroidal_angles(C, nfp, stellsym)
    for c in range(C):
        X = np.zeros((3, K))
        cph, sph = np.cos(phis[c]), np.sin(phis[c])
        X[0, 0], X[1, 0] = R0 * cph, R0 * sph                      # centre
        X[0, 2], X[1, 2] = r * cph, r * sph                        # cos(2 pi s) radial
        X[2, 1] = r                                                # sin(2 pi s) vertical
        X += amp * rng.normal(size=(3, K)) / (1 + np.arange(K))
        xs.append(X.ravel())
    return np.concatenate(xs)


def test_bs_circle():
    R, I, M = 1.3, 2.5e5, 3
    K = 2 * M + 1; X = np.zeros((3, K)); X[0, 2] = R; X[1, 1] = R
    cb = fr.make_basis("fourier", 128, M)
    S = sy.build(1, 1, False)
    z = np.array([0.0, 0.4, 1.0, 2.0, -0.7])
    pts = jnp.asarray(np.stack([np.zeros_like(z), np.zeros_like(z), z], 1))
    B = np.asarray(rs.field_on(jnp.asarray(X.ravel()), cb, S, pts, I))
    Bz = 4e-7 * np.pi * I * R ** 2 / (2 * (R ** 2 + z ** 2) ** 1.5)
    assert np.abs(B[:, 2] - Bz).max() / np.abs(Bz).max() <= 1e-14
    assert np.abs(B[:, :2]).max() / np.abs(Bz).max() <= 1e-14
    # the analytic residual kernel reproduces the same field component
    r, _ = jac.residual_and_jacobian(jnp.asarray(X.ravel()), cb, S, pts,
                                     jnp.tile(jnp.array([[0.0, 0, 1]]), (5, 1)), jnp.ones(5), I)
    assert np.abs(np.asarray(r) - Bz).max() / np.abs(Bz).max() <= 1e-14


@pytest.mark.parametrize("nfp,stellsym", [(2, True), (4, True), (3, False)])
def test_bs_vs_simsopt(nfp, stellsym):
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    C, M, nq = 4, 6, 128
    x = _random_coils(C, M, nfp, stellsym)
    base = [fr.to_simsopt(v, nq, M) for v in x.reshape(C, -1)]
    I0 = 3.1e5
    bsf = BiotSavart(coils_via_symmetries(base, [Current(I0) for _ in range(C)], nfp, stellsym))
    pts = np.random.default_rng(5).uniform(-0.6, 0.6, (200, 3)) + np.array([0.5, 0.3, 0.0])
    bsf.set_points(pts)
    B_ref = bsf.B()
    B = np.asarray(rs.field_on(jnp.asarray(x), fr.make_basis("fourier", nq, M),
                               sy.build(C, nfp, stellsym), jnp.asarray(pts), I0))
    assert np.abs(B - B_ref).max() / np.abs(B_ref).max() <= 1e-12


@pytest.mark.parametrize("name,nfp_override,stellsym,M", [("landreman_paul_qa", None, True, 8),
                                                          ("w7x", None, True, 12),
                                                          ("landreman_paul_qa", 3, False, 5)])
def test_jacobian_vs_jacfwd(name, nfp_override, stellsym, M):
    b = bd.load(name)
    nfp = nfp_override or b.nfp
    C, nq = 4, 64
    x = jnp.asarray(_random_coils(C, M, nfp, stellsym, R0=b.R00, r=0.45 * b.R00 ** 0 *
                                  (1.0 if b.R00 < 2 else 2.2)))
    cb = fr.make_basis("fourier", nq, M); S = sy.build(C, nfp, stellsym)
    p = sp.build(b, 8, 16, "fundamental")
    X, n, w = sp.evaluate(p, 0.7)
    I0 = -1e6
    B_ref = float(rs.B_ref_from_seed(x, cb, S, X, w, I0))
    cq = rs.surface_coeffs(w, B_ref)
    r_a, J_a = jac.residual_and_jacobian(x, cb, S, X, n, cq, I0)
    fwd = lambda v: rs.magnetic_residual(v, cb, S, X, n, w, B_ref, I0)
    r_f = fwd(x); J_f = jax.jacfwd(fwd)(x)
    assert float(jnp.max(jnp.abs(r_a - r_f)) / jnp.max(jnp.abs(r_f))) <= 1e-13
    err = float(jnp.max(jnp.abs(J_a - J_f)) / jnp.max(jnp.abs(J_f)))
    print(f"{name} nfp={nfp} stellsym={stellsym} M={M}: |J_analytic - jacfwd|/|J| = {err:.2e}")
    assert err <= 1e-12


@pytest.mark.parametrize("name", ["landreman_paul_qa", "w7x", "qh_reactor"])
def test_fundamental_domain(name):
    b = bd.load(name)
    C, M, nq = 4, 8, 64
    x = jnp.asarray(_random_coils(C, M, b.nfp, True, R0=b.R00, r=(0.45 if b.R00 < 2 else
                                                                   (2.2 if b.R00 < 10 else 3.5))))
    cb = fr.make_basis("fourier", nq, M); S = sy.build(C, b.nfp, True)
    nphi, nth = 8, 16
    for t in (0.0, 0.55, 1.0):
        Xf, nf, wf = sp.evaluate(sp.build(b, nphi, nth, "fundamental"), t)
        Xt, nt, wt = sp.evaluate(sp.build(b, 2 * b.nfp * nphi, nth, "full"), t)
        B_ref = float(rs.B_ref_from_seed(x, cb, S, Xt, wt, 1e6))
        fB = lambda X, n, w: float(jnp.sum(rs.magnetic_residual(x, cb, S, X, n, w, B_ref, 1e6) ** 2))
        a, bb = fB(Xf, nf, wf), fB(Xt, nt, wt)
        assert abs(a - bb) / bb <= 1e-12, (name, t, a, bb)


def test_regularizer_jacobian_vs_jacfwd():
    C, M, nq = 4, 8, 64
    x = jnp.asarray(_random_coils(C, M, 2, True))
    cb = fr.make_basis("fourier", nq, M)
    cfg = HConfig(w_msc_final=2e-4)
    for t in (0.0, 0.4, 1.0):
        Ja = rs.regularizer_jacobian(x, t, cb, C, cfg, 11.6)
        Jf = jax.jacfwd(lambda v: rs.regularizer_residual(v, t, cb, C, cfg, 11.6))(x)
        assert float(jnp.max(jnp.abs(Ja - Jf)) / jnp.max(jnp.abs(Jf))) <= 1e-12
        # t-derivative of the residual is finite at t = 1 (fading weights)
        _, dr = jax.jvp(lambda tt: rs.regularizer_residual(x, tt, cb, C, cfg, 11.6), (t,), (1.0,))
        assert bool(jnp.all(jnp.isfinite(dr)))
    cfg0 = HConfig(w_msc_final=0.0)
    _, dr = jax.jvp(lambda tt: rs.regularizer_residual(x, tt, cb, C, cfg0, 11.6), (1.0,), (1.0,))
    assert bool(jnp.all(jnp.isfinite(dr)))


@pytest.mark.parametrize("name,P,M,grid", [("landreman_paul_qa", 32, 8, (32, 64)),
                                           ("w7x", 64, 12, (32, 64))])
def test_peak_memory_production(name, P, M, grid):
    """Compiled temp memory of the batched analytic Jacobian at production P (sec. 12.3)."""
    b = bd.load(name)
    C, nq = 4, 128
    cb = fr.make_basis("fourier", nq, M); S = sy.build(C, b.nfp, True)
    X, n, w = sp.evaluate(sp.build(b, *grid, "fundamental"), 1.0)
    cq = rs.surface_coeffs(w, 1.0)
    f = jax.jit(jax.vmap(lambda v: jac.residual_and_jacobian(v, cb, S, X, n, cq, 1e6)))
    comp = f.lower(jnp.zeros((P, C * 3 * (2 * M + 1)))).compile()
    ma = comp.memory_analysis()
    gb = (ma.temp_size_in_bytes + ma.output_size_in_bytes + ma.argument_size_in_bytes) / 1e9
    print(f"{name} P={P}: compiled peak (temp+out+args) = {gb:.2f} GB")
    assert gb < 20.0


def test_msc_final_ramp():
    """w_msc(t) = w_msc_final t^2 + w_msc0 f_ref (1 - t)^2 (msc_final_ramp = "t2"): zero at t = 0,
    the doc weight at t = 1, finite t-derivatives everywhere; "none" keeps the doc form."""
    from xcoil.homotopy import schedules as sch
    for w0 in (0.0, 1e-2):
        cfg = HConfig(w_msc_final=2e-4, w_msc0=w0)
        w = lambda t: sch.amp_msc(t, cfg) ** 2
        assert abs(float(w(1.0)) - 2e-4) <= 1e-18
        assert abs(float(w(0.0)) - w0 * cfg.f_ref) <= 1e-18
        assert abs(float(w(0.5)) - (2e-4 * 0.25 + w0 * cfg.f_ref * 0.25)) <= 1e-15
        for t in (0.0, 0.3, 1.0):
            assert bool(jnp.isfinite(jax.jvp(lambda tt: sch.amp_msc(tt, cfg), (t,), (1.0,))[1]))
    doc = HConfig(w_msc_final=2e-4, w_msc0=0.0, msc_final_ramp="none")
    assert abs(float(sch.amp_msc(0.0, doc)) ** 2 - 2e-4) <= 1e-18
