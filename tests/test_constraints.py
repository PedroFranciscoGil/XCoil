"""Constraints.

All directional derivatives vs FD, relative error below 1e-7 (per block), plus exact agreement of
the analytic Jacobian with jax.jacfwd of the values; soft-min bounds hold on random point sets;
Gauss linking matches simsopt LinkingNumber on linked and unlinked pairs.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, surface_path as sp, seeds
from xcoil.homotopy.devices import DEVICES
from xcoil.constraints import pointwise as pw
from xcoil.curves import fourier as fr, symmetry as sy


class lk:
    """Reference Gauss linking integral (the certificate's linking_numbers computes the same on the device)."""

    @staticmethod
    def gauss_linking(ga, gda, gb, gdb):
        D = ga[:, None, :] - gb[None, :, :]
        r3 = np.linalg.norm(D, axis=-1) ** 3
        cr = np.cross(gda[:, None, :], gdb[None, :, :])
        return float(np.sum(np.sum(cr * D, -1) / r3) / (ga.shape[0] * gb.shape[0]) / (4 * np.pi))

    @staticmethod
    def linking_matrix(G, Gd):
        J = G.shape[0]
        L = np.zeros((J, J), int); raw = np.zeros((J, J))
        for a in range(J):
            for b in range(a + 1, J):
                v = lk.gauss_linking(G[a], Gd[a], G[b], Gd[b])
                raw[a, b] = raw[b, a] = v
                L[a, b] = L[b, a] = int(np.rint(v))
        return L, raw


def _setup(key, nq=64, pert=0.02, seed=0):
    d = DEVICES[key]; b = bd.load(d.name)
    lim = pw.with_alpha(pw.Limits(d.kappa_lim, d.msc_lim, d.dcc, d.dcs))
    r = seeds.build(b, d.n_coils, lim, "fourier", d.order, nq, (24, 24))
    x = jnp.asarray(r.x) + pert * b.R00 * jnp.asarray(
        np.random.default_rng(seed).normal(size=r.x.size)) / (1 + np.arange(r.x.size) % (2 * d.order + 1))
    cb = fr.make_basis("fourier", nq, d.order); S = sy.build(d.n_coils, b.nfp, True)
    Sd = sp.eval_xyz(sp.build(b, 24, 24, "full"), 0.6)
    return x, cb, S, Sd, lim, 1.2 * r.L_seed, d.n_coils


@pytest.mark.bigmem
@pytest.mark.parametrize("key", ["lpqa", "w7x", "qh_reactor"])
def test_constraint_grads(key):
    x, cb, S, Sd, lim, eps, C = _setup(key)
    vals = lambda v: pw.values_x(v, cb, S, Sd, lim, eps)
    A = pw.constraint_jacobian(x, cb, S, Sd, lim, eps)
    Af = jax.jacfwd(vals)(x)
    assert float(jnp.max(jnp.abs(A - Af)) / jnp.max(jnp.abs(Af))) <= 1e-12
    blocks = pw.blocks(C, cb.B0.shape[0])
    rng = np.random.default_rng(1)
    for trial in range(3):
        v = jnp.asarray(rng.normal(size=x.size)); v = v / jnp.linalg.norm(v)
        Av = A @ v
        for name, sl in blocks.items():
            if sl.stop == sl.start:                      # optional block switched off (e.g. arc)
                continue
            errs = []
            for h in (1e-4, 3e-5, 1e-5):
                fd = (-vals(x + 2 * h * v) + 8 * vals(x + h * v) - 8 * vals(x - h * v)
                      + vals(x - 2 * h * v)) / (12 * h)
                errs.append(float(jnp.max(jnp.abs(Av[sl] - fd[sl])) /
                                  max(float(jnp.max(jnp.abs(fd[sl]))), 1e-300)))
            print(f"{key} {name:5s} dir{trial}: rel err {min(errs):.2e}")
            assert min(errs) <= 1e-7, (key, name, errs)


def test_softmin_bounds():
    rng = np.random.default_rng(3)
    for trial in range(50):
        n = int(rng.integers(2, 4000)); alpha = float(10 ** rng.uniform(0, 4))
        d = jnp.asarray(rng.uniform(0.01, 3.0, n))
        s = float(pw.softmin(d, alpha)); m = float(jnp.min(d))
        assert s <= m + 1e-15 and m <= s + np.log(n) / alpha + 1e-15
    # masked entries are ignored exactly
    d = jnp.asarray([0.5, 0.0, 2.0]); mask = jnp.asarray([True, False, True])
    assert float(pw.softmin(d, 50.0, mask)) == pytest.approx(float(pw.softmin(d[jnp.array([0, 2])], 50.0)), abs=0)


def _circle(center, e1, e2, r, n=512):
    s = np.arange(n) / n; w = 2 * np.pi
    g = center + r * (np.cos(w * s)[:, None] * e1 + np.sin(w * s)[:, None] * e2)
    gd = r * w * (-np.sin(w * s)[:, None] * e1 + np.cos(w * s)[:, None] * e2)
    return g, gd


def test_linking():
    from simsopt.geo import CurveXYZFourier, LinkingNumber
    ex, ey, ez = np.eye(3)
    a, ad = _circle(np.zeros(3), ex, ey, 1.0)
    b, bd_ = _circle(np.array([1.0, 0, 0]), ex, ez, 1.0)          # Hopf link
    c, cd = _circle(np.array([5.0, 0, 0]), ex, ez, 1.0)           # separated
    assert int(np.rint(lk.gauss_linking(a, ad, b, bd_))) in (-1, 1)
    assert abs(lk.gauss_linking(a, ad, b, bd_)) == pytest.approx(1.0, abs=1e-8)
    assert int(np.rint(lk.gauss_linking(a, ad, c, cd))) == 0

    def to_curve(center, e1, e2, r):
        cu = CurveXYZFourier(512, 1); x = np.zeros(9)
        for d in range(3):
            x[3 * d + 0] = center[d]; x[3 * d + 1] = r * e2[d]; x[3 * d + 2] = r * e1[d]
        cu.x = x; return cu
    linked = [to_curve(np.zeros(3), ex, ey, 1.0), to_curve(np.array([1.0, 0, 0]), ex, ez, 1.0)]
    unlinked = [to_curve(np.zeros(3), ex, ey, 1.0), to_curve(np.array([5.0, 0, 0]), ex, ez, 1.0)]
    for curves, expect in ((linked, 1), (unlinked, 0)):
        L, raw = lk.linking_matrix(np.stack([c.gamma() for c in curves]),
                                   np.stack([c.gammadash() for c in curves]))
        assert abs(L[0, 1]) == expect
        assert int(round(abs(LinkingNumber(curves).J()))) == expect


@pytest.mark.bigmem
def test_segment_distance_rows():
    """dist_segments = n: coil-coil / coil-surface rows are soft-mins over n contiguous segments
    (identity: soft-min of the points' soft-mins); analytic Jacobian == jacfwd; each segment row is
    a lower bound of the true minimum distance over its points (conservative)."""
    import numpy as np
    from xcoil.homotopy import boundary as bd, seeds, problem_h as ph
    from xcoil.homotopy.config import for_device
    from xcoil.homotopy import surface_path as sp
    cfg = for_device("lpqa", dist_segments=8); b = bd.load(cfg.surface)
    lim = ph.limits_from(cfg)
    seed = seeds.build(b, 4, lim, cfg.basis, cfg.order, cfg.nq_opt, cfg.grid_dist)
    T = ph.build_tables(b, cfg, seed, grid=(16, 32))
    x = jnp.asarray(seed.x) + 1e-2 * jnp.asarray(np.random.default_rng(3).normal(size=seed.x.size))
    S = sp.eval_xyz(T.dist, 0.4)
    g = pw.values_x(x, T.cb, T.sym, S, lim, 14.0)
    assert g.shape[0] == pw.n_constraints(4, cfg.nq_opt, 8)
    A = pw.constraint_jacobian(x, T.cb, T.sym, S, lim, 14.0)
    Af = jax.jacfwd(lambda v: pw.values_x(v, T.cb, T.sym, S, lim, 14.0))(x)
    assert float(jnp.max(jnp.abs(A - Af)) / jnp.max(jnp.abs(Af))) <= 1e-12
    bl = pw.blocks(4, cfg.nq_opt, 8)
    from xcoil.curves import fourier as fr
    gc = fr.coils_curves(x, T.cb, 4)[0]
    dmin = jnp.min(jnp.linalg.norm(gc[:, :, None, :] - S[None, None], axis=-1), axis=-1).reshape(4, 8, -1).min(-1)
    dseg = (1 + lim.beta_d) * lim.dcs - g[bl["cs"]].reshape(4, 8) * lim.dcs
    assert bool(jnp.all(dseg <= dmin + 1e-12))
