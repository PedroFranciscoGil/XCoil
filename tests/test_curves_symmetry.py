"""Curves and symmetry.

Basis derivatives vs FD (1e-9); Fourier curves == simsopt CurveXYZFourier (identical DOFs);
expansion vs simsopt coils_via_symmetries: coordinates, tangents and current signs (1e-14);
least-squares fit exact for representable curves (1e-13).
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym


@pytest.mark.parametrize("kind,order", [("fourier", 8), ("fourier", 12), ("bspline", 25)])
def test_curve_derivs(kind, order):
    s = np.random.default_rng(0).uniform(0, 1, 64)
    B0, B1, B2 = fr.scalar_basis_at(kind, s, order)
    h = 1e-4
    d4 = lambda f: (-f(s + 2 * h) + 8 * f(s + h) - 8 * f(s - h) + f(s - 2 * h)) / (12 * h)
    F0 = lambda ss: fr.scalar_basis_at(kind, ss, order)[0]
    F1 = lambda ss: fr.scalar_basis_at(kind, ss, order)[1]
    fd1, fd2 = d4(F0), d4(F1)
    assert np.abs(fd1 - B1).max() / np.abs(B1).max() <= 1e-9
    assert np.abs(fd2 - B2).max() / np.abs(B2).max() <= 1e-9


@pytest.mark.parametrize("M,nq", [(8, 128), (12, 128), (5, 97)])
def test_fourier_matches_simsopt(M, nq):
    x = np.random.default_rng(1).normal(size=3 * (2 * M + 1)) * np.repeat(
        1.0 / (1 + np.arange(2 * M + 1)), 1)[np.tile(np.arange(2 * M + 1), 3)]
    cb = fr.make_basis("fourier", nq, M)
    g, gd, gdd = [np.asarray(a) for a in fr.coil_curve(jnp.asarray(x), cb)]
    c = fr.to_simsopt(x, nq, M)
    assert np.abs(g - c.gamma()).max() <= 1e-13 * max(1, np.abs(g).max())
    assert np.abs(gd - c.gammadash()).max() <= 1e-13 * np.abs(gd).max()
    assert np.abs(gdd - c.gammadashdash()).max() <= 1e-13 * np.abs(gdd).max()


@pytest.mark.parametrize("nfp,stellsym", [(2, True), (4, True), (3, False), (1, True)])
def test_symmetry_vs_simsopt(nfp, stellsym):
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import CurveXYZFourier
    C, M, nq = 4, 6, 64
    rng = np.random.default_rng(2)
    base, xs = [], []
    for i in range(C):
        c = CurveXYZFourier(nq, M)
        x = rng.normal(size=c.x.size) * 0.1
        x[0] += 1.0 + 0.1 * i; x[2 * M + 1 + 1] += 0.3          # offset + a loop
        c.x = x; base.append(c); xs.append(x)
    coils = coils_via_symmetries(base, [Current(1.7e5) for _ in range(C)], nfp, stellsym)
    S = sym.build(C, nfp, stellsym)
    assert S.n_phys == len(coils) == C * nfp * (2 if stellsym else 1)
    cb = fr.make_basis("fourier", nq, M)
    g, gd, _ = fr.coils_curves(jnp.asarray(np.concatenate(xs)), cb, C)
    G, Gd = np.asarray(sym.expand(g, S)), np.asarray(sym.expand(gd, S))
    for j, co in enumerate(coils):
        assert np.abs(G[j] - co.curve.gamma()).max() <= 1e-14 * max(1, np.abs(G[j]).max())
        assert np.abs(Gd[j] - co.curve.gammadash()).max() <= 1e-14 * np.abs(Gd[j]).max()
        assert float(S.sign[j]) * 1.7e5 == pytest.approx(co.current.get_value(), rel=0, abs=0)
        assert int(S.coil[j]) == j % C


@pytest.mark.parametrize("kind,order", [("fourier", 8), ("fourier", 12), ("bspline", 20)])
def test_ls_fit_exact(kind, order):
    nq = 128
    rng = np.random.default_rng(3)
    K = 2 * order + 1 if kind == "fourier" else order
    x = rng.normal(size=3 * K)
    cb = fr.make_basis(kind, nq, order)
    g = np.asarray(fr.coil_curve(jnp.asarray(x), cb)[0])
    xf = fr.fit(g, fr.uniform_s(nq), kind, order)
    assert np.abs(xf - x).max() / np.abs(x).max() <= 1e-13
    # also from a non-uniform, over-sampled parameterization
    s = np.sort(rng.uniform(0, 1, 300))
    gs = fr.scalar_basis_at(kind, s, order)[0] @ x.reshape(3, K).T
    xf2 = fr.fit(gs, s, kind, order)
    assert np.abs(xf2 - x).max() / np.abs(x).max() <= 1e-13


def test_base_toroidal_angles():
    a = sym.base_toroidal_angles(4, 2, True)
    assert np.allclose(a, (np.arange(4) + 0.5) * np.pi / 8)
    # 2C coils per period evenly spaced after the stellarator flip
    S = sym.build(4, 2, True)
    phis = []
    for j in range(S.n_phys):
        v = np.asarray(S.R[j]) @ np.array([np.cos(a[j % 4]), np.sin(a[j % 4]), 0.0])
        phis.append(np.mod(np.arctan2(v[1], v[0]), 2 * np.pi))
    d = np.diff(np.sort(phis))
    assert np.allclose(d, 2 * np.pi / 16)
