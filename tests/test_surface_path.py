"""Target-surface path.

t = 1 equals the input (1e-14, relative to R00); t = 0 surface independent of phi (1e-14);
Fourier and grid axisymmetrization agree (1e-12); normals unit, outward, equal to the FD cross
product (1e-10); t-derivatives vs central FD h = 1e-6 (1e-8); stellarator symmetry (1e-14);
mode-ordered path endpoints equal the linear path's and ramps are monotone.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401  (x64)
from xcoil.homotopy import boundary as bd
from xcoil.homotopy import surface_path as sp

DEVICES = ["landreman_paul_qa", "w7x", "qh_reactor", "rotating_ellipse", "cfqs_2b40"]


def _scaled(b, t, path="linear"):
    """Boundary whose shaped modes are scaled by their ramp value at t (reference for FD)."""
    base, groups = sp._mode_groups(b, path)
    scale = np.ones_like(b.rc)
    if path == "linear":
        scale[groups[0]] = t
    else:
        G = len(groups); end = 1.0 - sp.MODE_STRIDE * np.arange(G - 1, -1, -1); end[-1] = 1.0
        s = np.clip((t - (end - sp.MODE_WIDTH)) / sp.MODE_WIDTH, 0, 1)
        for g, idx in enumerate(groups):
            scale[idx] = s[g]
    return bd.from_modes(b.xm, b.xn, b.rc * scale, b.zs * scale, b.nfp, b.stellsym,
                         b.rs * scale, b.zc * scale)


@pytest.mark.parametrize("name", DEVICES)
def test_surface_endpoints(name):
    b = bd.load(name)
    for kind, (nphi, nth) in (("fundamental", (16, 32)), ("full", (40, 24))):
        p = sp.build(b, nphi, nth, kind)
        TH, PH, _ = sp.grid_angles(b.nfp, nphi, nth, kind, b.stellsym)
        err1 = float(jnp.max(jnp.abs(sp.eval_xyz(p, 1.0) - b.eval_xyz(TH, PH)))) / b.R00
        X0 = np.asarray(sp.eval_xyz(p, 0.0)).reshape(nphi, nth, 3)
        R0 = np.hypot(X0[..., 0], X0[..., 1]); Z0 = X0[..., 2]
        err0 = max(np.abs(R0 - R0.mean(0)).max(), np.abs(Z0 - Z0.mean(0)).max()) / b.R00
        assert err1 <= 1e-14, (name, kind, err1)
        assert err0 <= 1e-14, (name, kind, err0)


@pytest.mark.parametrize("name", DEVICES)
def test_axisym_fourier_vs_grid(name):
    b = bd.load(name)
    nth = 64; nph = 2 * int(np.abs(b.xn).max()) + 3
    th = 2 * np.pi * np.arange(nth) / nth
    ph = 2 * np.pi * np.arange(nph) / (b.nfp * nph)
    R, Z = b.eval_rz(th[:, None], ph[None, :])
    Rg, Zg = sp.axisymmetrize_grid(R, Z)
    Rf, Zf = sp.axisymmetric_cross_section(b, th)
    assert max(np.abs(Rg - Rf).max(), np.abs(Zg - Zf).max()) / b.R00 <= 1e-12
    # grid form -> Fourier (exact DFT) -> n = 0 truncation gives the same cross-section
    bg = bd.fit_grid(R, Z, b.nfp)
    Rh, Zh = sp.axisymmetric_cross_section(bg, th)
    assert max(np.abs(Rh - Rf).max(), np.abs(Zh - Zf).max()) / b.R00 <= 1e-12


def _inside_polygon(r, z, R, Z):
    """Even-odd point-in-polygon for many points against one closed polygon (R, Z)."""
    inside = np.zeros_like(r, bool)
    R2, Z2 = np.roll(R, -1), np.roll(Z, -1)
    for a, c, b_, d in zip(R, Z, R2, Z2):
        cond = ((c > z) != (d > z)) & (r < (b_ - a) * (z - c) / (d - c + 1e-300) + a)
        inside ^= cond
    return inside


@pytest.mark.parametrize("name", DEVICES)
@pytest.mark.parametrize("t", [0.0, 0.43, 1.0])
def test_surface_normals(name, t):
    b = bd.load(name)
    nphi, nth = 12, 20
    p = sp.build(b, nphi, nth, "fundamental")
    x, n, w = [np.asarray(a) for a in sp.evaluate(p, t)]
    assert np.abs(np.linalg.norm(n, axis=1) - 1).max() <= 1e-14
    TH, PH, cell = sp.grid_angles(b.nfp, nphi, nth, "fundamental", b.stellsym)
    bs_ = _scaled(b, t); h = 3e-4          # 4th-order FD, h at the roundoff/truncation optimum
    d4 = lambda f: (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)   # 4th-order FD
    Xt = d4(lambda e: bs_.eval_xyz(TH + e, PH))
    Xp = d4(lambda e: bs_.eval_xyz(TH, PH + e))
    N = p.sigma * np.cross(Xp, Xt)
    nfd = N / np.linalg.norm(N, axis=1)[:, None]
    assert np.abs(nfd - n).max() <= 1e-10
    # stricter extra check: exact (independent) derivative of the scaled Fourier series
    a = bs_.xm * TH[:, None] - bs_.nfp * bs_.xn * PH[:, None]; c_, s_ = np.cos(a), np.sin(a)
    R = c_ @ bs_.rc + s_ @ bs_.rs
    Rt = (-s_ * bs_.xm) @ bs_.rc + (c_ * bs_.xm) @ bs_.rs; Zt = (c_ * bs_.xm) @ bs_.zs - (s_ * bs_.xm) @ bs_.zc
    k = bs_.nfp * bs_.xn
    Rp = (s_ * k) @ bs_.rc - (c_ * k) @ bs_.rs; Zp = (-c_ * k) @ bs_.zs + (s_ * k) @ bs_.zc
    cp, sp_ = np.cos(PH), np.sin(PH)
    Ne = p.sigma * np.cross(np.stack([Rp * cp - R * sp_, Rp * sp_ + R * cp, Zp], -1),
                            np.stack([Rt * cp, Rt * sp_, Zt], -1))
    assert np.abs(Ne / np.linalg.norm(Ne, axis=1)[:, None] - n).max() <= 1e-14
    assert np.abs(np.linalg.norm(N, axis=1) * cell - w).max() / w.max() <= 1e-10
    # outward: a point displaced along +n leaves the cross-section at its own phi
    delta = 1e-4 * b.R00
    q = x + delta * n
    phq = np.arctan2(q[:, 1], q[:, 0]); rq = np.hypot(q[:, 0], q[:, 1]); zq = q[:, 2]
    thd = 2 * np.pi * np.arange(2048) / 2048
    outside = []
    for k in range(q.shape[0]):
        Rc, Zc = bs_.eval_rz(thd, np.full_like(thd, phq[k]))
        outside.append(not _inside_polygon(np.array([rq[k]]), np.array([zq[k]]), Rc, Zc)[0])
    assert all(outside), (name, t, int(np.sum(~np.asarray(outside))))


@pytest.mark.parametrize("name", DEVICES)
@pytest.mark.parametrize("path", ["linear", "mode_ordered"])
def test_surface_dt(name, path):
    b = bd.load(name)
    p = sp.build(b, 10, 16, "fundamental", path)
    h = 1e-6
    for t in (0.1, 0.5, 0.73):
        (x, n, w), (dx, dn, dw) = sp.evaluate_with_dt(p, t)
        xp, np_, wp = sp.evaluate(p, t + h); xm, nm, wm = sp.evaluate(p, t - h)
        # central FD h = 1e-6, error measured in the units of the evaluated quantity
        # (an h = 1e-6 central difference resolves ~eps/h ~ 1e-10 of the quantity's scale)
        for a, fd, f in ((dx, (xp - xm) / (2 * h), x), (dn, (np_ - nm) / (2 * h), n),
                         (dw, (wp - wm) / (2 * h), w)):
            assert float(jnp.max(jnp.abs(a - fd))) / float(jnp.max(jnp.abs(f))) <= 1e-8, (name, path, t)
        # stricter extra check, relative to the derivative itself: 4th-order FD, h = 1e-3
        H = 1e-3
        ev = lambda tt: sp.evaluate(p, tt)
        f2, f1, g1, g2 = ev(t + 2 * H), ev(t + H), ev(t - H), ev(t - 2 * H)
        for k, a in enumerate((dx, dn, dw)):
            fd4 = (-f2[k] + 8 * f1[k] - 8 * g1[k] + g2[k]) / (12 * H)
            # floor = what a 4th-order FD at H = 1e-3 can resolve (~1e-13 of the quantity)
            sc = float(jnp.max(jnp.abs(fd4))) + 1e-4 * float(jnp.max(jnp.abs(f1[k])))
            assert float(jnp.max(jnp.abs(a - fd4))) / sc <= 1e-8, (name, path, t, k)
        # exact: x is linear in the ramps, so dx/dt = sum_g ds_g/dt DeltaX_g
        sdot = np.where((t > np.asarray(p.tau)) & (t < np.asarray(p.end)),
                        1.0 / (np.asarray(p.end) - np.asarray(p.tau)), 0.0)
        dx_exact = np.einsum("g,gpd->pd", sdot, np.asarray(p.dX))
        assert np.abs(np.asarray(dx) - dx_exact).max() <= 1e-14 * np.abs(np.asarray(x)).max()


@pytest.mark.parametrize("name", DEVICES)
def test_stellarator_symmetry(name):
    b = bd.load(name)
    nphi, nth = 24, 18
    p = sp.build(b, nphi, nth, "full")
    for t in (0.0, 0.37, 1.0):
        x, n, w = [np.asarray(a).reshape(nphi, nth, -1) for a in sp.evaluate(p, t)]
        j = np.arange(nphi); i = np.arange(nth)
        xr = x[(nphi - 1 - j)][:, (-i) % nth]
        flip = np.array([1.0, -1.0, -1.0])
        assert np.abs(x - xr * flip).max() / b.R00 <= 1e-14
        wr = w[(nphi - 1 - j)][:, (-i) % nth]
        assert np.abs(w - wr).max() / w.max() <= 1e-13


@pytest.mark.parametrize("name", DEVICES)
def test_mode_ordered_path(name):
    b = bd.load(name)
    pl = sp.build(b, 8, 12, "fundamental", "linear")
    pm = sp.build(b, 8, 12, "fundamental", "mode_ordered")
    for t in (0.0, 1.0):
        d = float(jnp.max(jnp.abs(sp.eval_xyz(pl, t) - sp.eval_xyz(pm, t)))) / b.R00
        assert d <= 1e-14, (name, t, d)
    ts = np.linspace(0, 1, 401)
    S = np.stack([np.asarray(sp.ramps(pm, t)) for t in ts])
    assert S.shape[1] == sp.MODE_GROUPS
    assert np.all(np.diff(S, axis=0) >= 0) and S.min() >= 0 and S.max() <= 1
    assert np.allclose(S[0], 0) and np.allclose(S[-1], 1)
    # low (|n|, m) modes finish first
    assert np.all(S[-1] == 1.0)                           # endpoint exact
    finish = [ts[np.argmax(S[:, g] >= 1.0)] for g in range(S.shape[1])]
    assert all(np.diff(finish) > 0), finish
