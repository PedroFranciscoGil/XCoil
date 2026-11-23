"""Planar seeds from the axisymmetrized target.

Untimed geometry tables: the planar seed curve per base coil is chosen here (offset -> ellipse
blend -> circle fallback, first feasible wins). Timed assembly: x = Phi^+ P (one matrix product
per coil) from the stored seed points P and the precomputed pseudo-inverse.

1. toroidal placement phi_c = (c + 1/2) pi/(nfp C) (stellsym) or (c + 1/2) 2 pi/(nfp C);
2. offset (R_0(theta), Z_0(theta)) outward in-plane by d_0 = (1 + beta_0) d_cs + delta;
3. if curvature > 0.9 kappa_lim or non-convex: blend toward the best-fit (moment) ellipse,
   smallest a in {0, .25, .5, .75, 1};
4. feasibility of the FITTED curves at t = 0: every constraint with margin (g <= 0);
5. fallback: circles about (R_0 average, 0), radius from max(dist to section) + d_0 up in 5% steps;
6. LS fit on an arclength-uniform parameterization;
7. no feasible seed within 10 radius steps -> SEED_INFEASIBLE (do not proceed).
Seeds depend only on the axisymmetrized target and the limits (no-hidden-warm-start rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp

from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod
from xcoil.constraints import pointwise as pw
from . import boundary as bd
from . import surface_path as sp

BETA0 = 0.1
DELTA_FRAC = 0.25
BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)
CIRCLE_STEPS = 10
CIRCLE_GROWTH = 0.05


@dataclass
class SeedResult:
    status: str                    # "OK" or "SEED_INFEASIBLE"
    kind: str                      # "offset", "blend a=..", "circle r=.."
    x: np.ndarray | None           # (C*3K,) fitted DOFs
    points: np.ndarray | None      # (C, n_fit, 3) seed points (timed assembly input)
    pinv: np.ndarray | None        # (K, n_fit) scalar pseudo-inverse
    current_sign: float = 1.0
    L_seed: float = float("nan")
    gmax: float = float("nan")     # max constraint value at t = 0 (<= 0 feasible)
    log: list = field(default_factory=list)


def _section(b, n=4096):
    th = 2 * np.pi * np.arange(n) / n
    sel = b.xn == 0
    m = b.xm[sel]; a = m * th[:, None]; c, s = np.cos(a), np.sin(a)
    R = c @ b.rc[sel] + s @ b.rs[sel]; Z = s @ b.zs[sel] + c @ b.zc[sel]
    Rt = (-s * m) @ b.rc[sel] + (c * m) @ b.rs[sel]; Zt = (c * m) @ b.zs[sel] - (s * m) @ b.zc[sel]
    Rtt = (-c * m * m) @ b.rc[sel] - (s * m * m) @ b.rs[sel]
    Ztt = (-s * m * m) @ b.zs[sel] - (c * m * m) @ b.zc[sel]
    return th, R, Z, Rt, Zt, Rtt, Ztt


def _planar_curvature(R, Z):
    """Signed curvature of a closed polyline sampled densely (spectral derivatives)."""
    n = R.size; k = np.fft.fftfreq(n, 1.0 / n)
    d = lambda f, o: np.real(np.fft.ifft((1j * k) ** o * np.fft.fft(f)))
    R1, Z1, R2, Z2 = d(R, 1), d(Z, 1), d(R, 2), d(Z, 2)
    return (R1 * Z2 - Z1 * R2) / (R1 ** 2 + Z1 ** 2) ** 1.5


def _offset(b, d0, sigma):
    th, R, Z, Rt, Zt, *_ = _section(b)
    nrm = np.hypot(Rt, Zt)
    return R + d0 * sigma * Zt / nrm, Z - d0 * sigma * Rt / nrm


def _ellipse_blend(R, Z, a):
    if a == 0.0:
        return R, Z
    c = np.array([R.mean(), Z.mean()])
    P = np.stack([R, Z], 1) - c
    lam, V = np.linalg.eigh(np.cov(P.T, bias=True))
    ea, eb = np.sqrt(2 * lam[1]), np.sqrt(2 * lam[0]); e1, e2 = V[:, 1], V[:, 0]
    psi = np.arctan2(P @ e2, P @ e1)
    re = ea * eb / np.sqrt((eb * np.cos(psi)) ** 2 + (ea * np.sin(psi)) ** 2)
    E = c + re[:, None] * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
    return (1 - a) * R + a * E[:, 0], (1 - a) * Z + a * E[:, 1]


def _resample_arclength(R, Z, n_fit):
    seg = np.hypot(np.diff(np.append(R, R[0])), np.diff(np.append(Z, Z[0])))
    s = np.concatenate([[0.0], np.cumsum(seg)]); s /= s[-1]
    u = np.arange(n_fit) / n_fit
    return np.interp(u, s, np.append(R, R[0])), np.interp(u, s, np.append(Z, Z[0])), u


def _to_3d(R, Z, phi):
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], -1)


def _evaluate(x, C, cb, S, lim, S_dist):
    g, gd, gdd = fr.coils_curves(jnp.asarray(x), cb, C)
    G = sym_mod.expand(g, S)
    gv = pw.constraint_values(g, gd, gdd, G, S_dist, S, lim, jnp.inf)
    L = float(jnp.sum(pw.lengths(gd)))
    nL = C if lim.per_coil_length else 1
    return np.asarray(gv)[nL:], L                    # drop the (uncapped) length row(s)


def build(b: bd.FourierBoundary, C: int, lim: pw.Limits, kind="fourier", order=8, nq=128,
          grid_dist=(64, 64), n_fit=None, stellsym=True, target_length=None) -> SeedResult:
    """Planar seed. target_length: after the first feasible seed
    is found, its planar section is scaled about (R_0, 0) so the total length reaches
    target_length; the scale steps back toward 1 until the grown seed is feasible."""
    base = _build(b, C, lim, kind, order, nq, grid_dist, n_fit, stellsym)
    if target_length is None or base.status != "OK" or base.L_seed >= target_length:
        return base
    lim = lim if lim.alpha_cc > 0 else pw.with_alpha(lim)
    n_fit = n_fit or 4 * nq
    cb = fr.make_basis(kind, nq, order)
    S = sym_mod.build(C, b.nfp, stellsym)
    S_dist = sp.eval_xyz(sp.build(b, grid_dist[0], grid_dist[1], "full"), 0.0)
    phis = sym_mod.base_toroidal_angles(C, b.nfp, stellsym)
    # planar section of base coil 0 in its own plane: (R, Z) about the axis
    P0 = base.points[0]
    R = np.hypot(P0[:, 0], P0[:, 1]); Z = P0[:, 2]
    log = list(base.log)
    s_target = target_length / base.L_seed
    for k in range(GROW_STEPS + 1):
        sc = s_target ** (1.0 - k / GROW_STEPS)
        Rg, Zg = b.R00 + sc * (R - b.R00), sc * Z
        Rr, Zr, _ = _resample_arclength(Rg, Zg, n_fit)
        pts = np.stack([_to_3d(Rr, Zr, ph) for ph in phis])
        x = np.concatenate([(base.pinv @ pts[c]).T.reshape(-1) for c in range(C)])
        gv, L = _evaluate(x, C, cb, S, lim, S_dist)
        gmax = float(np.max(gv))
        log.append(f"grown x{sc:.4f}: gmax={gmax:+.3e} L={L:.3f}")
        if gmax <= 0:
            return _finish(f"{base.kind} grown x{sc:.3f}", x, pts, base.pinv, L, gmax, log, b, C, cb, S)
    return base


def _build(b: bd.FourierBoundary, C: int, lim: pw.Limits, kind="fourier", order=8, nq=128,
           grid_dist=(64, 64), n_fit=None, stellsym=True) -> SeedResult:
    lim = lim if lim.alpha_cc > 0 else pw.with_alpha(lim)
    n_fit = n_fit or 4 * nq
    cb = fr.make_basis(kind, nq, order)
    S = sym_mod.build(C, b.nfp, stellsym)
    phis = sym_mod.base_toroidal_angles(C, b.nfp, stellsym)
    pd = sp.build(b, grid_dist[0], grid_dist[1], "full")
    S_dist = sp.eval_xyz(pd, 0.0)
    sigma = sp._cross_section_orientation(b)
    B0 = fr.scalar_basis_at(kind, np.arange(n_fit) / n_fit, order)[0]
    pinv = np.linalg.pinv(B0)                         # (K, n_fit)
    d0 = (1 + BETA0) * lim.dcs + DELTA_FRAC * lim.dcs
    log = []

    best = {}                                         # least-infeasible attempt (gmax smallest)

    def attempt(R, Z, label):
        Rr, Zr, _ = _resample_arclength(R, Z, n_fit)
        pts = np.stack([_to_3d(Rr, Zr, ph) for ph in phis])            # (C, n_fit, 3)
        x = np.concatenate([(pinv @ pts[c]).T.reshape(-1) for c in range(C)])
        gv, L = _evaluate(x, C, cb, S, lim, S_dist)
        gmax = float(np.max(gv))
        log.append(f"{label}: gmax={gmax:+.3e} L={L:.3f}")
        if not best or gmax < best["gmax"]:
            best.update(gmax=gmax, x=x, pts=pts, L=L, label=label)
        return gmax, x, pts, L

    Ro, Zo = _offset(b, d0, sigma)
    for a in BLENDS:
        R, Z = _ellipse_blend(Ro, Zo, a)
        k = _planar_curvature(R, Z) * sigma
        if not (np.abs(k).max() <= 0.9 * lim.kappa_lim and k.min() > 0):
            log.append(f"blend a={a}: planar kappa max {np.abs(k).max():.3f} convex={k.min() > 0}")
            continue
        gmax, x, pts, L = attempt(R, Z, "offset" if a == 0 else f"blend a={a}")
        if gmax <= 0:
            return _finish("offset" if a == 0 else f"blend a={a}", x, pts, pinv, L, gmax, log,
                           b, C, cb, S)
        break                                           # smoothest acceptable shape infeasible
    th, R0, Z0, *_ = _section(b)
    Rc = b.R00
    r = np.max(np.hypot(R0 - Rc, Z0)) + d0
    for step in range(CIRCLE_STEPS + 1):
        rr = r * (1 + CIRCLE_GROWTH) ** step
        gmax, x, pts, L = attempt(Rc + rr * np.cos(th), rr * np.sin(th), f"circle r={rr:.4f}")
        if gmax <= 0:
            return _finish(f"circle r={rr:.4f}", x, pts, pinv, L, gmax, log, b, C, cb, S)
    # no feasible planar seed: keep the least-infeasible one (it still fixes the problem's scales,
    # B_ref, I0 and L_ref; the status records that it is infeasible)
    res = _finish(best["label"], best["x"], best["pts"], pinv, best["L"], best["gmax"], log, b, C, cb, S)
    res.status = "SEED_INFEASIBLE"
    return res


GROW_STEPS = 12


def _finish(kind, x, pts, pinv, L, gmax, log, b, C, cb, S):
    """Current sign: positive toroidal field at the axis estimate (R_0 average, 0, 0)."""
    from xcoil.field import biot_savart as bs
    g, gd, _ = fr.coils_curves(jnp.asarray(x), cb, C)
    G, Gd = sym_mod.expand(g, S), sym_mod.expand(gd, S)
    B = bs.field(jnp.asarray([[b.R00, 0.0, 0.0]]), G, Gd, S.sign)
    sign = 1.0 if float(B[0, 1]) > 0 else -1.0          # e_phi = +y at phi = 0
    return SeedResult("OK", kind, x, pts, pinv, sign, L, gmax, log)
