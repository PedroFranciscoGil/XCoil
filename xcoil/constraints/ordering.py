"""Toroidal ordering ("slot") constraint: neighbouring coils must not cross in the (phi, theta) projection.

Angles of a coil point g = (x, y, z): phi its cylindrical angle about the z-axis (continuous across the half period,
branch cut opposite its middle), theta its poloidal angle about a REFERENCE AXIS (R_ax(phi), Z_ax(phi)) at the point's
own phi, theta = atan2(z - Z_ax(phi), R - R_ax(phi)). The reference axis is a stellarator-symmetric curve

    R_ax(phi) = sum_n a_n cos(n nfp phi),     Z_ax(phi) = sum_n b_n sin(n nfp phi),     n = 0 .. nmax,

given explicitly (`axis=(a, b)`), so the rows depend on the coils only. The usual choice is the geometric axis of a
boundary, its m = 0 Fourier modes (`axis_from_boundary` / `axis_from_modes`; equal to the poloidal mean of R and Z
at each phi); `axis_from_points` gives the same from a full-torus grid of boundary points.

Rows (C + 1) * n_seg, <= 0 feasible: for every pair (left L, right R) of neighbouring base coils, plus the image of
coil 0 across phi = 0 | coil 0 and coil C-1 | its image across phi = pi / nfp, and every point k of the left coil,

    E_k = softmin_k' [phi_R,k' - phi_L,k + kappa (1 - cos(theta_L,k - theta_R,k'))]     (<= the hard min: conservative)
    row = (gap - E_k) / (pi / (nfp C)),

soft-maxed over n_seg contiguous segments of the left coil. kappa [rad] penalizes point pairs far apart in theta, so
only the right coil's points near theta_L,k bound the gap.
"""
from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from xcoil.curves import fourier as fr

BETA_ROW = 50.0          # soft-max sharpness over a segment's points (normalized row units)
NMAX = 8                 # axis harmonics kept by default


def axis_from_modes(xm, xn, rc, zs, nfp=None, nmax=NMAX):
    """(a, b), each (nmax + 1,), from boundary modes R = sum rc cos(m theta - n nfp phi), Z = sum zs sin(m theta -
    n nfp phi) (n in units of nfp, the XCoil / simsopt convention): a_n = rc(0, n), b_n = -zs(0, n) (n >= 0; a mode
    (0, -n) counts as (0, n) with zs's sign flipped). Exactly the poloidal mean of R and Z at each phi."""
    a, b = np.zeros(nmax + 1), np.zeros(nmax + 1)
    for m, n, r, z in zip(np.asarray(xm).ravel(), np.asarray(xn).ravel(), np.asarray(rc).ravel(),
                          np.asarray(zs).ravel()):
        if int(m) != 0 or abs(int(n)) > nmax:
            continue
        k = abs(int(n))
        a[k] += float(r)
        b[k] += -float(z) * (1.0 if int(n) >= 0 else -1.0)
    return a, b


def axis_from_boundary(b, nmax=NMAX):
    """Geometric axis (a, b) of an XCoil FourierBoundary (its m = 0 modes)."""
    return axis_from_modes(b.xm, b.xn, b.rc, b.zs, b.nfp, nmax)


def axis_from_points(S, nphi, ntheta, nfp, nmax=NMAX):
    """(a, b) from a full-torus grid of boundary points S (nphi * ntheta, 3), phi-major, phi_j = (j + 1/2) 2 pi /
    nphi: the theta-mean of R and Z per phi column, as the Fourier series above (JAX; exact when nphi > 2 nmax nfp)."""
    P = S.reshape(nphi, ntheta, 3)
    R = jnp.mean(jnp.sqrt(P[..., 0] ** 2 + P[..., 1] ** 2), 1)
    Z = jnp.mean(P[..., 2], 1)
    ph = (jnp.arange(nphi) + 0.5) * 2 * np.pi / nphi
    nn = jnp.arange(nmax + 1)[:, None] * nfp * ph[None, :]
    a = (2.0 / nphi) * (jnp.cos(nn) @ R)
    return a.at[0].multiply(0.5), (2.0 / nphi) * (jnp.sin(nn) @ Z)


def angles(g, a, b, nfp):
    """phi (continuous across the half period) and theta about the axis (a, b) at the point's own phi, g (..., 3)."""
    pm = 0.5 * np.pi / nfp
    x, y = g[..., 0], g[..., 1]
    phi = pm + jnp.arctan2(-x * np.sin(pm) + y * np.cos(pm), x * np.cos(pm) + y * np.sin(pm))
    nn = jnp.arange(a.shape[0]) * nfp * phi[..., None]
    th = jnp.arctan2(g[..., 2] - jnp.sin(nn) @ b, jnp.sqrt(x * x + y * y) - jnp.cos(nn) @ a)
    return phi, th


def _pairs(phi, th, nfp):
    C = phi.shape[0]
    pairs = [(phi[c], th[c], phi[c + 1], th[c + 1]) for c in range(C - 1)]
    pairs.append((-phi[0], -th[0], phi[0], th[0]))
    pairs.append((phi[C - 1], th[C - 1], 2 * np.pi / nfp - phi[C - 1], -th[C - 1]))
    return pairs


def pair_names(C):
    """Names of the C + 1 pairs, in row order."""
    return [f"{c}|{c + 1}" for c in range(C - 1)] + ["plane0|0", f"{C - 1}|plane"]


def n_rows(C, n_seg):
    return (C + 1) * int(n_seg)


def rows(g, a, b, nfp, gap, kappa, beta, n_seg):
    """Normalized ordering rows (<= 0 feasible), (C + 1) * n_seg, from base-coil points g (C, nq, 3); gap [rad]."""
    C, nq = g.shape[0], g.shape[1]
    phi, th = angles(g, jnp.asarray(a), jnp.asarray(b), nfp)
    E = []
    for phL, thL, phR, thR in _pairs(phi, th, nfp):
        v = phR[None, :] - phL[:, None] + kappa * (1.0 - jnp.cos(thL[:, None] - thR[None, :]))
        E.append(-logsumexp(-beta * v, axis=1) / beta)
    r = (gap - jnp.stack(E)) / (np.pi / (nfp * C))
    r = r.reshape(C + 1, n_seg, nq // n_seg)
    return (logsumexp(BETA_ROW * r, axis=-1) / BETA_ROW).ravel()


def report(g, a, b, nfp, kappa, window=0.03):
    """Hard ordering metrics per pair [deg]: the smallest phi_R - phi_L over point pairs with |theta_L - theta_R| <
    window (negative: the coils cross in the projection), the hard theta-penalized gap min [phi_R - phi_L +
    kappa (1 - cos dtheta)] that the rows bound from below, and each coil's (min, max) phi."""
    phi, th = angles(g, jnp.asarray(a), jnp.asarray(b), nfp)
    win, pen = [], []
    for phL, thL, phR, thR in _pairs(phi, th, nfp):
        D = phR[None, :] - phL[:, None]
        cth = 1.0 - jnp.cos(thL[:, None] - thR[None, :])
        win.append(jnp.min(jnp.where(cth < 1.0 - np.cos(window), D, jnp.inf)))
        pen.append(jnp.min(D + kappa * cth))
    ext = jnp.stack([jnp.min(phi, 1), jnp.max(phi, 1)], 1)
    return jnp.degrees(jnp.stack(win)), jnp.degrees(jnp.stack(pen)), jnp.degrees(ext)


def ordering_extra(prob, axis=None, gap_deg=1.0, kappa=20.0, beta=500.0, n_seg=16):
    """Extra constraint rows for fixed.device_lm.build_solver(prob, extra=...): the ordering rows on the problem's
    coil quadrature about a FIXED reference axis (default: the geometric axis of prob's boundary). n_seg must divide
    the quadrature."""
    T = prob.T
    C, nq, nfp = T.sym.C, T.cb.B0.shape[0], int(T.sym.nfp)
    if nq % int(n_seg):
        raise ValueError(f"n_seg {n_seg} must divide the coil quadrature {nq}")
    a, b = axis if axis is not None else axis_from_boundary(prob.b)
    a, b = jnp.asarray(a, jnp.float64), jnp.asarray(b, jnp.float64)
    cb, gap = T.cb, math.radians(float(gap_deg))
    g = lambda x: rows(fr.coils_curves(x, cb, C)[0], a, b, nfp, gap, float(kappa), float(beta), int(n_seg))
    from xcoil.fixed.device_lm import Extra
    return Extra(lambda x: jnp.zeros((0,), x.dtype), g, jr=lambda x: jnp.zeros((0, x.shape[0]), x.dtype),
                 jg=jax.jacfwd(g))


def certify_ordering(x, prob, axis=None, kappa=20.0, gap_deg=1.0, nq=768, window=0.03):
    """Ordering report on an nq-point quadrature about the fixed axis (default: prob's boundary axis): per pair the
    windowed gap and the penalized gap [deg]; ok when no pair crosses (windowed gap > 0) and every penalized gap
    reaches gap_deg."""
    T = prob.T
    C, nfp = T.sym.C, int(T.sym.nfp)
    a, b = axis if axis is not None else axis_from_boundary(prob.b)
    from xcoil.constraints.forces import basis_like
    g = fr.coils_curves(jnp.asarray(x, jnp.float64), basis_like(T.cb, nq), C)[0]
    win, pen, ext = (np.asarray(v) for v in report(g, a, b, nfp, float(kappa), window))
    return dict(pairs=pair_names(C), gap_window_deg=win.tolist(), gap_penalized_deg=pen.tolist(),
                phi_extent_deg=ext.tolist(), axis=[np.asarray(a).tolist(), np.asarray(b).tolist()],
                ok=bool(np.all(win > 0.0) and np.all(pen >= float(gap_deg) * (1 - 1e-6))))
