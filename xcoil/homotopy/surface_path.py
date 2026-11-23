"""Target-surface homotopy path Sigma(t), t in [0, 1].

t = 0 is the axisymmetrized target (n = 0 modes only; equal to the phi-average of R and Z at
fixed theta), t = 1 is the real boundary. Every path is linear in a small ramp vector s(t):

    X(theta, phi; t) = X_axi + sum_g s_g(t) DeltaX_g          (same for X_theta, X_phi)

because Cartesian x = R cos(phi) is linear in R at fixed phi. All arrays are precomputed
(untimed); evaluation is one small contraction. Normals N = sigma (X_phi x X_theta), unit
normal n = N/|N|, area weight w = |N| dtheta dphi. d/dt comes from jax.jvp.

Grids (points ordered phi-major, theta fastest):
  fundamental : phi_j = (j + 1/2) Dphi, Dphi = pi/(nfp nphi) (stellsym) or 2 pi/(nfp nphi);
                the half-cell offset makes the grid map onto itself under the symmetry group,
                so the reduced flux sum equals the full-torus sum exactly (sec. 8.4).
  full        : phi_j = (j + 1/2) 2 pi / nphi over the whole torus.
theta_i = 2 pi i / ntheta in both cases.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from . import boundary as bd

MODE_GROUPS = 4          # mode-ordered path: 4 groups, stride 0.2, width 0.4 (50% overlap)
MODE_STRIDE = 0.2
MODE_WIDTH = 0.4


def grid_angles(nfp, nphi, ntheta, kind="fundamental", stellsym=True):
    if kind == "fundamental":
        dphi = (np.pi if stellsym else 2 * np.pi) / (nfp * nphi)
    elif kind == "full":
        dphi = 2 * np.pi / nphi
    else:
        raise ValueError(kind)
    phi = (np.arange(nphi) + 0.5) * dphi
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    PH, TH = np.meshgrid(phi, theta, indexing="ij")          # (nphi, ntheta)
    return TH.ravel(), PH.ravel(), (2 * np.pi / ntheta) * dphi


def _mode_groups(b: bd.FourierBoundary, path: str):
    """List of index arrays: group 0 = n = 0 (axisymmetric base), then the ramped groups."""
    base = np.where(b.xn == 0)[0]
    shaped = np.where(b.xn != 0)[0]
    if path == "linear":
        return base, [shaped]
    if path == "mode_ordered":
        order = shaped[np.lexsort((b.xm[shaped], np.abs(b.xn[shaped])))]   # by (|n|, m)
        return base, [g for g in np.array_split(order, MODE_GROUPS)]
    raise ValueError(path)


def _xyz_tables(b, idx, TH, PH):
    """Cartesian X, X_theta, X_phi contributed by the modes idx (linear in their coefficients).
    By default one jitted program on the default JAX device (FP64; the tables differ from numpy's at rounding
    level), much faster on fine full-torus grids; XCOIL_TABLES=numpy: the numpy sums."""
    if idx.size == 0:
        z = np.zeros((TH.size, 3)); return z, z.copy(), z.copy()
    if bd.tables_on_device():
        out = _xyz_tables_jax(b.xm[idx], b.xn[idx], b.rc[idx], b.rs[idx], b.zs[idx], b.zc[idx], float(b.nfp),
                              np.asarray(TH, float), np.asarray(PH, float))
        return tuple(np.asarray(o) for o in out)
    m = b.xm[idx]; n = b.xn[idx]
    a = m * TH[:, None] - b.nfp * n * PH[:, None]
    c, s = np.cos(a), np.sin(a)
    R = c @ b.rc[idx] + s @ b.rs[idx]
    Z = s @ b.zs[idx] + c @ b.zc[idx]
    R_t = (-s * m) @ b.rc[idx] + (c * m) @ b.rs[idx]
    Z_t = (c * m) @ b.zs[idx] + (-s * m) @ b.zc[idx]
    R_p = (s * (b.nfp * n)) @ b.rc[idx] + (-c * (b.nfp * n)) @ b.rs[idx]
    Z_p = (-c * (b.nfp * n)) @ b.zs[idx] + (s * (b.nfp * n)) @ b.zc[idx]
    cp, sp = np.cos(PH), np.sin(PH)
    X = np.stack([R * cp, R * sp, Z], -1)
    Xt = np.stack([R_t * cp, R_t * sp, Z_t], -1)
    Xp = np.stack([R_p * cp - R * sp, R_p * sp + R * cp, Z_p], -1)
    return X, Xt, Xp


@jax.jit
def _xyz_tables_jax(m, n, rc, rs, zs, zc, nfp, TH, PH):
    a = m * TH[:, None] - nfp * n * PH[:, None]
    c, s = jnp.cos(a), jnp.sin(a)
    R = c @ rc + s @ rs
    Z = s @ zs + c @ zc
    R_t = (-s * m) @ rc + (c * m) @ rs
    Z_t = (c * m) @ zs + (-s * m) @ zc
    R_p = (s * (nfp * n)) @ rc + (-c * (nfp * n)) @ rs
    Z_p = (-c * (nfp * n)) @ zs + (s * (nfp * n)) @ zc
    cp, sp_ = jnp.cos(PH), jnp.sin(PH)
    X = jnp.stack([R * cp, R * sp_, Z], -1)
    Xt = jnp.stack([R_t * cp, R_t * sp_, Z_t], -1)
    Xp = jnp.stack([R_p * cp - R * sp_, R_p * sp_ + R * cp, Z_p], -1)
    return X, Xt, Xp


def _cross_section_orientation(b):
    """+1 if theta runs counter-clockwise in the (R, Z) plane (N = X_phi x X_theta outward)."""
    th = 2 * np.pi * np.arange(512) / 512
    out = []
    for ph in (0.0, np.pi / b.nfp / 2):
        R, Z = b.eval_rz(th, np.full_like(th, ph))
        out.append(0.5 * np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z))
    if np.sign(out[0]) != np.sign(out[1]):
        raise ValueError("inconsistent cross-section orientation")
    return float(np.sign(out[0]))


@dataclass(frozen=True)
class PathTables:
    X0: jnp.ndarray       # (Np, 3) axisymmetric base
    Xt0: jnp.ndarray
    Xp0: jnp.ndarray
    dX: jnp.ndarray       # (G, Np, 3) ramped-group contributions
    dXt: jnp.ndarray
    dXp: jnp.ndarray
    tau: jnp.ndarray      # (G,) ramp start
    end: jnp.ndarray      # (G,) ramp end (s = 1 exactly for t >= end)
    linear: bool          # s(t) = t exactly (no clip) on the linear path
    sigma: float          # orientation sign
    cell: float           # dtheta dphi
    nphi: int
    ntheta: int


jax.tree_util.register_pytree_node(
    PathTables,
    lambda p: ((p.X0, p.Xt0, p.Xp0, p.dX, p.dXt, p.dXp, p.tau, p.end, p.sigma),
               (p.linear, p.cell, p.nphi, p.ntheta)),
    lambda aux, ch: PathTables(*ch[:8], aux[0], ch[8], *aux[1:]))   # sigma is data (per boundary), the rest structure


def build(b: bd.FourierBoundary, nphi, ntheta, kind="fundamental", path="linear") -> PathTables:
    TH, PH, cell = grid_angles(b.nfp, nphi, ntheta, kind, b.stellsym)
    base, groups = _mode_groups(b, path)
    X0, Xt0, Xp0 = _xyz_tables(b, base, TH, PH)
    tabs = [_xyz_tables(b, g, TH, PH) for g in groups]
    G = len(groups)
    if path == "linear":
        tau, end = np.zeros(1), np.ones(1)
    else:                     # ends 1 - (G-1-g) stride, width MODE_WIDTH; last group ends at 1
        end = 1.0 - MODE_STRIDE * np.arange(G - 1, -1, -1); tau = end - MODE_WIDTH
        end[-1] = 1.0
    return PathTables(jnp.asarray(X0), jnp.asarray(Xt0), jnp.asarray(Xp0),
                      jnp.asarray(np.stack([t[0] for t in tabs])),
                      jnp.asarray(np.stack([t[1] for t in tabs])),
                      jnp.asarray(np.stack([t[2] for t in tabs])),
                      jnp.asarray(tau), jnp.asarray(end), path == "linear",
                      _cross_section_orientation(b), float(cell), int(nphi), int(ntheta))


def ramps(p: PathTables, t):
    if p.linear:
        return jnp.reshape(t, (1,)) * jnp.ones_like(p.tau)
    return jnp.clip((t - p.tau) / (p.end - p.tau), 0.0, 1.0)


def eval_xyz(p: PathTables, t):
    s = ramps(p, t)
    return p.X0 + jnp.einsum("g,gpd->pd", s, p.dX)


def evaluate(p: PathTables, t):
    """Surface points x (Np,3), outward unit normals n (Np,3), area weights w (Np,)."""
    s = ramps(p, t)
    X = p.X0 + jnp.einsum("g,gpd->pd", s, p.dX)
    Xt = p.Xt0 + jnp.einsum("g,gpd->pd", s, p.dXt)
    Xp = p.Xp0 + jnp.einsum("g,gpd->pd", s, p.dXp)
    N = p.sigma * jnp.cross(Xp, Xt)
    nn = jnp.linalg.norm(N, axis=-1)
    return X, N / nn[:, None], nn * p.cell


def evaluate_with_dt(p: PathTables, t):
    """(x, n, w) and their t-derivatives via forward-mode AD (sec. 4.4)."""
    return jax.jvp(lambda tt: evaluate(p, tt), (jnp.asarray(t, float),), (jnp.ones((), float),))


def axisymmetric_cross_section(b: bd.FourierBoundary, theta):
    """R_0(theta), Z_0(theta) of the t = 0 surface (n = 0 modes)."""
    sel = b.xn == 0
    a = b.xm[sel] * np.asarray(theta, float)[..., None]
    c, s = np.cos(a), np.sin(a)
    return c @ b.rc[sel] + s @ b.rs[sel], s @ b.zs[sel] + c @ b.zc[sel]


def axisymmetrize_grid(R, Z):
    """Grid-form axisymmetrization: average over phi at fixed theta. R, Z: (ntheta, nphi)."""
    return np.mean(R, axis=1), np.mean(Z, axis=1)
