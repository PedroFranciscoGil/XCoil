"""High-resolution FP64 certification, batched and jittable.

Every candidate is re-evaluated at higher resolution against the TRUE limits (no margins):

  f_B,hi, mean / max |B.n|/|B|   same kernel on the full torus, verify_grid points per field period,
                                  N_q = nq_verify (area-weighted mean); the max is taken over this
                                  grid and the fine grid (MAX_GRID_FACTOR x finer in each direction)
  length                          quadrature of |gamma'|, N_q = nq_verify
  max curvature                   4096 samples per coil, then 3 golden-section refinements around
                                  the top 4 local maxima per coil
  MSC                             4096 samples per coil (arclength-weighted, as in optimization)
  coil-coil distance              exact segment-to-segment distance on 1024-segment polylines,
                                  restricted to the 8 nearest vertices per vertex (brute-force
                                  vertex pass), both segments adjacent to each vertex
  coil-surface distance           exact point-to-triangle distance against the triangulated
                                  verification grid, restricted to the 16 nearest triangles
                                  (by centroid) per point, 1024 points per coil
  linking                         Gauss linking integral per physical-coil pair, 512 points per
                                  coil, rounded

v(x) = max_j max(0, g_j) over (L - eps)/eps, (kappa_max - kappa_lim)/kappa_lim,
(MSC_c - MSC_lim)/MSC_lim, (d_cc,lim - d_cc)/d_cc,lim, (d_cs,lim - d_cs)/d_cs,lim.
Certified <=> v <= viol_tol, every linking number is 0, and no NaN.
Magnetic convergence: f_B on the fine grid, |f(kN) - f(N)| / max(f(kN), 1e-8) < 1 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from xcoil.curves import fourier as fr
from xcoil.curves import currents as cur
from xcoil.curves import symmetry as sym_mod
from xcoil.constraints import pointwise as pw
from xcoil.field import residuals as rs
from xcoil.homotopy import surface_path as sp

N_KAPPA = 4096; N_KAPPA_PEAKS = 4; N_GOLDEN = 3
N_SEG = 1024; N_NEAR_V = 8
N_PTS_CS = N_SEG; N_NEAR_T = 16
N_LINK = 512
# fine grid for the maximum of |B.n|/|B| (and the magnetic convergence check). With free currents the
# maximum can sit in a narrow peak that a doubled grid misses by up to ~1 %; grids 3x and 4x finer
# agree to ~0.03 %, so the fine grid is 3x finer in each direction.
MAX_GRID_FACTOR = 3
FLUX_CHUNK = 16384                       # surface points per field evaluation (bounds the memory)
GOLD = (math.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class VerifyTables:
    X: jnp.ndarray; n: jnp.ndarray; w: jnp.ndarray            # full-torus flux grid at t = 1
    X2: jnp.ndarray; n2: jnp.ndarray; w2: jnp.ndarray         # fine grid (MAX_GRID_FACTOR x): max, convergence
    tri: jnp.ndarray                                          # (Ntri, 3, 3) triangle vertices
    tri_c: jnp.ndarray                                        # (Ntri, 3) centroids
    cbq: fr.CurveBasis                                        # N_q = nq_verify
    cbk: fr.CurveBasis                                        # N_KAPPA samples
    cbs: fr.CurveBasis                                        # N_SEG polyline vertices
    cbl: fr.CurveBasis                                        # N_LINK samples
    sym: sym_mod.Symmetry
    B_ref: jnp.ndarray; I0: jnp.ndarray
    kind: str; order: int


jax.tree_util.register_pytree_node(
    VerifyTables,
    lambda V: ((V.X, V.n, V.w, V.X2, V.n2, V.w2, V.tri, V.tri_c, V.cbq, V.cbk, V.cbs, V.cbl,
                V.sym, V.B_ref, V.I0), (V.kind, V.order)),
    lambda aux, ch: VerifyTables(*ch, *aux))


def _full_grid(b, npp, nth):
    """Full-torus flux grid with npp points per field period (half-cell phi offset), at t = 1."""
    tab = sp.build(b, npp * b.nfp, nth, "full", "linear")
    return sp.evaluate(tab, 1.0)


def _triangulation(b, npp, nth):
    """Vertices on the full-torus verification grid (phi = 2 pi i / nphi, theta = 2 pi j / nth),
    each quad split into two triangles, periodic in both angles."""
    nph = npp * b.nfp
    ph = 2 * np.pi * np.arange(nph) / nph; th = 2 * np.pi * np.arange(nth) / nth
    PH, TH = np.meshgrid(ph, th, indexing="ij")
    V = np.asarray(b.eval_xyz(TH.ravel(), PH.ravel())).reshape(nph, nth, 3)
    i, j = np.meshgrid(np.arange(nph), np.arange(nth), indexing="ij")
    i1, j1 = (i + 1) % nph, (j + 1) % nth
    t1 = np.stack([V[i, j], V[i1, j], V[i1, j1]], -2).reshape(-1, 3, 3)
    t2 = np.stack([V[i, j], V[i1, j1], V[i, j1]], -2).reshape(-1, 3, 3)
    T = np.concatenate([t1, t2])
    return jnp.asarray(T), jnp.asarray(T.mean(1))


def build(boundary, cfg, T):
    """VerifyTables from a problem Tables T (B_ref, I0 shared with the optimization)."""
    npp, nth = cfg.verify_grid
    X, n, w = _full_grid(boundary, npp, nth)
    X2, n2, w2 = _full_grid(boundary, MAX_GRID_FACTOR * npp, MAX_GRID_FACTOR * nth)
    tri, tri_c = _triangulation(boundary, npp, nth)
    order = cfg.order if cfg.basis == "fourier" else cfg.n_ctrl
    mk = lambda nq: fr.make_basis(cfg.basis, nq, order)
    return VerifyTables(X, n, w, X2, n2, w2, tri, tri_c, mk(cfg.nq_verify), mk(N_KAPPA),
                        mk(N_SEG), mk(N_LINK), T.sym, T.B_ref, T.I0, cfg.basis, order)


# ---------------------------------------------------------------------------------------------
# curves at arbitrary parameters (golden-section refinement)
# ---------------------------------------------------------------------------------------------

def _basis_at(kind, order, s):
    if kind == "fourier":
        m = jnp.arange(1, order + 1)
        w = 2 * jnp.pi * m
        a = w[None, :] * s[:, None]
        sn, cs = jnp.sin(a), jnp.cos(a)
        z = jnp.zeros_like(s)[:, None]
        inter = lambda u, v: jnp.stack([u, v], -1).reshape(s.shape[0], -1)
        B1 = jnp.concatenate([z, inter(w * cs, -w * sn)], 1)
        B2 = jnp.concatenate([z, inter(-w * w * sn, -w * w * cs)], 1)
        return B1, B2
    from xcoil.geometry import periodic_bspline as pb
    Bs = pb.basis_matrices(s, order)
    return Bs[1], Bs[2]


def _kappa_at(x_c, kind, order, s):
    B1, B2 = _basis_at(kind, order, jnp.mod(s, 1.0))
    Xc = x_c.reshape(3, -1).T
    return pw.curvature(B1 @ Xc, B2 @ Xc)


def max_curvature(x, V: VerifyTables, C):
    """Dense 4096 samples, then 3 golden-section steps around the top 4 local maxima per coil."""
    _, gd, gdd = fr.coils_curves(x, V.cbk, C)
    kap = pw.curvature(gd, gdd)                                               # (C, N)
    N = kap.shape[1]; h = 1.0 / N
    is_max = (kap >= jnp.roll(kap, 1, 1)) & (kap >= jnp.roll(kap, -1, 1))
    top_v, top_i = jax.lax.top_k(jnp.where(is_max, kap, -jnp.inf), N_KAPPA_PEAKS)

    def refine(x_c, idx):
        s0 = idx.astype(x.dtype) * h
        a, b = s0 - h, s0 + h
        f = lambda s: _kappa_at(x_c, V.kind, V.order, s)
        best = f(s0)
        for _ in range(N_GOLDEN):
            c1 = b - GOLD * (b - a); c2 = a + GOLD * (b - a)
            f1, f2 = f(c1), f(c2)
            best = jnp.maximum(best, jnp.maximum(f1, f2))
            left = f1 > f2
            a, b = jnp.where(left, a, c1), jnp.where(left, c2, b)
        return best
    ref = jax.vmap(refine)(cur.geo(x, V.cbk, C).reshape(C, -1), top_i)                          # (C, 4)
    return jnp.maximum(jnp.max(kap), jnp.max(jnp.where(jnp.isfinite(top_v), ref, -jnp.inf)))


# ---------------------------------------------------------------------------------------------
# exact distances
# ---------------------------------------------------------------------------------------------

def segment_distance(p1, q1, p2, q2):
    """Exact distance between segments [p1, q1] and [p2, q2], elementwise over leading axes
    (Ericson, Real-Time Collision Detection 5.1.9, branch-free; degenerate segments allowed)."""
    tiny = 1e-300
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = jnp.sum(d1 * d1, -1); e = jnp.sum(d2 * d2, -1); f = jnp.sum(d2 * r, -1)
    c = jnp.sum(d1 * r, -1); b = jnp.sum(d1 * d2, -1)
    sa = jnp.where(a > tiny, a, 1.0); se = jnp.where(e > tiny, e, 1.0)
    den = a * e - b * b
    # general case
    s_g = jnp.where(den > tiny, jnp.clip((b * f - c * e) / jnp.where(den > tiny, den, 1.0), 0.0, 1.0), 0.0)
    t_g = (b * s_g + f) / se
    s_g = jnp.where(t_g < 0, jnp.clip(-c / sa, 0.0, 1.0), jnp.where(t_g > 1, jnp.clip((b - c) / sa, 0.0, 1.0), s_g))
    t_g = jnp.clip(t_g, 0.0, 1.0)
    # degenerate cases
    s_e0 = jnp.clip(-c / sa, 0.0, 1.0)                  # second segment is a point (t = 0)
    t_a0 = jnp.clip(f / se, 0.0, 1.0)                   # first segment is a point (s = 0)
    a0, e0 = a <= tiny, e <= tiny
    s = jnp.where(a0, 0.0, jnp.where(e0, s_e0, s_g))
    t = jnp.where(a0, jnp.where(e0, 0.0, t_a0), jnp.where(e0, 0.0, t_g))
    c1 = p1 + s[..., None] * d1; c2 = p2 + t[..., None] * d2
    return jnp.linalg.norm(c1 - c2, axis=-1)


def point_triangle_distance(p, tri):
    """Exact distance from points p (..., 3) to triangles tri (..., 3, 3): interior projection if
    inside, otherwise the nearest edge (segment distance with a degenerate first segment)."""
    A, B, Cc = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    nrm = jnp.cross(B - A, Cc - A)
    nn = jnp.linalg.norm(nrm, axis=-1, keepdims=True)
    u = nrm / jnp.where(nn > 0, nn, 1.0)
    dpl = jnp.sum((p - A) * u, -1)
    q = p - dpl[..., None] * u
    inside = ((jnp.sum(jnp.cross(B - A, q - A) * nrm, -1) >= 0)
              & (jnp.sum(jnp.cross(Cc - B, q - B) * nrm, -1) >= 0)
              & (jnp.sum(jnp.cross(A - Cc, q - Cc) * nrm, -1) >= 0))
    de = jnp.minimum(jnp.minimum(segment_distance(p, p, A, B), segment_distance(p, p, B, Cc)),
                     segment_distance(p, p, Cc, A))
    return jnp.where(inside, jnp.abs(dpl), de)


def coil_coil_distance(x, V: VerifyTables):
    C = V.sym.C
    g, _, _ = fr.coils_curves(x, V.cbs, C)                                    # (C, N, 3)
    G = sym_mod.expand(g, V.sym)                                              # (J, N, 3)
    J, N = G.shape[0], G.shape[1]
    own = pw.own_mask(V.sym)                                                  # (C, J)
    Gf = G.reshape(-1, 3); nxt = jnp.roll(G, -1, 1).reshape(-1, 3); prv = jnp.roll(G, 1, 1).reshape(-1, 3)
    valid = jnp.repeat(own, N, axis=1)                                        # (C, J*N)

    def per_coil(gc, vc):
        d2 = jnp.sum((gc[:, None, :] - Gf[None]) ** 2, -1)                    # (N, J*N)
        d2 = jnp.where(vc[None], d2, jnp.inf)
        _, idx = jax.lax.top_k(-d2, N_NEAR_V)                                  # (N, 8)
        p1 = gc[:, None, :]; q1 = jnp.roll(gc, -1, 0)[:, None, :]; o1 = jnp.roll(gc, 1, 0)[:, None, :]
        P2, Q2, O2 = Gf[idx], nxt[idx], prv[idx]
        ds = jnp.stack([segment_distance(p1, q1, P2, Q2), segment_distance(p1, q1, O2, P2),
                        segment_distance(o1, p1, P2, Q2), segment_distance(o1, p1, O2, P2)])
        return jnp.min(ds)
    return jnp.min(jax.lax.map(lambda a: per_coil(*a), (g, valid)))


def coil_surface_distance(x, V: VerifyTables):
    C = V.sym.C
    g, _, _ = fr.coils_curves(x, V.cbs, C)                                    # N_PTS_CS = N_SEG points

    def per_coil(gc):
        d2 = jnp.sum((gc[:, None, :] - V.tri_c[None]) ** 2, -1)               # (N, Ntri)
        _, idx = jax.lax.top_k(-d2, N_NEAR_T)
        return jnp.min(point_triangle_distance(gc[:, None, :], V.tri[idx]))
    return jnp.min(jax.lax.map(per_coil, g))


def linking_numbers(x, V: VerifyTables):
    """Rounded Gauss linking numbers of all physical-coil pairs (J, J), diagonal 0."""
    C = V.sym.C
    g, gd, _ = fr.coils_curves(x, V.cbl, C)
    G, Gd = sym_mod.expand(g, V.sym), sym_mod.expand(gd, V.sym)
    J, N = G.shape[0], G.shape[1]

    def pair(a, b):
        D = G[a][:, None, :] - G[b][None, :, :]
        r3 = jnp.linalg.norm(D, axis=-1) ** 3
        cr = jnp.cross(Gd[a][:, None, :], Gd[b][None, :, :])
        return jnp.sum(jnp.sum(cr * D, -1) / r3) / (N * N) / (4 * jnp.pi)
    ia, ib = np.triu_indices(J, 1)
    vals = jax.lax.map(lambda ab: pair(ab[0], ab[1]), (jnp.asarray(ia), jnp.asarray(ib)))
    Lk = jnp.zeros((J, J)).at[ia, ib].set(jnp.rint(vals))
    return Lk + Lk.T, vals


# ---------------------------------------------------------------------------------------------
# flux quality
# ---------------------------------------------------------------------------------------------

def field_chunked(x, V: VerifyTables, X, chunk=FLUX_CHUNK, field=None):
    """B at X, evaluated in chunks of surface points (lax.map) so that fine grids fit in memory."""
    field = field or (lambda P: rs.field_on(x, V.cbq, V.sym, P, V.I0))
    N = X.shape[0]
    if N <= chunk:
        return field(X)
    nc = -(-N // chunk)
    Xp = jnp.concatenate([X, jnp.broadcast_to(X[:1], (nc * chunk - N, 3))]).reshape(nc, chunk, 3)
    return jax.lax.map(field, Xp).reshape(nc * chunk, 3)[:N]


def flux_metrics(x, V: VerifyTables, X, n, w):
    B = field_chunked(x, V, X)
    Bn = jnp.sum(B * n, 1); Bm = jnp.linalg.norm(B, axis=1)
    W = jnp.sum(w)
    f = jnp.sum(w * Bn ** 2) / W / V.B_ref ** 2
    ratio = jnp.abs(Bn) / Bm
    return f, jnp.sum(w * ratio) / W, jnp.max(ratio)


# ---------------------------------------------------------------------------------------------
# one candidate
# ---------------------------------------------------------------------------------------------

def certify_one(x, eps, V: VerifyTables, lim: pw.Limits, viol_tol=1e-3):
    C = V.sym.C
    f_hi, mean_bn, max_bn = flux_metrics(x, V, V.X, V.n, V.w)
    f_2n, _, max_2n = flux_metrics(x, V, V.X2, V.n2, V.w2)
    # the maximum of |B.n|/|B| can sit in a narrow peak: take it over both grids (free-current QI:
    # the doubled grid is up to 1.2 % low, 3x and 4x agree to 0.03 %); the mean is converged on the first
    max_bn = jnp.maximum(max_bn, max_2n)
    _, gdq, _ = fr.coils_curves(x, V.cbq, C)
    Lc = pw.lengths(gdq)
    L = jnp.sum(Lc)
    eps, s_k = pw.split_cap(eps); kl = s_k * lim.kappa_lim
    L_viol = (jnp.max(Lc) - eps) / eps if lim.per_coil_length else (L - eps) / eps
    kmax = max_curvature(x, V, C)
    _, gdk, gddk = fr.coils_curves(x, V.cbk, C)
    msc = pw.msc(gdk, gddk)                                                   # (C,)
    arcv = pw.arc_variance(gdk)                              # (C,) 'full' definition, 4096 samples
    dcc = coil_coil_distance(x, V)
    dcs = coil_surface_distance(x, V)
    Lk, Lraw = linking_numbers(x, V)
    g = jnp.stack([L_viol, (kmax - kl) / kl,
                   jnp.max(msc - lim.msc_lim) / lim.msc_lim,
                   (lim.dcc - dcc) / lim.dcc, (lim.dcs - dcs) / lim.dcs,
                   (jnp.max(arcv) - lim.arc_var_lim) / lim.arc_var_lim if lim.arc_var_lim > 0 else jnp.asarray(-1.0)])
    v = jnp.max(jnp.maximum(g, 0.0))
    vals = jnp.stack([f_hi, f_2n, mean_bn, max_bn, L, kmax, dcc, dcs, v])
    finite = jnp.all(jnp.isfinite(vals)) & jnp.all(jnp.isfinite(x))
    unlinked = jnp.all(Lk == 0)
    mag_conv = jnp.abs(f_2n - f_hi) / jnp.maximum(f_2n, 1e-8) < 0.01
    return dict(f_B_hi=f_hi, f_B_2n=f_2n, mag_conv=mag_conv, mean_bn=mean_bn, max_bn=max_bn,
                L=L, L_max_coil=jnp.max(Lc), I_rel=cur.base_currents(x, V.cbq, C, 1.0), arc_var_max=jnp.max(arcv), max_kappa=kmax, msc=msc, d_cc=dcc, d_cs=dcs, linking=Lk,
                linking_raw_max=jnp.max(jnp.abs(Lraw)), g=g, v=v,
                certified=(v <= viol_tol) & unlinked & finite, linked=~unlinked, finite=finite)


def certify(X, eps, V: VerifyTables, lim: pw.Limits, viol_tol=1e-3):
    """Batched certification of candidates X (P, D) with caps eps (P,) (lax.map: bounded memory)."""
    return jax.lax.map(lambda a: certify_one(a[0], a[1], V, lim, viol_tol), (X, eps))
