"""Pointwise geometric constraints. All g <= 0 means feasible.

    g = [ (L - eps)/eps                                          (1)
          (kappa_ck - (1 - beta) kappa_lim) / kappa_lim           (C N_q)  pointwise, no LSE max
          (MSC_c - (1 - beta) MSC_lim) / MSC_lim                  (C)
          ((1 + beta_d) d_cc - softmin_cc(c, k)) / d_cc            (C N_q)
          ((1 + beta_d) d_cs - softmin_cs(c, k; t)) / d_cs          (C N_q) ]

Curvature kappa = |gamma' x gamma''| / |gamma'|^3. Length = sum_c mean_k |gamma'_ck|.
MSC is the arclength-weighted mean of kappa^2 (simsopt MeanSquaredCurvature); it equals the plain
mean over quadrature points at uniform speed, and using the physical definition keeps the
optimization constraint identical to the certified metric.

Soft-min d_soft = -(1/alpha) log sum exp(-alpha d) (a sum, not a mean): a conservative lower
bound, d_soft <= min d <= d_soft + log(n)/alpha. Coil-coil: over all points of every other
physical coil (the identity copy of the same base coil excluded). Coil-surface: over the full-
torus distance grid of Sigma(t).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@dataclass(frozen=True)
class Limits:
    kappa_lim: float
    msc_lim: float
    dcc: float
    dcs: float
    beta: float = 0.03       # optimization margin on curvature / MSC (removed at certification)
    beta_d: float = 0.0      # distance margin (soft-min is already conservative)
    alpha_cc: float = 0.0    # soft-min sharpness; 0 -> softmin_alpha / d_cc
    alpha_cs: float = 0.0
    n_seg: int = 0           # distance rows per coil: 0 = pointwise; n > 0 = soft-min over
                             # n contiguous segments
    per_coil_length: bool = False  # True: one length row per base coil, L_c <= eps (simsopt
                                   # auglag_qa convention); False: total sum_c L_c <= eps
    arc_var_lim: float = 0.0       # > 0: one row per base coil, Var_k |gamma'_ck| <= arc_var_lim [m^2]
                                   # (simsopt ArclengthVariation, nintervals="full"); 0 = off


# limit values (and the soft-min sharpness derived from them) are pytree data, so one compiled
# program serves devices with different limits; margins, row layout and switches are structure
jax.tree_util.register_pytree_node(
    Limits,
    lambda L: ((L.kappa_lim, L.msc_lim, L.dcc, L.dcs, L.alpha_cc, L.alpha_cs),
               (L.beta, L.beta_d, L.n_seg, L.per_coil_length, L.arc_var_lim)),
    lambda aux, ch: Limits(*ch[:4], aux[0], aux[1], ch[4], ch[5], *aux[2:]))


def with_alpha(lim: Limits, softmin_alpha=100.0) -> Limits:
    return Limits(lim.kappa_lim, lim.msc_lim, lim.dcc, lim.dcs, lim.beta, lim.beta_d,
                  softmin_alpha / lim.dcc, softmin_alpha / lim.dcs, lim.n_seg, lim.per_coil_length,
                  lim.arc_var_lim)


def segment_softmin(s, alpha, n_seg):
    """(C, nq) pointwise soft-mins -> (C, n_seg) soft-min over contiguous segments and the
    in-segment weights w (C, nq). Exact identity: the soft-min over the union of a segment's
    point sets equals the soft-min (same alpha) of the points' soft-mins, so the segment row's
    derivative is sum_k w_k d s_k."""
    C, nq = s.shape
    sg = s.reshape(C, n_seg, nq // n_seg)
    val = -logsumexp(-alpha * sg, axis=-1) / alpha
    w = jax.nn.softmax(-alpha * sg, axis=-1)
    return val, w.reshape(C, nq)


def _seg_rows(s, alpha, n_seg):
    return s if n_seg == 0 else segment_softmin(s, alpha, n_seg)[0]


def _seg_jac(J, s, alpha, n_seg):
    """J (C*nq, D) pointwise rows -> (C*n_seg, D) segment rows."""
    if n_seg == 0:
        return J
    C, nq = s.shape
    _, w = segment_softmin(s, alpha, n_seg)
    return jnp.einsum("csk,cskd->csd", w.reshape(C, n_seg, nq // n_seg),
                      J.reshape(C, n_seg, nq // n_seg, -1)).reshape(C * n_seg, -1)


def curvature(gd, gdd):
    sp = jnp.linalg.norm(gd, axis=-1)
    return jnp.linalg.norm(jnp.cross(gd, gdd), axis=-1) / sp ** 3


def lengths(gd):
    return jnp.mean(jnp.linalg.norm(gd, axis=-1), axis=-1)


def arc_variance(gd):
    """simsopt ArclengthVariation (nintervals="full"): variance of the incremental arclength
    |gamma'| over the quadrature points, per coil. gd (..., nq, 3) -> (...)."""
    return jnp.var(jnp.linalg.norm(gd, axis=-1), axis=-1)


def msc(gd, gdd):
    sp = jnp.linalg.norm(gd, axis=-1)
    k = curvature(gd, gdd)
    return jnp.sum(k ** 2 * sp, axis=-1) / jnp.sum(sp, axis=-1)


def softmin(d, alpha, where=None, axis=-1):
    a = -alpha * d
    if where is not None:
        # finite fill (exp underflows to exactly 0): identical values, and reverse-mode second
        # derivatives stay finite (a -inf fill gave garbage Hessian-vector products)
        a = jnp.where(where, a, -1e30)
    return -logsumexp(a, axis=axis) / alpha


def own_mask(sym):
    """(C, Nphys) True where physical coil j is NOT the identity copy of base coil c."""
    C = sym.C
    same = (sym.coil[None, :] == jnp.arange(C)[:, None]) & sym.ident[None, :]
    return ~same


def _k(a, kdt):
    """Mixed precision: round an FP64 array (a separation) to the kernel dtype; None = as is."""
    return a if kdt is None else a.astype(kdt)


def coil_coil_softmin(g_base, G_phys, sym, alpha, kdt=None):
    """(C, nq) soft-min distance from each base-coil point to all other physical coils.
    kdt: kernel dtype - separations are formed in the input precision (FP64) and rounded to kdt,
    the distance / log-sum-exp kernel runs in kdt, the result is returned in the input dtype."""
    C, nq, _ = g_base.shape
    J, nq2, _ = G_phys.shape
    # component-wise (x, y, z separate): XLA fuses the pair terms into the reduction instead of
    # writing (C, nq, J, nq', 3) separations to memory (20 % faster on a GPU, identical values)
    comp = lambda i: _k(g_base[:, :, None, None, i] - G_phys[None, None, :, :, i], kdt)   # (C,nq,J,nq')
    dx, dy, dz = comp(0), comp(1), comp(2)
    mask = jnp.broadcast_to(own_mask(sym)[:, None, :, None], (C, nq, J, nq2))
    # masked (own-coil) entries include d = 0 at the row's own point: keep them finite so no
    # NaN reaches the derivatives through sqrt(0) or 0/0
    d = jnp.sqrt(jnp.where(mask, dx * dx + dy * dy + dz * dz, 1.0))
    return softmin(d.reshape(C, nq, J * nq2), alpha, mask.reshape(C, nq, J * nq2)).astype(g_base.dtype)


def _surface_diff(g_base, S_pts):
    """Separations (C, nq, P, 3) from the base-coil points to S_pts: the whole distance grid (P, 3), or a window of
    surface points per coil point (C, nq, P, 3) (device_lm cs_window)."""
    return g_base[:, :, None, :] - (S_pts if S_pts.ndim == 4 else S_pts[None, None, :, :])


def coil_surface_softmin(g_base, S_pts, alpha, kdt=None):
    """(C, nq) soft-min distance from each base-coil point to the surface distance grid (or its window, see
    _surface_diff)."""
    diff = _k(_surface_diff(g_base, S_pts), kdt)
    return softmin(jnp.sqrt(jnp.sum(diff * diff, axis=-1)), alpha).astype(g_base.dtype)


def split_cap(eps):
    """The front parameter of the constraints: a scalar length cap, or [length cap, s] with s the
    scale of the curvature limit (curvature-limit fronts: kappa_lim -> s kappa_lim at run time)."""
    e = jnp.asarray(eps)
    if e.ndim == 0:
        return e, 1.0
    return e[0], e[1]


def constraint_values(g, gd, gdd, G_phys, S_dist, sym, lim: Limits, eps, kdt=None):
    """Stacked normalized constraint vector (m,) from base-coil curves (C, nq, 3). kdt: kernel
    dtype of the two distance soft-mins (length / curvature / MSC stay in the input dtype)."""
    eps, s_k = split_cap(eps); kl = s_k * lim.kappa_lim
    Lc = lengths(gd)
    L_rows = ((Lc - eps) / eps) if lim.per_coil_length else jnp.reshape((jnp.sum(Lc) - eps) / eps, (1,))
    kap = curvature(gd, gdd)
    M = msc(gd, gdd)
    arc_rows = ([(arc_variance(gd) - lim.arc_var_lim) / lim.arc_var_lim] if lim.arc_var_lim > 0 else [])
    dcc = coil_coil_softmin(g, G_phys, sym, lim.alpha_cc, kdt)
    dcs = coil_surface_softmin(g, S_dist, lim.alpha_cs, kdt)
    return jnp.concatenate([
        L_rows,
        ((kap - (1 - lim.beta) * kl) / kl).ravel(),
        (M - (1 - lim.beta) * lim.msc_lim) / lim.msc_lim,
        *arc_rows,
        (((1 + lim.beta_d) * lim.dcc - _seg_rows(dcc, lim.alpha_cc, lim.n_seg)) / lim.dcc).ravel(),
        (((1 + lim.beta_d) * lim.dcs - _seg_rows(dcs, lim.alpha_cs, lim.n_seg)) / lim.dcs).ravel()])


def n_constraints(C, nq, n_seg=0, per_coil_length=False, arc=False):
    nd = C * (n_seg or nq)
    return (C if per_coil_length else 1) + C * nq + C + (C if arc else 0) + 2 * nd


def blocks(C, nq, n_seg=0, per_coil_length=False, arc=False):
    """Index slices of the constraint blocks in the stacked vector."""
    i = 0; out = {}; nd = C * (n_seg or nq)
    for name, n in (("L", C if per_coil_length else 1), ("curv", C * nq), ("msc", C),
                    ("arc", C if arc else 0), ("cc", nd), ("cs", nd)):
        out[name] = slice(i, i + n); i += n
    return out


# ----------------------------------------------------------------------------------------------
# x-level values and the analytic constraint Jacobian A = dg/dx (sec. 6.2, 7.2)
# ----------------------------------------------------------------------------------------------

def values_x(x, cb, sym, S_dist, lim: Limits, eps, kdt=None):
    from xcoil.curves import fourier as fr
    from xcoil.curves import symmetry as sym_mod
    g, gd, gdd = fr.coils_curves(x, cb, sym.C)
    return constraint_values(g, gd, gdd, sym_mod.expand(g, sym), S_dist, sym, lim, eps, kdt)


def blocks_for(C, nq, lim: Limits):
    """blocks() with every layout option taken from the limits."""
    return blocks(C, nq, lim.n_seg, lim.per_coil_length, lim.arc_var_lim > 0)


def _coil_arcvar(x_c, cb):
    from xcoil.curves import fourier as fr
    return arc_variance(fr.coil_curve(x_c, cb)[1])


def _coil_geom(x_c, cb):
    from xcoil.curves import fourier as fr
    g, gd, gdd = fr.coil_curve(x_c, cb)
    return jnp.mean(jnp.linalg.norm(gd, axis=-1)), curvature(gd, gdd), msc(gd[None], gdd[None])[0]


def _softmin_weights(d, alpha, mask=None):
    a = -alpha * d
    if mask is not None:
        a = jnp.where(mask, a, -jnp.inf)
    return jax.nn.softmax(a, axis=-1)


def coil_coil_jacobian(g, cb, sym, alpha, kdt=None):
    """d(softmin_cc)/dx: (C*nq, C*3K). Row (c,k): own point term sum_j pi u, other points -pi u
    rotated back with R_j^T and summed over the copies of each base coil. kdt: kernel dtype
    (FP64 separations rounded to kdt, kernel and contractions in kdt, FP64 out)."""
    from xcoil.curves import symmetry as sym_mod
    C, nq, _ = g.shape
    G = sym_mod.expand(g, sym)                                        # (J, nq, 3)
    J = G.shape[0]; ncopy = J // C
    diff = _k(g[:, :, None, None, :] - G[None, None], kdt)            # (C,nq,J,nq,3)
    mask4 = jnp.broadcast_to(own_mask(sym)[:, None, :, None], (C, nq, J, nq))
    d = jnp.sqrt(jnp.where(mask4, jnp.sum(diff * diff, -1), 1.0))    # masked entries finite
    u = diff / d[..., None]
    mask = mask4.reshape(C, nq, J * nq)
    pi = _softmin_weights(d.reshape(C, nq, J * nq), alpha, mask).reshape(C, nq, J, nq)
    wu = pi[..., None] * u                                            # (C,nq,J,nq,3)
    own = jnp.sum(wu, axis=(2, 3))                                    # (C,nq,3)
    R = _k(sym.R, kdt).reshape(ncopy, C, 3, 3)
    other = -jnp.einsum("rcba,xkrcqb->xkcqa", R, wu.reshape(C, nq, ncopy, C, nq, 3))
    eyeC = jnp.eye(C, dtype=wu.dtype); eyeq = jnp.eye(nq, dtype=wu.dtype)
    other = other + jnp.einsum("xc,kq,xka->xkcqa", eyeC, eyeq, own)  # own point p = gamma_ck
    Jx = jnp.einsum("xkcqd,qm->xkcdm", other, _k(cb.B0, kdt))         # (C,nq,C,3,K)
    return Jx.reshape(C * nq, C * 3 * cb.K).astype(g.dtype)


def coil_coil_jacobian_rows(g, cb, sym, alpha, ci, ki, kdt=None):
    """Rows (ci[r], ki[r]) of coil_coil_jacobian only: (R, C*3K). Same maths, the separations
    restricted to the selected base-coil points (cost R / (C nq) of the full block)."""
    from xcoil.curves import symmetry as sym_mod
    C, nq, _ = g.shape
    G = sym_mod.expand(g, sym)                                        # (J, nq, 3)
    J = G.shape[0]; ncopy = J // C; R = ci.shape[0]
    diff = _k(g[ci, ki][:, None, None, :] - G[None], kdt)            # (R,J,nq,3)
    mask3 = jnp.broadcast_to(own_mask(sym)[ci][:, :, None], (R, J, nq))
    d = jnp.sqrt(jnp.where(mask3, jnp.sum(diff * diff, -1), 1.0))
    u = diff / d[..., None]
    pi = _softmin_weights(d.reshape(R, J * nq), alpha, mask3.reshape(R, J * nq)).reshape(R, J, nq)
    wu = pi[..., None] * u                                            # (R,J,nq,3)
    own = jnp.sum(wu, axis=(1, 2))                                    # (R,3)
    Rm = _k(sym.R, kdt).reshape(ncopy, C, 3, 3)
    B0 = _k(cb.B0, kdt)
    other = -jnp.einsum("rcba,xrcqb->xcqa", Rm, wu.reshape(R, ncopy, C, nq, 3))    # (R,C,nq,3)
    Jx = jnp.einsum("xcqd,qm->xcdm", other, B0)                       # (R,C,3,K)
    Jx = Jx + jnp.einsum("xc,xd,xm->xcdm", jax.nn.one_hot(ci, C, dtype=wu.dtype), own, B0[ki])
    return Jx.reshape(R, C * 3 * cb.K).astype(g.dtype)


def coil_surface_jacobian(g, cb, S_pts, alpha, kdt=None):
    """d(softmin_cs)/dx: (C*nq, C*3K), block-diagonal (only the row's own coil)."""
    C, nq, _ = g.shape
    diff = _k(_surface_diff(g, S_pts), kdt)
    d = jnp.sqrt(jnp.sum(diff * diff, -1))
    pi = _softmin_weights(d, alpha)
    v = jnp.einsum("xks,xksd->xkd", pi, diff / d[..., None]).astype(g.dtype)   # (C,nq,3)
    Jx = jnp.einsum("xkd,km,xc->xkcdm", v, cb.B0, jnp.eye(C, dtype=v.dtype))
    return Jx.reshape(C * nq, C * 3 * cb.K)


def coil_surface_jacobian_rows(g, cb, S_pts, alpha, ci, ki, kdt=None):
    """Rows (ci[r], ki[r]) of coil_surface_jacobian only: (R, C*3K) (cost R / (C nq) of the full block)."""
    C, nq, _ = g.shape
    R = ci.shape[0]
    diff = _k(g[ci, ki][:, None, :] - S_pts[None], kdt)                      # (R, S, 3)
    d = jnp.sqrt(jnp.sum(diff * diff, -1))
    pi = _softmin_weights(d, alpha)
    v = jnp.einsum("xs,xsd->xd", pi, diff / d[..., None]).astype(g.dtype)   # (R, 3)
    Jx = jnp.einsum("xd,xm,xc->xcdm", v, cb.B0[ki], jax.nn.one_hot(ci, C, dtype=v.dtype))
    return Jx.reshape(R, C * 3 * cb.K)


def constraint_jacobian(x, cb, sym, S_dist, lim: Limits, eps, kdt=None, cc_rows=None, cs_rows=None):
    """Dense A (m, D) assembled from per-coil jacfwd blocks (length, curvature, MSC) and the
    analytic soft-min blocks (coil-coil, coil-surface). kdt: kernel dtype of the soft-min blocks.
    cc_rows: optional (ci, ki) - only those coil-coil rows are evaluated, all other coil-coil rows
    are returned as zero (exact for any use that weights rows by a vector vanishing off cc_rows,
    e.g. the AL terms A^T h and A_a^T A_a restricted to rows with h > 0); cs_rows: the same for
    the coil-surface rows."""
    eps, s_k = split_cap(eps)
    from xcoil.curves import fourier as fr
    from xcoil.curves import currents as cur
    C = sym.C; xc = cur.geo(x, cb, C).reshape(C, -1); Dc = xc.shape[1]
    Jl, Jk, Jm = jax.vmap(jax.jacfwd(lambda v: _coil_geom(v, cb)))(xc)   # (C,Dc),(C,nq,Dc),(C,Dc)
    nq = Jk.shape[1]
    eye = jnp.eye(C, dtype=x.dtype)
    blk = lambda Jb: jnp.einsum("cqd,ce->cqed", Jb, eye).reshape(Jb.shape[0] * Jb.shape[1], C * Dc)
    g = fr.coils_curves(x, cb, C)[0]
    if cc_rows is None or lim.n_seg:
        Jcc = coil_coil_jacobian(g, cb, sym, lim.alpha_cc, kdt)
    else:
        ci, ki = cc_rows
        Jcc = jnp.zeros((C * nq, C * Dc), x.dtype).at[ci * nq + ki].set(
            coil_coil_jacobian_rows(g, cb, sym, lim.alpha_cc, ci, ki, kdt))
    if cs_rows is None or lim.n_seg:
        Jcs = coil_surface_jacobian(g, cb, S_dist, lim.alpha_cs, kdt)
    else:
        ci, ki = cs_rows
        Jcs = jnp.zeros((C * nq, C * Dc), x.dtype).at[ci * nq + ki].set(
            coil_surface_jacobian_rows(g, cb, S_dist, lim.alpha_cs, ci, ki, kdt))
    if lim.n_seg:
        from xcoil.curves import symmetry as sym_mod
        Jcc = _seg_jac(Jcc, coil_coil_softmin(g, sym_mod.expand(g, sym), sym, lim.alpha_cc, kdt), lim.alpha_cc, lim.n_seg)
        Jcs = _seg_jac(Jcs, coil_surface_softmin(g, S_dist, lim.alpha_cs, kdt), lim.alpha_cs, lim.n_seg)
    Ja = ([blk(jax.vmap(jax.grad(lambda v: _coil_arcvar(v, cb)))(xc)[:, None, :]) / lim.arc_var_lim]
          if lim.arc_var_lim > 0 else [])
    return cur.pad_cols(jnp.concatenate([                      # geometric rows: zero current columns
        (blk(Jl[:, None, :]) if lim.per_coil_length else Jl.reshape(1, C * Dc)) / eps,
        blk(Jk) / (s_k * lim.kappa_lim),
        blk(Jm[:, None, :]) / lim.msc_lim,
        *Ja,
        -Jcc / lim.dcc,
        -Jcs / lim.dcs]), x, cb, C)
