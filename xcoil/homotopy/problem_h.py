"""Coil problem P(t, epsilon): objective residuals and constraint rows on a boundary path.

Tables is a pytree of static arrays (basis, symmetry maps, surface-path tables for one flux grid
and the distance grid, B_ref, I0, L_ref, eps(0)) with the limits and HConfig as static aux data.
It is passed as an ARGUMENT to the jitted programs (never closed over as constants). The per-
candidate homotopy parameter is p = (t, eps_i): the cap follows eps_i(t) = eps(0) + (eps_i -
eps(0)) h(t) (sec. 10.2).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod
from xcoil.curves import currents as cur
from xcoil.constraints import pointwise as pw
from xcoil.field import residuals as rs
from xcoil.field import jacobian as jac
from . import schedules as sch
from . import surface_path as sp
from .config import HConfig


@dataclass(frozen=True)
class Tables:
    cb: fr.CurveBasis
    sym: sym_mod.Symmetry
    flux: sp.PathTables
    dist: sp.PathTables
    B_ref: jnp.ndarray
    I0: jnp.ndarray
    L_ref: jnp.ndarray
    eps0: jnp.ndarray
    lim: pw.Limits
    cfg: HConfig


jax.tree_util.register_pytree_node(
    Tables,
    lambda T: ((T.cb, T.sym, T.flux, T.dist, T.B_ref, T.I0, T.L_ref, T.eps0, T.lim, T.cfg), ()),
    lambda aux, ch: Tables(*ch))


def limits_from(cfg: HConfig) -> pw.Limits:
    return pw.with_alpha(pw.Limits(cfg.kappa_lim, cfg.msc_lim, cfg.dcc, cfg.dcs, cfg.beta_curv,
                                   cfg.beta_d, n_seg=cfg.dist_segments,
                                   per_coil_length=cfg.per_coil_length, arc_var_lim=cfg.arc_var_lim),
                         cfg.softmin_alpha)


def cap_at(T: Tables, p):
    """Constraint caps at p = (t, front parameter): the length cap, or - curvature-limit fronts,
    cfg.front_param == "kappa" - [length cap front_L, kappa / kappa_lim] (see pointwise.split_cap)."""
    t, eps_i = p
    if getattr(T.cfg, "front_param", "length") == "kappa":
        return jnp.stack([jnp.asarray(sch.cap(t, T.eps0, T.cfg.front_L, T.cfg.t_h), jnp.float64),
                          jnp.asarray(eps_i, jnp.float64) / T.lim.kappa_lim])
    return sch.cap(t, T.eps0, eps_i, T.cfg.t_h)


def surface(T: Tables, t):
    return sp.evaluate(T.flux, t)


def resid_jac(T: Tables, x, p):
    t = p[0]
    return rs.stacked(x, t, T.cb, T.sym, surface(T, t), T.B_ref, T.I0, T.cfg, T.L_ref)


def resid(T: Tables, x, p):
    t = p[0]
    return rs.stacked_residual(x, t, T.cb, T.sym, surface(T, t), T.B_ref, T.I0, T.cfg, T.L_ref)


def cons(T: Tables, x, p, kdt=None):
    """kdt: kernel dtype of the distance soft-mins (None = input precision, see pointwise)."""
    return pw.values_x(x, T.cb, T.sym, sp.eval_xyz(T.dist, p[0]), T.lim, cap_at(T, p), kdt)


def cons_jac(T: Tables, x, p, kdt=None):
    S = sp.eval_xyz(T.dist, p[0]); eps = cap_at(T, p)
    return (pw.values_x(x, T.cb, T.sym, S, T.lim, eps, kdt),
            pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, eps, kdt))


def lagrangian_blockdiag(T: Tables, x, p, z):
    """Exact per-coil Hessian of the z-weighted curvature, MSC and length constraint terms,
    projected to PSD by eigenvalue clipping (option lagrangian_hessian='blockdiag')."""
    C = T.sym.C; xc = cur.geo(x, T.cb, C).reshape(C, -1); Dc = xc.shape[1]; nq = T.cb.B0.shape[0]
    bl = pw.blocks_for(C, nq, T.lim); eps, s_k = pw.split_cap(cap_at(T, p))
    zk = z[bl["curv"]].reshape(C, nq); zm = z[bl["msc"]]
    zL = z[bl["L"]] if T.lim.per_coil_length else jnp.full(C, z[bl["L"]][0])     # per coil

    def coil_term(v, zkc, zmc, zLc):
        L, kap, M = pw._coil_geom(v, T.cb)
        return (jnp.dot(zkc, kap) / (s_k * T.lim.kappa_lim) + zmc * M / T.lim.msc_lim + zLc * L / eps)
    Hc = jax.vmap(jax.hessian(coil_term))(xc, zk, zm, zL)                 # (C, Dc, Dc)
    w, V = jnp.linalg.eigh(0.5 * (Hc + jnp.swapaxes(Hc, 1, 2)))
    Hc = jnp.einsum("cij,cj,ckj->cik", V, jnp.maximum(w, 0.0), V)
    return cur.pad_square(jax.scipy.linalg.block_diag(*[Hc[c] for c in range(C)]), x, T.cb, C)


def _point_softmin_hessians(g, P_pts, alpha, mask=None):
    """3x3 Hessian of softmin_l |g_k - P_l| w.r.t. g_k for every row point g (n, 3)."""
    def one(gk, mk):
        def d(y):
            d2 = jnp.sum((y[None] - P_pts) ** 2, -1)
            if mk is not None:                  # masked rows include the point itself (d = 0)
                d2 = jnp.where(mk, d2, 1.0)
            return pw.softmin(jnp.sqrt(d2), alpha, mk)
        return jax.hessian(d)(gk)
    if mask is None:
        return jax.vmap(lambda gk: one(gk, None))(g)
    return jax.vmap(one)(g, mask)


def exact_blockdiag(T: Tables, x, p, z, r, clip=True):
    """Per-coil exact curvature missing from the Gauss-Newton matrix (option
    lagrangian_hessian = "exact_blockdiag"):
      2 c_f S_c,  S = sum_i r_i d2 r_i   (flux term in closed form, field/jacobian.py; regularizer
                                          residuals by autodiff) -- exactly block-diagonal in coils,
      + d2 (z^T g) for the length, curvature, MSC and coil-surface rows (each depends on one coil),
      + the point-self 3x3 part of the coil-coil rows (cross-coil couplings dropped),
    each coil block projected to PSD by eigenvalue clipping (as the blockdiag option)."""
    cfg = T.cfg; C = T.sym.C; xc = cur.geo(x, T.cb, C).reshape(C, -1); nq = T.cb.B0.shape[0]
    t = p[0]; bl = pw.blocks_for(C, nq, T.lim); eps, s_k = pw.split_cap(cap_at(T, p))
    X, n, w = surface(T, t); Ns = X.shape[0]
    SB = jac.residual_hessian_blocks(x, T.cb, T.sym, X, n, rs.surface_coeffs(w, T.B_ref), T.I0, r[:Ns])
    rr = r[Ns:]
    arc_b = rr[:C * nq].reshape(C, nq); L_b = rr[C * nq]
    msc_b = rr[C * nq + 1:2 * C * nq + 1].reshape(C, nq)          # (arc-penalty rows follow; not in S)
    a_arc = sch.amp_arc(cfg) / jnp.sqrt(C * nq); a_msc = sch.amp_msc(t, cfg) / jnp.sqrt(C * nq)
    a_L = sch.amp_L(t, cfg) / T.L_ref

    def reg_c(v, ab, mb):
        arc, lbar, kap = rs._coil_reg(v, T.cb, cfg.kappa_lim)
        return a_arc * jnp.dot(ab, arc) + L_b * a_L * lbar + a_msc * jnp.dot(mb, kap)
    SR = jax.vmap(jax.hessian(reg_c))(xc, arc_b, msc_b)

    zk = z[bl["curv"]].reshape(C, nq); zm = z[bl["msc"]]
    zL = z[bl["L"]] if T.lim.per_coil_length else jnp.full(C, z[bl["L"]][0])     # per coil

    def geom_c(v, zkc, zmc, zLc):
        L, kap, M = pw._coil_geom(v, T.cb)
        return jnp.dot(zkc, kap) / (s_k * T.lim.kappa_lim) + zmc * M / T.lim.msc_lim + zLc * L / eps
    HG = jax.vmap(jax.hessian(geom_c))(xc, zk, zm, zL)
    # distance rows: g = (d_lim - softmin)/d_lim, each row a function of one point gamma_ck
    g, _, _ = fr.coils_curves(x, T.cb, C)
    S_dist = sp.eval_xyz(T.dist, t)
    Hcs = _point_softmin_hessians(g.reshape(-1, 3), S_dist, T.lim.alpha_cs)
    G_phys = sym_mod.expand(g, T.sym).reshape(-1, 3)
    mask = jnp.repeat(pw.own_mask(T.sym), nq, axis=1)                               # (C, J*nq)
    mask = jnp.repeat(mask, nq, axis=0)                                             # (C*nq, J*nq)
    Hcc = _point_softmin_hessians(g.reshape(-1, 3), G_phys, T.lim.alpha_cc, mask)
    zcs, zcc = z[bl["cs"]], z[bl["cc"]]
    if T.lim.n_seg:                    # segment multipliers spread over points by the soft-min weights
        ns = T.lim.n_seg
        _, wcs = pw.segment_softmin(pw.coil_surface_softmin(g, S_dist, T.lim.alpha_cs), T.lim.alpha_cs, ns)
        _, wcc = pw.segment_softmin(pw.coil_coil_softmin(g, sym_mod.expand(g, T.sym), T.sym, T.lim.alpha_cc),
                                    T.lim.alpha_cc, ns)
        zcs = (jnp.repeat(zcs.reshape(C, ns), nq // ns, axis=1) * wcs).ravel()
        zcc = (jnp.repeat(zcc.reshape(C, ns), nq // ns, axis=1) * wcc).ravel()
    wpt = -(zcs / T.lim.dcs)[:, None, None] * Hcs - (zcc / T.lim.dcc)[:, None, None] * Hcc
    wpt = wpt.reshape(C, nq, 3, 3)
    B0 = T.cb.B0; K = T.cb.K
    HD = jnp.einsum("km,ckde,kl->cdmel", B0, wpt, B0).reshape(C, 3 * K, 3 * K)
    Hc = 2.0 * (1.0 / cfg.f_ref) * (SB + SR) + HG + HD
    Hc = 0.5 * (Hc + jnp.swapaxes(Hc, 1, 2))
    if clip:
        wv, V = jnp.linalg.eigh(Hc)
        Hc = jnp.einsum("cij,cj,ckj->cik", V, jnp.maximum(wv, 0.0), V)
    # free currents: the current block of the flux curvature is zero (r linear in I) and the
    # coefficient-current cross terms are left to the Gauss-Newton part J^T J
    return cur.pad_square(jax.scipy.linalg.block_diag(*[Hc[c] for c in range(C)]), x, T.cb, C)


def f_B(T: Tables, x, t):
    X, n, w = surface(T, t)
    r = rs.magnetic_residual(x, T.cb, T.sym, X, n, w, T.B_ref, T.I0)
    return jnp.dot(r, r)


def build_tables(boundary, cfg: HConfig, seed, grid=None, path=None):
    """Tables for one flux grid (default: fine) from a SeedResult (B_ref, I0, L_ref, eps0)."""
    grid = grid or cfg.grid_fine
    path = path or cfg.path
    cb = fr.make_basis(cfg.basis, cfg.nq_opt, cfg.order if cfg.basis == "fourier" else cfg.n_ctrl)
    sym = sym_mod.build(cfg.n_coils, boundary.nfp, cfg.stellsym)
    flux = sp.build(boundary, grid[0], grid[1], "fundamental", path)
    dist = sp.build(boundary, cfg.grid_dist[0], cfg.grid_dist[1], "full", path)
    I0 = seed.current_sign * 1.0e6
    fine = sp.build(boundary, cfg.grid_fine[0], cfg.grid_fine[1], "fundamental", path)
    X0, _, w0 = sp.evaluate(fine, 0.0)
    B_ref = rs.B_ref_from_seed(jnp.asarray(seed.x), cb, sym, X0, w0, I0)
    return Tables(cb, sym, flux, dist, B_ref, jnp.asarray(I0), jnp.asarray(seed.L_seed),
                  jnp.asarray(seed.L_seed * (1.0 + cfg.beta_L)), limits_from(cfg), cfg)


def coil_disp(T: Tables):
    """dx -> largest coil-point displacement max_{c,k} |B0 dx_c| (sec. 9.4). Symmetry copies are
    rotations of the base coils, so the base coils give the maximum over all physical coils."""
    def disp(dx):
        g = jnp.einsum("qk,cdk->cqd", T.cb.B0, cur.geo(dx, T.cb, T.sym.C).reshape(T.sym.C, 3, T.cb.K))
        return jnp.max(jnp.linalg.norm(g, axis=-1))
    return disp


def guard_distance(T: Tables):
    """disp_guard * min(d_cc, d_cs) (sec. 9.4)."""
    return T.cfg.disp_guard * min(T.lim.dcc, T.lim.dcs)
