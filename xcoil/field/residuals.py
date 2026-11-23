"""Stacked least-squares residual r = (r_B, r_arc, r_L, r_msc) and its Jacobian.
f = sum r^2 = f_B + R, the objective is F = c_f f with c_f = 1/f_ref.

  r_B,q   = sqrt(w_q/W) (B(x_q) . n_q) / B_ref                     (N_s)
  r_arc,ck = sqrt(w_arc f_ref / (C N_q)) (|gamma'_ck| - lbar_c) / lbar_c   (C N_q)
  r_L     = sqrt(w_L(t)) L / L_ref                                  (1)
  r_msc,ck = sqrt(w_msc(t) / (C N_q)) kappa_ck / kappa_lim           (C N_q)

B_ref is frozen once: the area-weighted RMS of |B| on Sigma(0) from the seed coils. Constant
normalization keeps the Gauss-Newton matrix non-singular. The forward-only path reuses the
memory-lean kernel xcoil/field/biot_savart.py (merit evaluations); the analytic
path returns r_B and J_B together (xcoil/field/jacobian.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod
from xcoil.curves import currents as cur
from xcoil.constraints import pointwise as pw
from xcoil.homotopy import schedules as sch
from . import biot_savart as bs
from . import jacobian as jac


def surface_coeffs(w, B_ref):
    return jnp.sqrt(w / jnp.sum(w)) / B_ref


def field_on(x, cb, sym, X, I0):
    g, gd, _ = fr.coils_curves(x, cb, sym.C)
    return bs.field(X, sym_mod.expand(g, sym), sym_mod.expand(gd, sym), cur.phys_currents(x, cb, sym, I0))


def B_ref_from_seed(x_seed, cb, sym, X, w, I0):
    B = field_on(x_seed, cb, sym, X, I0)
    return jnp.sqrt(jnp.sum(w * jnp.sum(B * B, 1)) / jnp.sum(w))


def magnetic_residual(x, cb, sym, X, n, w, B_ref, I0):
    """Forward-only r_B (N_s,)."""
    B = field_on(x, cb, sym, X, I0)
    return surface_coeffs(w, B_ref) * jnp.sum(B * n, 1)


def _coil_reg(x_c, cb, lim_kappa):
    """Per-coil pieces: arc residual core (nq,), length (), curvature/kappa_lim (nq,)."""
    g, gd, gdd = fr.coil_curve(x_c, cb)
    sp_ = jnp.linalg.norm(gd, axis=-1)
    lbar = jnp.mean(sp_)
    return (sp_ - lbar) / lbar, lbar, pw.curvature(gd, gdd) / lim_kappa


def arc_intervals(nq, nint):
    """simsopt ArclengthVariation interval-averaging matrix (nint, nq): nint = nq ("full") or any
    number of intervals ("partial" = 2 * order); same floor(linspace) partition as simsopt."""
    import numpy as np
    idx = np.floor(np.linspace(0, nq, nint + 1, endpoint=True)).astype(int)
    M = np.zeros((nint, nq))
    for i in range(nint):
        M[i, idx[i]:idx[i + 1]] = 1.0 / (idx[i + 1] - idx[i])
    return jnp.asarray(M)


def _arc_nint(cfg, nq):
    n = int(cfg.arc_pen_intervals)
    return nq if n == 0 else (2 * cfg.order if n < 0 else n)


def _coil_arc_rho(v, cb, M):
    """rho_i = (l_i - mean l) / sqrt(N) with l = M |gamma'| the interval-averaged incremental
    arclength, so that sum_i rho_i^2 = Var(l) = simsopt ArclengthVariation."""
    l = M @ jnp.linalg.norm(fr.coil_curve(v, cb)[1], axis=-1)
    return (l - jnp.mean(l)) / jnp.sqrt(l.shape[0])


def _coil_arcvar(v, cb, M=None):
    """simsopt ArclengthVariation of one coil [m^2] (M: interval matrix; None = "full")."""
    sp = jnp.linalg.norm(fr.coil_curve(v, cb)[1], axis=-1)
    return jnp.var(sp if M is None else M @ sp)


def _arc_pen_rows(xc, cb, cfg):
    """Arclength-variation penalty residuals, one per coil: sqrt(w) max(0, Var_c - T), so that the
    objective gains w sum_c max(0, Var_c - T)^2 (simsopt QuadraticPenalty(ArclengthVariation, T,
    "max")) in the units of the normalized squared flux. Empty when cfg.arc_pen_thresh == 0."""
    if not cfg.arc_pen_thresh:
        return jnp.zeros((0,), xc.dtype)
    M = arc_intervals(cb.B0.shape[0], _arc_nint(cfg, cb.B0.shape[0]))
    v = jax.vmap(lambda c: _coil_arcvar(c, cb, M))(xc)
    return jnp.sqrt(cfg.arc_pen_weight) * jnp.maximum(0.0, v - cfg.arc_pen_thresh)


def regularizer_residual(x, t, cb, C, cfg, L_ref):
    xc = cur.geo(x, cb, C).reshape(C, -1)
    arc, lens, kap = jax.vmap(lambda v: _coil_reg(v, cb, cfg.kappa_lim))(xc)
    nq = arc.shape[1]
    return jnp.concatenate([
        sch.amp_arc(cfg) / jnp.sqrt(C * nq) * arc.ravel(),
        jnp.reshape(sch.amp_L(t, cfg) * jnp.sum(lens) / L_ref, (1,)),
        sch.amp_msc(t, cfg) / jnp.sqrt(C * nq) * kap.ravel(),
        _arc_pen_rows(xc, cb, cfg)])


def regularizer_jacobian(x, t, cb, C, cfg, L_ref):
    """Block structure: arc and msc rows of coil c depend only on x_c; the length row on all.
    Per-coil jacfwd, vmapped over coils, assembled into the dense (n_reg, D) matrix (zero columns
    for the currents)."""
    xc = cur.geo(x, cb, C).reshape(C, -1); Dc = xc.shape[1]
    Ja, Jl, Jk = jax.vmap(jax.jacfwd(lambda v: _coil_reg(v, cb, cfg.kappa_lim)))(xc)
    nq = Ja.shape[1]
    eye = jnp.eye(C, dtype=x.dtype)
    blk = lambda Jb: jnp.einsum("cqd,ce->cqed", Jb, eye).reshape(C * nq, C * Dc)
    rows = [sch.amp_arc(cfg) / jnp.sqrt(C * nq) * blk(Ja),
            (sch.amp_L(t, cfg) / L_ref * Jl.reshape(1, C * Dc)),
            sch.amp_msc(t, cfg) / jnp.sqrt(C * nq) * blk(Jk)]
    if cfg.arc_pen_thresh:                           # d/dx sqrt(w) max(0, Var_c - T), block diagonal
        M = arc_intervals(nq, _arc_nint(cfg, nq))
        v = jax.vmap(lambda c: _coil_arcvar(c, cb, M))(xc)
        dv = jax.vmap(jax.grad(lambda c: _coil_arcvar(c, cb, M)))(xc)              # (C, Dc)
        act = (v > cfg.arc_pen_thresh).astype(x.dtype)
        rows.append(jnp.sqrt(cfg.arc_pen_weight) * jnp.einsum("c,cd,ce->ced", act, dv, eye).reshape(C, C * Dc))
    return cur.pad_cols(jnp.concatenate(rows), x, cb, C)


def stacked(x, t, cb, sym, surf, B_ref, I0, cfg, L_ref):
    """(r, J) with r = (r_B, r_arc, r_L, r_msc); surf = (X, n, w) of Sigma(t)."""
    X, n, w = surf
    rB, JB = jac.residual_and_jacobian(x, cb, sym, X, n, surface_coeffs(w, B_ref), I0)
    rR = regularizer_residual(x, t, cb, sym.C, cfg, L_ref)
    JR = regularizer_jacobian(x, t, cb, sym.C, cfg, L_ref)
    return jnp.concatenate([rB, rR]), jnp.concatenate([JB, JR])


def stacked_residual(x, t, cb, sym, surf, B_ref, I0, cfg, L_ref):
    """Forward-only stacked residual (merit evaluations, t-jvp for the predictor)."""
    X, n, w = surf
    return jnp.concatenate([magnetic_residual(x, cb, sym, X, n, w, B_ref, I0),
                            regularizer_residual(x, t, cb, sym.C, cfg, L_ref)])


def arc_penalty_curvature(x, cb, C, cfg, c_f):
    """Second-order term of the arclength penalty missing from the Gauss-Newton model:
    phi(V) = c_f w max(0, V - T)^2 with V = |rho|^2 gives Hess = phi'' dV dV^T + phi' d2V; the first
    part is already in J^T J (the residual row sqrt(w) max(0, V - T)), this returns the second with
    d2V replaced by its Gauss-Newton form 2 R^T R (R = d rho / dx_c, PSD, full rank on the
    reparametrization directions): 4 c_f w max(0, V_c - T) R_c^T R_c, block diagonal (n, n)."""
    xc = cur.geo(x, cb, C).reshape(C, -1); nq = cb.B0.shape[0]
    M = arc_intervals(nq, _arc_nint(cfg, nq))
    rho = jax.vmap(lambda c: _coil_arc_rho(c, cb, M))(xc)                       # (C, N)
    R = jax.vmap(jax.jacfwd(lambda c: _coil_arc_rho(c, cb, M)))(xc)             # (C, N, Dc)
    V = jnp.sum(rho ** 2, axis=1)
    coef = 4.0 * c_f * cfg.arc_pen_weight * jnp.maximum(0.0, V - cfg.arc_pen_thresh)
    blocks = coef[:, None, None] * jnp.einsum("cnd,cne->cde", R, R)
    return cur.pad_square(jax.scipy.linalg.block_diag(*[blocks[c] for c in range(C)]), x, cb, C)
