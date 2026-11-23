"""Analytic magnetic residual Jacobian.

    r_q = c_q n_q . B(x_q),  c_q = sqrt(w_q / W) / B_ref,
    B(x_q) = (mu0/4pi)(1/N_q) sum_j I_j sum_k gamma'_jk x Delta_qjk / rho^3,  Delta = x_q - gamma_jk.

With a_qj = c_q (mu0/4pi) I_j / N_q and u = n_q . (gamma' x Delta):
    dr_q/dgamma'_jk = a_qj (Delta x n_q) / rho^3
    dr_q/dgamma_jk  = a_qj ( -(n_q x gamma'_jk)/rho^3 + 3 u Delta / rho^5 ).
Copies j of base coil c share the basis (gamma_j = R_j Phi x_c), so the partials are rotated back
with R_j^T and summed over copies before one contraction with (B0, B1):
    dr/dx_c[d, :] = sum_k Ghat^g_c(q,k)[d] B0[k,:] + Ghat^g'_c(q,k)[d] B1[k,:].
A lax.scan over the C base coils bounds memory to one coil's (N_s, copies, N_q, 3) block; the
residual itself is accumulated in the same pass.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from xcoil.curves import currents as cur

MU0_4PI = 1e-7


def copy_tables(sym):
    """Per base coil: rotations (C, ncopy, 3, 3) and current signs (C, ncopy) (simsopt order
    j = r*C + c, so the copies of coil c are j = c, c + C, c + 2C, ...)."""
    C = sym.C; ncopy = sym.n_phys // C
    return (sym.R.reshape(ncopy, C, 3, 3).transpose(1, 0, 2, 3),
            sym.sign.reshape(ncopy, C).T)


def residual_and_jacobian(x, cb, sym, X, n, cq, I0):
    """r_B (Ns,) and J_B (Ns, n) for surface points X, unit normals n, coefficients cq; n = C*3K,
    plus C-1 current columns when x carries currents (curves/currents): r is linear in the base
    currents, r = sum_c I_c r1_c with r1_c the unit-current residual of coil c (all its copies), so
    dr/du = [r1_c] dI/du and the coefficient block of coil c is scaled by I_c."""
    C, K = sym.C, cb.K
    Xc = cur.geo(x, cb, C).reshape(C, 3, K)
    g = jnp.einsum("qk,cdk->cqd", cb.B0, Xc)
    gd = jnp.einsum("qk,cdk->cqd", cb.B1, Xc)
    Rc, sc = copy_tables(sym)
    Ib = cur.base_currents(x, cb, C, I0)
    nq = g.shape[1]; Ns = X.shape[0]

    def per_coil(r_acc, inp):
        gcc, gdc, Rj, sj, Ic = inp
        Gj = jnp.einsum("jab,kb->jka", Rj, gcc)                 # (ncopy, nq, 3)
        Gdj = jnp.einsum("jab,kb->jka", Rj, gdc)
        D = X[:, None, None, :] - Gj[None]                       # (Ns, ncopy, nq, 3)
        rho2 = jnp.sum(D * D, axis=-1)
        inv3 = rho2 ** -1.5
        inv5 = inv3 / rho2
        nb = n[:, None, None, :]
        u = jnp.sum(nb * jnp.cross(Gdj[None], D), axis=-1)       # n . (gamma' x Delta)
        a = (cq * (MU0_4PI / nq))[:, None] * sj[None, :]         # (Ns, ncopy), unit current
        r1 = jnp.einsum("sj,sjk->s", a, u * inv3)
        r_acc = r_acc + Ic * r1
        dgp = a[:, :, None, None] * jnp.cross(D, nb) * inv3[..., None]
        dg = a[:, :, None, None] * (-jnp.cross(nb, Gdj[None]) * inv3[..., None]
                                    + 3.0 * (u * inv5)[..., None] * D)
        Gh = jnp.einsum("jba,sjkb->ska", Rj, dg)                  # R_j^T fold, sum over copies
        Ghp = jnp.einsum("jba,sjkb->ska", Rj, dgp)
        Jc = jnp.einsum("skd,km->sdm", Gh, cb.B0) + jnp.einsum("skd,km->sdm", Ghp, cb.B1)
        return r_acc, (Ic * Jc.reshape(Ns, 3 * K), r1)

    r, (Js, R1) = jax.lax.scan(per_coil, jnp.zeros(Ns, X.dtype), (g, gd, Rc, sc, Ib))
    J = jnp.transpose(Js, (1, 0, 2)).reshape(Ns, C * 3 * K)
    if cur.has_u(x, cb, C):
        J = jnp.concatenate([J, R1.T @ cur.dI_du(C, I0, J.dtype)], 1)
    return r, J


def residual_hessian_blocks(x, cb, sym, X, n, cq, I0, rbar):
    """Second-order term of the Gauss-Newton gap, S_c = sum_q rbar_q d^2 r_q / dx_c^2, as C dense
    (3K, 3K) blocks. It is exactly block-diagonal over base coils (B is a sum over coils) and,
    per coil, block-diagonal over quadrature points in (gamma_k, gamma'_k):
      d2T/dgamma'^2        = 0
      d2T/dgamma'_a dgamma_e = a ( -eps_aec n_c / rho^3 + 3 (Delta x n)_a Delta_e / rho^5 )
      d2T/dgamma^2         = a ( -3 [(n x gamma') Delta^T + Delta (n x gamma')^T] / rho^5
                                 - 3 u I / rho^5 + 15 u Delta Delta^T / rho^7 )
    for T = a u / rho^3 (notation of the module docstring), weighted by rbar_q, rotated back with
    R_j^T, summed over copies and surface points, then contracted with (B0, B1). With free currents
    a carries the coil current I_c; the coefficient-current cross terms are not included (the
    residual is linear in the currents, so the current-current block is zero)."""
    C, K = sym.C, cb.K
    Xc = cur.geo(x, cb, C).reshape(C, 3, K)
    Ib = cur.base_currents(x, cb, C, I0)
    g = jnp.einsum("qk,cdk->cqd", cb.B0, Xc)
    gd = jnp.einsum("qk,cdk->cqd", cb.B1, Xc)
    Rc, sc = copy_tables(sym)
    nq = g.shape[1]
    eps3 = jnp.zeros((3, 3, 3), x.dtype)
    for (i, j, k), v in (((0, 1, 2), 1.0), ((1, 2, 0), 1.0), ((2, 0, 1), 1.0),
                         ((0, 2, 1), -1.0), ((2, 1, 0), -1.0), ((1, 0, 2), -1.0)):
        eps3 = eps3.at[i, j, k].set(v)

    def per_coil(_, inp):
        gcc, gdc, Rj, sj, Ic = inp
        Gj = jnp.einsum("jab,kb->jka", Rj, gcc)
        Gdj = jnp.einsum("jab,kb->jka", Rj, gdc)
        D = X[:, None, None, :] - Gj[None]                       # (Ns, ncopy, nq, 3)
        rho2 = jnp.sum(D * D, axis=-1)
        inv3 = rho2 ** -1.5; inv5 = inv3 / rho2; inv7 = inv5 / rho2
        nb = n[:, None, None, :]
        u = jnp.sum(nb * jnp.cross(Gdj[None], D), axis=-1)
        a = ((rbar * cq) * (MU0_4PI * Ic / nq))[:, None] * sj[None, :]   # rbar-weighted
        a3, a5, a7 = a[..., None] * inv3, a[..., None] * inv5, a[..., None] * inv7
        # gamma'-gamma block (rows gamma', cols gamma), summed over surface points
        Hpg = (-jnp.einsum("aec,sjkc,sjk->jkae", eps3, jnp.broadcast_to(nb, D.shape), a3)
               + 3.0 * jnp.einsum("sjka,sjke,sjk->jkae", jnp.cross(D, nb), D, a5))
        nxg = jnp.cross(jnp.broadcast_to(nb, D.shape), Gdj[None])
        Hgg = (-3.0 * (jnp.einsum("sjka,sjke,sjk->jkae", nxg, D, a5)
                       + jnp.einsum("sjka,sjke,sjk->jkae", D, nxg, a5))
               - 3.0 * jnp.einsum("sjk,ae->jkae", u * a5, jnp.eye(3, dtype=x.dtype))
               + 15.0 * jnp.einsum("sjka,sjke,sjk->jkae", D, D, u * a7))
        # rotate back to the base-coil frame and sum copies: R^T H R
        Hpg = jnp.einsum("jba,jkbc,jcd->kad", Rj, Hpg, Rj)
        Hgg = jnp.einsum("jba,jkbc,jcd->kad", Rj, Hgg, Rj)
        # contract with the basis: x_c index (d, m) -> d*K + m
        Sgg = jnp.einsum("km,kde,kl->dmel", cb.B0, Hgg, cb.B0)
        Spg = jnp.einsum("km,kde,kl->dmel", cb.B1, Hpg, cb.B0)   # rows gamma', cols gamma
        S = Sgg + Spg + jnp.transpose(Spg, (2, 3, 0, 1))
        return None, S.reshape(3 * K, 3 * K)

    _, S = jax.lax.scan(per_coil, None, (g, gd, Rc, sc, Ib))
    return S
