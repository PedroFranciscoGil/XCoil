"""Coil forces per unit length and the opt-in maximum-force constraint.

Physics (simsopt's coil_force): on base coil c with current I_c and unit tangent t,

    |dF/dl| = |I_c t x (B_mutual + B_self,reg)|

at every quadrature point, where B_mutual is the Biot-Savart field of every other physical coil (all field periods and
stellarator images; the base coil's own identity copy left out) and B_self,reg the regularized self-field of a circular
conductor of radius a (delta = a^2 / sqrt(e); Landreman, Hurwitz & Antonsen). FP64, pure JAX: differentiable in the
curve coefficients and the currents.

The constraint: one row per base coil and quadrature point,

    g = F / F_lim - (1 - margin) <= 0,

added to the solver as extra constraint rows (`force_extra`, for fixed.device_lm.build_solver(extra=...)); the
certificate is `max_force` (the maximum per base coil on a given quadrature) compared with F_lim.

Limits given per turn at another scale: |dF/dl| scales as I^2 / L at fixed shape and a / L, so a limit F_turn [N/m per
turn] at length scale S with total current I_ref per half period is, in a problem with total current I_norm,

    F_lim = F_turn * turns * S / (I_ref / I_norm)^2,      with the conductor radius a / S      (normalized_limit).
"""
from __future__ import annotations

import math

import numpy as np
import jax
import jax.numpy as jnp

from xcoil.curves import currents as cur
from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod
from xcoil.field.jacobian import MU0_4PI


def regularization_circ(a):
    """delta of a circular conductor of radius a: a^2 / sqrt(e)."""
    return a * a / math.sqrt(math.e)


def normalized_limit(F_turn, turns, S, I_ref, I_norm):
    """Per-coil |dF/dl| limit [N/m] in the problem's units from a per-turn limit at scale S and total current I_ref."""
    return float(F_turn) * float(turns) * float(S) / (float(I_ref) / float(I_norm)) ** 2


def self_field(g, g1, g2, current, delta):
    """(nq, 3) regularized self-field [T] on one coil (simsopt B_regularized_pure); g1, g2 derivatives in s in [0, 1)."""
    n = g.shape[0]
    phi = 2 * jnp.pi * jnp.arange(n) / n
    rp, rpp = g1 / (2 * jnp.pi), g2 / (4 * jnp.pi ** 2)
    nrp = jnp.linalg.norm(rp, axis=1)
    analytic = jnp.cross(rp, rpp) * (0.5 * (-2 + jnp.log(64 * nrp * nrp / delta)) / nrp ** 3)[:, None]
    dr = g[:, None] - g[None, :]
    first = jnp.cross(rp[None, :], dr) / ((jnp.sum(dr * dr, 2) + delta) ** 1.5)[:, :, None]
    cf = 2.0 - 2.0 * jnp.cos(phi[None, :] - phi[:, None])
    second = (jnp.cross(rpp, rp)[:, None, :]
              * (0.5 * cf / (cf * jnp.sum(rp * rp, 1)[:, None] + delta) ** 1.5)[:, :, None])
    return current * MU0_4PI * (analytic + (2 * jnp.pi / n) * jnp.sum(first + second, 1))


def coil_forces(xc, I, cb, sym, delta):
    """(C, nq) |dF/dl| [N/m] on the base coils: base-curve coefficients xc (C * 3K,) on the basis cb, base-coil currents
    I (C,) [A], symmetry tables sym, self-field regularization delta."""
    C = sym.C
    g, g1, g2 = fr.coils_curves(xc, cb, C)
    G, G1 = sym_mod.expand(g, sym), sym_mod.expand(g1, sym)
    Iphys = I[sym.coil] * sym.sign
    ident = np.asarray(sym.ident)
    coil = np.asarray(sym.coil)
    self_index = np.array([int(np.where(ident & (coil == c))[0][0]) for c in range(C)])
    W = Iphys[None, :] * jnp.asarray(np.arange(G.shape[0])[None, :] != self_index[:, None])     # (C, J)
    d = g[:, :, None, None, :] - G[None, None]                                                  # (C, nq, J, nqs, 3)
    r2 = jnp.sum(d * d, -1)
    r2 = jnp.where(r2 > 0, r2, 1.0)
    k = jnp.cross(G1[None, None], d) / (r2 ** 1.5)[..., None]
    Bm = MU0_4PI * jnp.einsum("cj,cpjsk->cpk", W, k) / G.shape[1]
    Bs = jax.vmap(self_field, (0, 0, 0, 0, None))(g, g1, g2, I, delta)
    t = g1 / jnp.linalg.norm(g1, axis=-1, keepdims=True)
    return jnp.linalg.norm(I[:, None, None] * jnp.cross(t, Bm + Bs), axis=-1)


def forces_of_x(x, cb, sym, I0, delta):
    """(C, nq) |dF/dl| of a design vector x (coefficients, then the free-current variables u if present)."""
    return coil_forces(cur.geo(x, cb, sym.C), cur.base_currents(x, cb, sym.C, I0), cb, sym, delta)


def force_rows(x, cb, sym, I0, F_lim, delta, margin=0.0):
    """(C * nq,) constraint rows F / F_lim - (1 - margin) <= 0."""
    return (forces_of_x(x, cb, sym, I0, delta) / F_lim - (1.0 - margin)).ravel()


def max_force(x, cb, sym, I0, delta):
    """(C,) maximum |dF/dl| per base coil [N/m] on the basis cb (certificate: pass a fine quadrature basis)."""
    return jnp.max(forces_of_x(x, cb, sym, I0, delta), axis=1)


def basis_like(cb, nq):
    """The curve basis of cb (same kind and size) at nq quadrature points."""
    order = (cb.K - 1) // 2 if cb.kind == "fourier" else cb.K
    return fr.make_basis(cb.kind, int(nq), order)


def force_extra(prob, F_lim, radius, margin=0.0, nq=None):
    """Extra constraint rows for fixed.device_lm.build_solver(prob, extra=...): the maximum-force constraint
    F / F_lim - (1 - margin) <= 0 on every base coil at nq quadrature points (default: the problem's own), conductor
    radius `radius` (same length unit as the problem). No residual rows; the flux rows keep their weight."""
    T = prob.T
    cb = T.cb if nq is None else basis_like(T.cb, nq)
    sym, I0, delta = T.sym, T.I0, regularization_circ(float(radius))
    from xcoil.fixed.device_lm import Extra
    g = lambda x: force_rows(x, cb, sym, I0, float(F_lim), delta, float(margin))
    r = lambda x: jnp.zeros((0,), x.dtype)
    # forward mode: C * nq rows but only C * 3K + C - 1 columns (reverse mode would be ~nq / 3K times slower)
    return Extra(r, g, jr=lambda x: jnp.zeros((0, x.shape[0]), x.dtype), jg=jax.jacfwd(g))


def certify_force(x, prob, F_lim, radius, nq=768):
    """Certificate of the force limit on an nq-point quadrature: dict(max_force (C,) [N/m], F_lim, violation
    max(0, max F / F_lim - 1), ok: violation <= 1e-3, the certificate's tolerance)."""
    T = prob.T
    cb = basis_like(T.cb, nq)
    F = np.asarray(max_force(jnp.asarray(x, jnp.float64), cb, T.sym, T.I0, regularization_circ(float(radius))))
    v = max(0.0, float(np.max(F)) / float(F_lim) - 1.0)
    return dict(max_force=F.tolist(), F_lim=float(F_lim), violation=v, ok=v <= 1e-3)   # viol_tol of the certificate
