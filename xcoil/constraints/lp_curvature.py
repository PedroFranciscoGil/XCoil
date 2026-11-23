"""Lp curvature (opt-in): a smooth integral limit on how far each coil's curvature exceeds a threshold.

Per base coil c, simsopt's LpCurveCurvature functional

    J_c = (1/p) ∫ max(κ - κ₀, 0)^p dl          (dl = |γ'| ds, s in [0, 1), uniform quadrature)

and the constraint row J_c / J_max - 1 <= 0. With κ₀ above zero only the parts of a coil curved more than κ₀ count,
so it limits sharp bends without a pointwise maximum (C rows instead of C·nq). The certificate evaluates J_c on a
fine quadrature (4096 points per coil).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from xcoil.constraints import pointwise as pw
from xcoil.curves import fourier as fr


def lp_curvature(gd, gdd, p, threshold):
    """Per-coil J_c = (1/p) mean_k max(κ_k - threshold, 0)^p |γ'_k| on a uniform quadrature (gd, gdd: (..., nq, 3)),
    the simsopt LpCurveCurvature functional."""
    sp = jnp.linalg.norm(gd, axis=-1)
    k = pw.curvature(gd, gdd)
    return jnp.mean(jnp.maximum(k - threshold, 0.0) ** p * sp, axis=-1) / p


def lp_of_x(x, cb, C, p, threshold):
    """(C,) J_c of a design vector x (coil coefficients first) on the basis cb."""
    _, gd, gdd = fr.coils_curves(x, cb, C)
    return lp_curvature(gd, gdd, p, threshold)


def lp_rows(x, cb, C, J_max, p=2.0, threshold=0.0):
    """(C,) constraint rows J_c / J_max - 1 <= 0."""
    return lp_of_x(x, cb, C, p, threshold) / J_max - 1.0


def lp_extra(prob, J_max, p=2.0, threshold=0.0, nq=None):
    """Extra constraint rows for fixed.device_lm.build_solver(prob, extra=...): J_c / J_max - 1 <= 0 for every base
    coil, on the problem's quadrature (or nq points). J_max in the units of J (1/m^(p-1) for κ in 1/m)."""
    from xcoil.constraints.forces import basis_like
    from xcoil.fixed.device_lm import Extra
    T = prob.T
    cb = T.cb if nq is None else basis_like(T.cb, nq)
    C = T.sym.C
    g = lambda x: lp_rows(x, cb, C, float(J_max), float(p), float(threshold))
    return Extra(lambda x: jnp.zeros((0,), x.dtype), g, jr=lambda x: jnp.zeros((0, x.shape[0]), x.dtype),
                 jg=jax.jacfwd(g))


def certify_lp(x, prob, J_max, p=2.0, threshold=0.0, nq=4096):
    """Certificate of the Lp-curvature limit on an nq-point quadrature: dict(lp_curvature (C,), J_max, violation
    max(0, max J / J_max - 1), ok)."""
    from xcoil.constraints.forces import basis_like
    T = prob.T
    J = np.asarray(lp_of_x(jnp.asarray(x, jnp.float64), basis_like(T.cb, nq), T.sym.C, float(p), float(threshold)))
    v = max(0.0, float(np.max(J)) / float(J_max) - 1.0)
    return dict(lp_curvature=J.tolist(), J_max=float(J_max), violation=v, ok=v <= 1e-3)
