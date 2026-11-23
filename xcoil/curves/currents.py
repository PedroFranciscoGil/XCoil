"""Free coil currents: the design vector is x = (coil coefficients, u),
coefficients (C * 3K,) in CurveXYZFourier order followed by u (C - 1,), with base-coil currents

    I_c = I0 (1 + (P u)_c),   P = [I_{C-1}; -1^T]   (C, C - 1),

so sum_c I_c = C I0: the total current of the base coils (one half field period with stellarator
symmetry, one field period without) is fixed and the individual currents are free and unbounded.
u = 0 gives equal currents. A vector without u (size C * 3K) is evaluated with equal currents, so
equal-current designs stay valid inputs everywhere. Every shape test here is static under jit.
"""
from __future__ import annotations

import jax.numpy as jnp


def n_geo(cb, C):
    return C * 3 * cb.K


def n_cur(C, free=True):
    return C - 1 if free else 0


def geo(x, cb, C):
    """Coil-coefficient part of x."""
    return x[:n_geo(cb, C)]


def has_u(x, cb, C):
    return x.shape[0] > n_geo(cb, C)


def basis(C, dtype=jnp.float64):
    """P (C, C - 1): u -> current perturbation with zero sum."""
    return jnp.concatenate([jnp.eye(C - 1, dtype=dtype), -jnp.ones((1, C - 1), dtype)], 0)


def base_currents(x, cb, C, I0):
    """(C,) base-coil currents of x."""
    if not has_u(x, cb, C):
        return I0 * jnp.ones(C, x.dtype)
    u = x[n_geo(cb, C):]
    return I0 * (1.0 + basis(C, x.dtype) @ u)


def phys_currents(x, cb, sym, I0):
    """(Nphys,) signed currents of the physical coils (symmetry copies, sign -1 for flipped)."""
    return base_currents(x, cb, sym.C, I0)[sym.coil] * sym.sign


def dI_du(C, I0, dtype=jnp.float64):
    """d I / d u (C, C - 1)."""
    return I0 * basis(C, dtype)


def pad_cols(J, x, cb, C):
    """Append zero columns for u to a Jacobian of a quantity that does not depend on the currents."""
    nu = x.shape[0] - n_geo(cb, C)
    if nu == 0:
        return J
    return jnp.concatenate([J, jnp.zeros(J.shape[:-1] + (nu,), J.dtype)], -1)


def pad_square(H, x, cb, C):
    """Embed a (n_geo, n_geo) matrix in (n, n) with a zero current block."""
    nu = x.shape[0] - n_geo(cb, C)
    if nu == 0:
        return H
    ng = H.shape[0]
    return jnp.zeros((ng + nu, ng + nu), H.dtype).at[:ng, :ng].set(H)


def with_currents(x, cb, C, free=True):
    """x with u = 0 appended (free currents) if it has none; unchanged otherwise."""
    import numpy as np
    x = np.asarray(x, float)
    if not free or x.size > n_geo(cb, C):
        return x
    return np.concatenate([x, np.zeros(C - 1)])
