"""Batched filament Biot-Savart kernel.

For a filament coil sampled at Q quadrature points with position gamma(s) and dl = gamma'(s) ds
(uniform s in [0,1), ds = 1/Q), the field at a set of evaluation points is

    B(x) = (mu0 / 4pi) sum_coils I_c (1/Q) sum_q gamma'_{c,q} x (x - gamma_{c,q}) / |x - gamma_{c,q}|^3.

The conceptual interaction tensor is (candidate, coil, coil_quadrature, surface_point, xyz); we
never materialize it fully. A lax.scan over coils bounds peak memory to one coil's (S, Q, 3)
block. Candidate batching is a vmap over the leading axis.

Tested against an independent reference to < 1e-8 relative in FP64 at points off the filaments:
(a) the closed-form on-axis circular-loop field and (b) a naive numpy einsum Biot-Savart
(tests/test_biot_savart.py).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

MU0_4PI = 1e-7  # mu0 / (4 pi)


def field(points, gamma, gamma_dash, currents):
    """B at points (S, 3) from coils gamma (C, Q, 3), gamma_dash (C, Q, 3), currents (C,)."""
    Q = gamma.shape[1]

    def coil_step(B, coil):
        g, gd, I = coil                                  # (Q,3), (Q,3), ()
        r = points[:, None, :] - g[None, :, :]           # (S, Q, 3)
        inv_r3 = jnp.sum(r * r, axis=-1) ** (-1.5)       # (S, Q)
        cross = jnp.cross(gd[None, :, :], r)             # (S, Q, 3)
        return B + (I / Q) * jnp.sum(cross * inv_r3[..., None], axis=1), None

    B0 = jnp.zeros((points.shape[0], 3), dtype=gamma.dtype)
    B, _ = jax.lax.scan(coil_step, B0, (gamma, gamma_dash, currents))
    return MU0_4PI * B


def field_batched(points, gamma, gamma_dash, currents):
    """Batched over a leading candidate axis: gamma (P, C, Q, 3) -> B (P, S, 3)."""
    return jax.vmap(lambda g, gd, I: field(points, g, gd, I))(gamma, gamma_dash, currents)


def _rot_z(phi):
    c, s = jnp.cos(phi), jnp.sin(phi)
    return jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def expand_symmetry(gamma, gamma_dash, currents, nfp, stellsym):
    """Expand base coils to the full stellarator-symmetric set.

    Applies the nfp field-period rotations about z and, if stellsym, the stellarator reflection
    (z -> -z, y -> -y with a rotation) that flips current sign. gamma (C, Q, 3) -> (C_full, Q, 3).
    """
    flips = [False, True] if stellsym else [False]
    gs, gds, cur = [], [], []
    for k in range(nfp):
        Rk = _rot_z(2.0 * jnp.pi * k / nfp)
        for flip in flips:
            if flip:
                D = jnp.diag(jnp.array([1.0, -1.0, -1.0]))
                Rm = (Rk @ D).T
            else:
                Rm = Rk.T
            gs.append(gamma @ Rm)
            gds.append(gamma_dash @ Rm)
            cur.append(-currents if flip else currents)
    return jnp.concatenate(gs, 0), jnp.concatenate(gds, 0), jnp.concatenate(cur, 0)
