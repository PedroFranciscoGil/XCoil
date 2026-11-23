"""Symmetry expansion of base coils, matching simsopt coils_via_symmetries.

Physical coil j = (k, flip, i) in simsopt order (for k: for flip: for i). In column form
    gamma_j = R_j gamma_i,   R_j = D Rz(2 pi k / nfp) if flip else Rz(2 pi k / nfp),
with D = diag(1, -1, -1) (simsopt RotatedCurve: gamma @ Rz^T @ D). Flipped copies carry current
sign -1 instead of a reversed parameterization. Tangent vectors transform with the same R_j.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

D_FLIP = np.diag([1.0, -1.0, -1.0])


def rot_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class Symmetry:
    R: jnp.ndarray        # (Nphys, 3, 3)
    coil: jnp.ndarray     # (Nphys,) base-coil index
    sign: jnp.ndarray     # (Nphys,) current sign
    ident: jnp.ndarray    # (Nphys,) True for the identity copy of its base coil
    nfp: int
    stellsym: bool
    C: int

    @property
    def n_phys(self):
        return int(self.R.shape[0])


# As a pytree: the rotations and current signs are data (so precision casts of the tables reach
# them); the index arrays (base coil of each copy, identity copies) are structure - they depend only
# on (nfp, stellsym, C), code uses them concretely while tracing (masks), and devices of the same
# structure share compiled programs.
jax.tree_util.register_pytree_node(
    Symmetry,
    lambda s: ((s.R, s.sign), (tuple(np.asarray(s.coil).tolist()), tuple(np.asarray(s.ident).tolist()),
                               s.nfp, s.stellsym, s.C)),
    lambda aux, ch: Symmetry(ch[0], np.asarray(aux[0]), ch[1], np.asarray(aux[1], bool), *aux[2:]))


def build(C: int, nfp: int, stellsym: bool) -> Symmetry:
    Rs, coil, sign, ident = [], [], [], []
    for k in range(nfp):
        for flip in ([False, True] if stellsym else [False]):
            for i in range(C):
                Rk = rot_z(2 * np.pi * k / nfp)
                Rs.append(D_FLIP @ Rk if flip else Rk)
                coil.append(i); sign.append(-1.0 if flip else 1.0)
                ident.append(k == 0 and not flip)
    return Symmetry(jnp.asarray(np.stack(Rs)), jnp.asarray(np.array(coil)),
                    jnp.asarray(np.array(sign)), jnp.asarray(np.array(ident)), nfp, stellsym, C)


def expand(pts, sym: Symmetry):
    """pts: (C, nq, 3) base-coil points or tangents -> (Nphys, nq, 3)."""
    return jnp.einsum("jab,jkb->jka", sym.R, pts[sym.coil])


def base_toroidal_angles(C: int, nfp: int, stellsym: bool):
    """Seed planes phi_c (sec. 5.3): 2C evenly spaced coils per field period with symmetry."""
    span = np.pi / (nfp * C) if stellsym else 2 * np.pi / (nfp * C)
    return (np.arange(C) + 0.5) * span
