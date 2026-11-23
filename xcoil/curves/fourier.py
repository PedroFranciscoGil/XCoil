"""Linear Cartesian coil curves: gamma = Phi x, gamma' = Phi' x, gamma'' = Phi'' x.

Each Cartesian component of a base coil is expanded in a scalar basis of K functions of the
curve parameter s in [0, 1), evaluated at fixed points s_k. The 3 N_q x 3K matrix Phi is the
Kronecker product I_3 (x) B0, so kernels only ever use the N_q x K scalar matrices B0, B1, B2.

Fourier basis columns follow simsopt CurveXYZFourier exactly: [1, sin(2 pi s), cos(2 pi s),
sin(4 pi s), cos(4 pi s), ...], and a coil's DOF vector is (x-block, y-block, z-block), so it is
bit-identical to simsopt's dof vector (used for offline certification).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CurveBasis:
    B0: jnp.ndarray   # (nq, K)
    B1: jnp.ndarray   # d/ds
    B2: jnp.ndarray   # d2/ds2
    kind: str
    K: int


jax.tree_util.register_pytree_node(
    CurveBasis, lambda b: ((b.B0, b.B1, b.B2), (b.kind, b.K)),
    lambda aux, ch: CurveBasis(*ch, *aux))


def fourier_scalar_basis(s, M):
    s = np.asarray(s, float)
    cols0, cols1, cols2 = [np.ones_like(s)], [np.zeros_like(s)], [np.zeros_like(s)]
    for m in range(1, M + 1):
        w = 2 * np.pi * m
        sn, cs = np.sin(w * s), np.cos(w * s)
        cols0 += [sn, cs]
        cols1 += [w * cs, -w * sn]
        cols2 += [-w * w * sn, -w * w * cs]
    return np.stack(cols0, 1), np.stack(cols1, 1), np.stack(cols2, 1)


def uniform_s(nq):
    return np.arange(nq) / nq


def make_basis(kind: str, nq: int, order: int) -> CurveBasis:
    """kind 'fourier' (order = M, K = 2M+1) or 'bspline' (order = N control points, K = N)."""
    s = uniform_s(nq)
    if kind == "fourier":
        B0, B1, B2 = fourier_scalar_basis(s, order)
    elif kind == "bspline":
        from xcoil.geometry import periodic_bspline as pb
        B0, B1, B2, _ = [np.asarray(a) for a in pb.basis_matrices(jnp.asarray(s), order)]
    else:
        raise ValueError(kind)
    return CurveBasis(jnp.asarray(B0), jnp.asarray(B1), jnp.asarray(B2), kind, B0.shape[1])


def scalar_basis_at(kind, s, order):
    if kind == "fourier":
        return fourier_scalar_basis(s, order)
    from xcoil.geometry import periodic_bspline as pb
    return tuple(np.asarray(a) for a in pb.basis_matrices(jnp.asarray(s), order)[:3])


def coil_curve(x_c, cb: CurveBasis):
    """x_c: (3K,) -> gamma, gamma', gamma'' each (nq, 3)."""
    X = x_c.reshape(3, cb.K).T                     # (K, 3)
    return cb.B0 @ X, cb.B1 @ X, cb.B2 @ X


def coils_curves(x, cb: CurveBasis, C: int):
    """x: (C*3K [+ C-1 currents],) -> (C, nq, 3) gamma, gamma', gamma'' (currents ignored)."""
    X = x[:C * 3 * cb.K].reshape(C, 3, cb.K)
    f = lambda B: jnp.einsum("qk,cdk->cqd", B, X)
    return f(cb.B0), f(cb.B1), f(cb.B2)


def fit(points, s, kind, order):
    """Least-squares DOFs of one coil from points (n, 3) sampled at parameters s (n,)."""
    B0 = scalar_basis_at(kind, s, order)[0]
    X, *_ = np.linalg.lstsq(B0, np.asarray(points, float), rcond=None)   # (K, 3)
    return X.T.reshape(-1)


def to_simsopt(x_c, nq, M):
    """simsopt CurveXYZFourier with identical DOFs (offline verification/export)."""
    from simsopt.geo import CurveXYZFourier
    c = CurveXYZFourier(nq, M)
    c.x = np.asarray(x_c, float)
    return c
