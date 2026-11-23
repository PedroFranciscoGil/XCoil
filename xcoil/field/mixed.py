"""Mixed-precision Biot-Savart residual and Jacobian: FP64 geometry, low-precision kernel.

The accuracy of an all-FP32 evaluation is lost in one place: the separation Delta = x_q - gamma_jk
of coordinates of size ~R0 (metres) whose difference is much smaller, plus the FP32 evaluation of
the curves themselves. Here the curves gamma, gamma' and the separations Delta are formed in FP64
and only then rounded to the kernel dtype `kdt` (FP32); 1/rho^3, the cross products and the
per-coil partial sums run in `kdt`, and the sums over coils are accumulated in FP64. On the W7-X
flux residual this gives |r|^2 to ~1e-9 relative at 8x less time than FP64 on a consumer GPU
(standard mixed-precision N-body practice: high-precision relative coordinates, low-precision
kernel, high-precision accumulation).

Same maths and output layout as field/jacobian.py. Every evaluation is one fused reduction over
all coil points (no loop over coils, whose per-iteration overhead dominated on a GPU), chunked over
the surface points only beyond PAIR_BUDGET pairs. The Jacobian is formed in each base coil's frame:
all symmetry copies are proper rotations R_j, so with X' = R_j^T X and n' = R_j^T n the folded
derivative sum_j R_j^T dB_j/dgamma is a plain sum over copies, and no per-copy tensor is stored.
kdt=None evaluates the same expressions in the input precision (FP64).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from xcoil.curves import fourier as fr, symmetry as sym_mod, currents as cur
from .jacobian import copy_tables, MU0_4PI
from .residuals import surface_coeffs, regularizer_residual, regularizer_jacobian


PAIR_BUDGET = 1 << 25      # surface-point x coil-point pairs per fused evaluation (bounds memory
                           # should XLA materialize the pair tensors; W7-X residual and Jacobian: one)


def _k(a, kdt):
    """Round an FP64 array to the kernel dtype; kdt=None keeps the input precision."""
    return a if kdt is None else a.astype(kdt)


def _over_points(fn, per_point, pairs_per_point):
    """fn(*chunk) over chunks of the per-point arrays (leading axis = surface points), at most
    PAIR_BUDGET pairs per chunk; a single fused evaluation when everything fits."""
    N = per_point[0].shape[0]
    chunk = max(1, min(N, PAIR_BUDGET // max(1, int(pairs_per_point))))
    if chunk >= N:
        return fn(*per_point)
    nc = -(-N // chunk); pad = nc * chunk - N
    split = lambda A: jnp.concatenate([A, jnp.broadcast_to(A[:1], (pad,) + A.shape[1:])]).reshape(
        (nc, chunk) + A.shape[1:])
    out = jax.lax.map(lambda c: fn(*c), tuple(split(A) for A in per_point))
    return jax.tree_util.tree_map(lambda o: o.reshape((nc * chunk,) + o.shape[2:])[:N], out)


def _all_coil_points(x, cb, sym, I0, kdt):
    """All physical coil points (J Q, 3) FP64, tangents (J Q, 3) in kdt, weights I_j / Q (J Q,) FP64."""
    g, gd, _ = fr.coils_curves(x, cb, sym.C)                       # FP64 curves
    Q = g.shape[1]
    G = sym_mod.expand(g, sym).reshape(-1, 3)
    Gd = _k(sym_mod.expand(gd, sym).reshape(-1, 3), kdt)
    wq = jnp.repeat(cur.phys_currents(x, cb, sym, I0) / Q, Q)
    return G, Gd, wq


# Kernels are written component-wise: x, y, z as separate arrays, explicit cross products and rsqrt,
# so that each output is a single reduction over the pairs of elementwise terms and XLA keeps the
# pair terms in registers (with (..., 3) arrays and jnp.cross it wrote them to memory on a GPU:
# 2.6x slower field, 1.7-3x slower Jacobian).

def _xyz(A):
    return A[..., 0], A[..., 1], A[..., 2]


def _inv13(r2):
    """1 / rho and 1 / rho^3 from rho^2 (rsqrt: one fast instruction, unlike pow(-1.5))."""
    ir = jax.lax.rsqrt(r2)
    return ir, ir * ir * ir


def magnetic_residual(x, cb, sym, X, n, w, B_ref, I0, kdt=jnp.float32):
    """Forward r_B (N_s,), FP64 in / out: FP64 separations rounded to kdt, kdt kernel, FP64 sum."""
    G, Gd, wq = _all_coil_points(x, cb, sym, I0, kdt)
    Gx, Gy, Gz = (c[None] for c in _xyz(G))
    gx, gy, gz = (c[None] for c in _xyz(Gd))

    def bn(Xb, nb):
        Xx, Xy, Xz = (c[:, None] for c in _xyz(Xb))
        nx, ny, nz = (_k(c[:, None], kdt) for c in _xyz(nb))
        dx, dy, dz = _k(Xx - Gx, kdt), _k(Xy - Gy, kdt), _k(Xz - Gz, kdt)
        _, inv3 = _inv13(dx * dx + dy * dy + dz * dz)
        u = nx * (gy * dz - gz * dy) + ny * (gz * dx - gx * dz) + nz * (gx * dy - gy * dx)
        return jnp.sum((u * inv3).astype(x.dtype) * wq[None], 1)

    Bn = _over_points(bn, (X, n), G.shape[0])
    return surface_coeffs(w, B_ref) * MU0_4PI * Bn


def field_on(x, cb, sym, X, I0, kdt=jnp.float32):
    """B (N, 3) at points X, FP64 in / out: FP64 curves and separations, kdt kernel, FP64 sum."""
    G, Gd, wq = _all_coil_points(x, cb, sym, I0, kdt)
    Gx, Gy, Gz = (c[None] for c in _xyz(G))
    gx, gy, gz = (c[None] for c in _xyz(Gd))
    f64 = x.dtype

    def bf(Xb):
        Xx, Xy, Xz = (c[:, None] for c in _xyz(Xb))
        dx, dy, dz = _k(Xx - Gx, kdt), _k(Xy - Gy, kdt), _k(Xz - Gz, kdt)
        _, inv3 = _inv13(dx * dx + dy * dy + dz * dz)
        acc = lambda c: jnp.sum((c * inv3).astype(f64) * wq[None], 1)
        return jnp.stack([acc(gy * dz - gz * dy), acc(gz * dx - gx * dz), acc(gx * dy - gy * dx)], -1)

    return MU0_4PI * _over_points(bf, (X,), G.shape[0])


def residual_and_jacobian(x, cb, sym, X, n, cq, I0, kdt=jnp.float32, j_minor=True):
    """r_B (N_s,) and J_B (N_s, n), FP64 out; kernel in kdt (see module doc). Current columns and
    per-coil current scaling as in field/jacobian.residual_and_jacobian.
    Copy j of base coil c is gamma_cj = R_cj gamma_c with R_cj a proper rotation, so with
    X' = R^T X, n' = R^T n and D' = X' - gamma_c (= R^T D): u = n' . (gamma_c' x D'),
    R^T dr/dgamma = a (-(n' x gamma_c') / rho^3 + 3 u D' / rho^5) and
    R^T dr/dgamma' = a (D' x n') / rho^3, summed over the copies (field/jacobian.py has the
    unrotated form). j_minor: pair terms arranged (s, c, k, j), the sum over copies over the minor
    axis (8 % faster on a GPU than (s, c, j, k))."""
    C, K = sym.C, cb.K
    f64 = x.dtype
    Xc = cur.geo(x, cb, C).reshape(C, 3, K)
    Ib = cur.base_currents(x, cb, C, I0)
    g = jnp.einsum("qk,cdk->cqd", cb.B0, Xc)                          # FP64 (C, nq, 3)
    gd = _k(jnp.einsum("qk,cdk->cqd", cb.B1, Xc), kdt)
    Rc, sc = copy_tables(sym)
    nq = g.shape[1]; Ns = X.shape[0]; ncopy = Rc.shape[1]
    Xp = jnp.einsum("cjba,sb->scja", Rc, X)                            # FP64 X' = R^T X (Ns, C, ncopy, 3)
    npk = _k(jnp.einsum("cjba,sb->scja", Rc, n), kdt)
    a = _k((cq * (MU0_4PI / nq))[:, None, None] * sc[None], kdt)       # (Ns, C, ncopy)
    if j_minor:      # pair terms (s, c, k, j): copies minor
        cp = lambda A: A[:, :, None, :]                                # (s, c, 1, j)
        cq_ = lambda A: A[None, :, :, None]                            # (1, c, k, 1)
        jax_ = -1
    else:            # pair terms (s, c, j, k)
        cp = lambda A: A[:, :, :, None]
        cq_ = lambda A: A[None, :, None, :]
        jax_ = 2
    gx, gy, gz = (cq_(c) for c in _xyz(g))
    hx, hy, hz = (cq_(c) for c in _xyz(gd))

    def blk(Xp, npk, a):
        Xx, Xy, Xz = (cp(c) for c in _xyz(Xp))
        nx, ny, nz = (cp(c) for c in _xyz(npk))
        aa = cp(a)
        dx, dy, dz = _k(Xx - gx, kdt), _k(Xy - gy, kdt), _k(Xz - gz, kdt)
        ir, inv3 = _inv13(dx * dx + dy * dy + dz * dz)
        u = nx * (hy * dz - hz * dy) + ny * (hz * dx - hx * dz) + nz * (hx * dy - hy * dx)
        t3 = aa * inv3                                                 # a / rho^3
        t5 = 3.0 * aa * u * inv3 * ir * ir                             # 3 a u / rho^5
        r1 = jnp.sum(t3 * u, axis=(2, 3)).astype(f64)                  # (s, c)
        sj = lambda e: jnp.sum(e, axis=jax_)                           # sum over copies -> (s, c, k)
        Gh = jnp.stack([sj(-t3 * (ny * hz - nz * hy) + t5 * dx),
                        sj(-t3 * (nz * hx - nx * hz) + t5 * dy),
                        sj(-t3 * (nx * hy - ny * hx) + t5 * dz)], -1)  # (s, c, k, 3)
        Ghp = jnp.stack([sj(t3 * (dy * nz - dz * ny)),
                         sj(t3 * (dz * nx - dx * nz)),
                         sj(t3 * (dx * ny - dy * nx))], -1)
        return r1, Gh, Ghp

    r1, Gh, Ghp = _over_points(blk, (Xp, npk, a), C * ncopy * nq)
    Jc = (jnp.einsum("sckd,km->scdm", Gh, _k(cb.B0, kdt))
          + jnp.einsum("sckd,km->scdm", Ghp, _k(cb.B1, kdt)))          # (s, c, 3, K)
    J = (Jc.astype(f64) * Ib[None, :, None, None]).reshape(Ns, C * 3 * K)
    r = r1 @ Ib
    if cur.has_u(x, cb, C):
        J = jnp.concatenate([J, r1 @ cur.dI_du(C, I0, f64)], 1)
    return r, J


def stacked(x, t, cb, sym, surf, B_ref, I0, cfg, L_ref, kdt=jnp.float32):
    """Mixed-precision counterpart of residuals.stacked (regularizers stay FP64: tiny)."""
    X, n, w = surf
    rB, JB = residual_and_jacobian(x, cb, sym, X, n, surface_coeffs(w, B_ref), I0, kdt)
    return (jnp.concatenate([rB, regularizer_residual(x, t, cb, sym.C, cfg, L_ref)]),
            jnp.concatenate([JB, regularizer_jacobian(x, t, cb, sym.C, cfg, L_ref)]))


def stacked_residual(x, t, cb, sym, surf, B_ref, I0, cfg, L_ref, kdt=jnp.float32):
    X, n, w = surf
    return jnp.concatenate([magnetic_residual(x, cb, sym, X, n, w, B_ref, I0, kdt),
                            regularizer_residual(x, t, cb, sym.C, cfg, L_ref)])
