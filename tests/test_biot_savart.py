"""Field vs independent reference < 1e-8 (FP64).
  (a) closed-form on-axis field of a circular loop;
  (b) naive numpy einsum Biot-Savart at random off-filament points (multi-coil);
  (c) batched == single-candidate; (d) symmetry expansion vs explicit rotated coils.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

import xcoil  # noqa: F401
from xcoil.field import biot_savart as bs

MU0 = 4 * np.pi * 1e-7


def _circle(Q, R=1.0):
    s = np.linspace(0, 1, Q, endpoint=False)
    g = np.stack([R * np.cos(2 * np.pi * s), R * np.sin(2 * np.pi * s), np.zeros(Q)], 1)
    gd = np.stack([-2 * np.pi * R * np.sin(2 * np.pi * s),
                   2 * np.pi * R * np.cos(2 * np.pi * s), np.zeros(Q)], 1)
    return jnp.asarray(g[None]), jnp.asarray(gd[None])   # (1, Q, 3)


def test_on_axis_circle_analytic():
    Q, R, I = 256, 1.3, 2.5e5
    g, gd = _circle(Q, R)
    z = np.array([0.0, 0.4, 1.0, 2.0, -0.7])
    pts = jnp.asarray(np.stack([np.zeros_like(z), np.zeros_like(z), z], 1))
    B = np.asarray(bs.field(pts, g, gd, jnp.asarray([I])))
    Bz_exact = MU0 * I * R ** 2 / (2 * (R ** 2 + z ** 2) ** 1.5)
    err_z = np.max(np.abs(B[:, 2] - Bz_exact) / np.abs(Bz_exact))
    err_xy = np.max(np.abs(B[:, :2]))                     # transverse field must vanish on axis
    print(f"[biot_savart] on-axis circle  Bz relerr={err_z:.2e}  |Bxy|={err_xy:.2e}")
    assert err_z < 1e-8 and err_xy < 1e-8


def _ref_field(points, gamma, gamma_dash, currents):
    # independent numpy einsum Biot-Savart
    P = np.asarray(points); G = np.asarray(gamma); GD = np.asarray(gamma_dash); I = np.asarray(currents)
    C, Q, _ = G.shape
    B = np.zeros((P.shape[0], 3))
    for c in range(C):
        r = P[:, None, :] - G[c][None, :, :]             # (S,Q,3)
        inv = np.sum(r * r, -1) ** -1.5
        cross = np.cross(np.broadcast_to(GD[c][None], r.shape), r)
        B += (I[c] / Q) * np.sum(cross * inv[..., None], 1)
    return 1e-7 * B


def test_vs_numpy_reference_multicoil():
    rng = np.random.default_rng(0)
    Q = 128
    # two tilted, offset circular-ish coils
    s = np.linspace(0, 1, Q, endpoint=False)
    coils_g, coils_gd = [], []
    for shift, tilt in [(np.array([0, 0, 0.0]), 0.0), (np.array([0.3, 0.0, 0.5]), 0.4)]:
        g = np.stack([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s), tilt * np.sin(2 * np.pi * s)], 1) + shift
        gd = np.gradient(g, s, axis=0)                    # rough tangent; kernel uses it as-is
        coils_g.append(g); coils_gd.append(gd)
    G = jnp.asarray(np.stack(coils_g)); GD = jnp.asarray(np.stack(coils_gd))
    I = jnp.asarray([1.0e6, -0.7e6])
    pts = jnp.asarray(rng.uniform(-0.5, 0.5, (200, 3)) + np.array([2.0, 2.0, 2.0]))  # off-filament
    B = np.asarray(bs.field(pts, G, GD, I))
    Bref = _ref_field(pts, G, GD, I)
    err = np.max(np.linalg.norm(B - Bref, axis=1) / np.maximum(np.linalg.norm(Bref, axis=1), 1e-300))
    print(f"[biot_savart] vs numpy einsum (2 coils, 200 pts) relerr={err:.2e}")
    assert err < 1e-12   # same math, independent code -> machine precision


def test_batched_matches_single():
    rng = np.random.default_rng(1)
    Q, C, P = 64, 3, 50
    G = jnp.asarray(rng.standard_normal((2, C, Q, 3)))
    GD = jnp.asarray(rng.standard_normal((2, C, Q, 3)))
    I = jnp.asarray(rng.standard_normal((2, C)))
    pts = jnp.asarray(rng.standard_normal((P, 3)) * 3 + 5)
    Bb = bs.field_batched(pts, G, GD, I)
    for p in range(2):
        B1 = bs.field(pts, G[p], GD[p], I[p])
        assert float(jnp.max(jnp.abs(Bb[p] - B1))) < 1e-14
    print("[biot_savart] batched == single-candidate  OK")


def test_symmetry_expansion():
    rng = np.random.default_rng(2)
    Q, C = 64, 2
    G = jnp.asarray(rng.standard_normal((C, Q, 3)))
    GD = jnp.asarray(rng.standard_normal((C, Q, 3)))
    I = jnp.asarray(rng.standard_normal(C))
    pts = jnp.asarray(rng.standard_normal((40, 3)) * 3 + 6)
    gf, gdf, cf = bs.expand_symmetry(G, GD, I, nfp=3, stellsym=True)
    assert gf.shape[0] == C * 3 * 2
    B_full = bs.field(pts, gf, gdf, cf)
    # field of the expanded set equals sum of fields of each explicitly transformed period/flip
    from xcoil.field.biot_savart import _rot_z
    B_manual = jnp.zeros((pts.shape[0], 3))
    for k in range(3):
        Rk = _rot_z(2 * jnp.pi * k / 3)
        for flip in (False, True):
            Rm = (Rk @ jnp.diag(jnp.array([1.0, -1.0, -1.0]))).T if flip else Rk.T
            B_manual = B_manual + bs.field(pts, G @ Rm, GD @ Rm, (-I if flip else I))
    err = float(jnp.max(jnp.abs(B_full - B_manual)))
    print(f"[biot_savart] symmetry expansion consistency = {err:.2e}")
    assert err < 1e-12


if __name__ == "__main__":
    test_on_axis_circle_analytic()
    test_vs_numpy_reference_multicoil()
    test_batched_matches_single()
    test_symmetry_expansion()
    print("[biot_savart] passed")
