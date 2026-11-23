"""Certification kernels vs brute-force references, and the certified metrics
of the simsopt LP-QA reference coils vs simsopt's own objects (offline verification)."""
from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.verify import certify as cf
from xcoil.homotopy import boundary as bd, seeds, problem_h as ph
from xcoil.homotopy.config import for_device
from xcoil.curves import fourier as fr, symmetry as sm

DATA = os.environ.get("XCOIL_REF_DIR", bd.DATA_DIR)   # reference solutions are not shipped


def test_segment_distance_bruteforce():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(400, 4, 3))
    P[:40, 1] = P[:40, 0]                                   # degenerate first segment
    P[40:80, 3] = P[40:80, 2]                               # degenerate second segment
    P[80:120, 3] = P[80:120, 2] + (P[80:120, 1] - P[80:120, 0]) * 0.7   # parallel
    d = np.asarray(cf.segment_distance(*[jnp.asarray(P[:, k]) for k in range(4)]))
    u = np.linspace(0, 1, 801)
    A = P[:, None, 0] + u[None, :, None] * (P[:, 1] - P[:, 0])[:, None]
    B = P[:, None, 2] + u[None, :, None] * (P[:, 3] - P[:, 2])[:, None]
    brute = np.array([np.min(np.linalg.norm(a[:, None] - b[None, :], axis=-1)) for a, b in zip(A, B)])   # 15 MB per pair, not 6 GB at once
    assert np.all(d <= brute + 1e-12)                        # exact is never above the sampled min
    assert np.max(brute - d) < 5e-3 * np.max(np.linalg.norm(P[:, 1] - P[:, 0], axis=-1))


def test_point_triangle_bruteforce():
    rng = np.random.default_rng(1)
    T = rng.normal(size=(300, 3, 3)); p = rng.normal(size=(300, 3)) * 1.5
    d = np.asarray(cf.point_triangle_distance(jnp.asarray(p), jnp.asarray(T)))
    n = 300
    a, b = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    m = (a + b) <= 1.0
    a, b = a[m], b[m]
    Q = T[:, None, 0] + a[None, :, None] * (T[:, 1] - T[:, 0])[:, None] + b[None, :, None] * (T[:, 2] - T[:, 0])[:, None]
    brute = np.min(np.linalg.norm(Q - p[:, None], axis=-1), axis=1)
    assert np.all(d <= brute + 1e-12)
    assert np.max(brute - d) < 2e-2


def test_linking_hopf():
    """Hopf-linked circles give |Lk| = 1, separated circles 0 (JAX Gauss integral)."""
    s = np.arange(512) / 512; w = 2 * np.pi
    c1 = np.stack([np.cos(w * s), np.sin(w * s), 0 * s], 1); d1 = w * np.stack([-np.sin(w * s), np.cos(w * s), 0 * s], 1)
    c2 = np.stack([1 + np.cos(w * s), 0 * s, np.sin(w * s)], 1); d2 = w * np.stack([-np.sin(w * s), 0 * s, np.cos(w * s)], 1)
    c3 = c2 + np.array([5.0, 0, 0])

    def lk(a, da, b, db):
        D = a[:, None] - b[None]; r3 = np.linalg.norm(D, axis=-1) ** 3
        return float(np.sum(np.sum(np.cross(da[:, None], db[None]) * D, -1) / r3) / 512 ** 2 / (4 * np.pi))
    assert abs(abs(lk(c1, d1, c2, d2)) - 1) < 1e-6 and abs(lk(c1, d1, c3, d2)) < 1e-6


@pytest.fixture(scope="module")
def lpqa_ref():
    cfg = for_device("lpqa"); b = bd.load(cfg.surface)
    seed = seeds.build(b, 4, ph.limits_from(cfg), cfg.basis, cfg.order, cfg.nq_opt, cfg.grid_dist)
    T = ph.build_tables(b, cfg, seed)
    V = cf.build(b, cfg, T)
    ref = os.path.join(DATA, "ref_lpqa_simsopt.npz")
    if os.path.exists(ref):                                  # simsopt stage-two reference coils, if provided
        x = jnp.asarray(np.load(ref)["x"])
    else:                                                    # otherwise a generated 3D-offset start (same layout)
        from xcoil.fixed.starts import generate
        s = [s for s in generate(b, 4, cfg.order, 18.2, ph.limits_from(cfg).dcs)
             if s.family == "offset_3d" and s.label == "L85"][0]
        x = jnp.asarray(s.x)
    m = jax.jit(lambda x: cf.certify_one(x, 18.2, V, ph.limits_from(cfg)))(x)
    return cfg, b, T, V, x, jax.tree_util.tree_map(np.asarray, m)


def test_max_curvature_dense(lpqa_ref):
    cfg, b, T, V, x, m = lpqa_ref
    _, gd, gdd = fr.coils_curves(x, fr.make_basis("fourier", 2 ** 17, cfg.order), 4)
    from xcoil.constraints import pointwise as pw
    true = float(jnp.max(pw.curvature(gd, gdd)))
    assert abs(float(m["max_kappa"]) - true) <= 1e-7 * true


def test_lpqa_reference_vs_simsopt(lpqa_ref):
    """Certified metrics of the simsopt stage-two LP-QA coils vs simsopt's own evaluation."""
    cfg, b, T, V, x, m = lpqa_ref
    from simsopt.geo import (SurfaceRZFourier, CurveLength, MeanSquaredCurvature, CurveCurveDistance,
                             CurveSurfaceDistance, curves_to_vtk)
    from simsopt.field import Current, coils_via_symmetries, BiotSavart
    from simsopt.field.coil import coils_via_symmetries as cvs
    base = [fr.to_simsopt(np.asarray(x).reshape(4, -1)[c], 4096, cfg.order) for c in range(4)]
    cur = [Current(1.0) for _ in range(4)]
    coils = coils_via_symmetries(base, cur, b.nfp, True)
    L = sum(float(CurveLength(c).J()) for c in base)
    msc = [float(MeanSquaredCurvature(c).J()) for c in base]
    s = SurfaceRZFourier(nfp=b.nfp, stellsym=True, mpol=int(b.xm.max()), ntor=int(np.abs(b.xn).max()),
                         quadpoints_phi=np.linspace(0, 1, 256, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 256, endpoint=False))
    for k in range(len(b.xm)):
        s.set_rc(int(b.xm[k]), int(b.xn[k]), float(b.rc[k])); s.set_zs(int(b.xm[k]), int(b.xn[k]), float(b.zs[k]))
    dcc_s = float(CurveCurveDistance([c.curve for c in coils], 0.0).shortest_distance())
    dcs_s = float(CurveSurfaceDistance(base, s, 0.0).shortest_distance())
    print(f"L {float(m['L']):.10f} vs {L:.10f}; MSC max {float(np.max(m['msc'])):.8f} vs {max(msc):.8f}; "
          f"d_cc {float(m['d_cc']):.6f} vs {dcc_s:.6f}; d_cs {float(m['d_cs']):.6f} vs {dcs_s:.6f}; "
          f"mean |Bn|/|B| {100 * float(m['mean_bn']):.4f}% max {100 * float(m['max_bn']):.4f}%; "
          f"v {float(m['v']):.2e}; linking {int(np.abs(m['linking']).max())}")
    assert abs(float(m["L"]) - L) <= 1e-10 * L
    assert abs(float(np.max(m["msc"])) - max(msc)) <= 1e-8 * max(msc)
    assert abs(float(m["d_cc"]) - dcc_s) <= 2e-4                  # polylines (1024 seg) vs 4096-pt sample min
    assert abs(float(m["d_cs"]) - dcs_s) <= 2e-3                  # triangulated 64x128/fp vs 256^2 samples
    assert int(np.abs(m["linking"]).max()) == 0 and bool(m["finite"])
