"""Starting-coil families for fixed-surface coil optimization (P1). Built from the target boundary
only (no previously optimized coils).

Base coils sit at the toroidal angles phi_c = (c + 1/2) pi / (nfp C) (stellarator symmetric, C per
half period, simsopt create_equally_spaced_curves convention). Families:

  circle         planar circle centred at (R_00, 0) in the (e_R, e_z) plane, radius R1
  offset_planar  planar offset (distance d) of the axisymmetric (n = 0) cross-section
  offset_3d      surface-following: X(theta, phi_c) + d n_hat(theta, phi_c) of the full 3D boundary
                 (non-planar: the unit normal has a toroidal component)
  modulated_3d   offset_3d on phi = phi_c + a f(m theta), f = cos / sin, m = 1, 2, both signs: on the
                 clearing offset, a = 1/4 of the coil spacing (modulated())

Sizing, per family: the smallest size that clears the coil-surface limit (x 1.1), the sizes giving
70 / 85 / 100 % of the total-length cap (root-found), and for circles the 3 rc(1,0) radius of
auglag_w7x.py. Variants on the 85 % member: coil planes tilted +-8 deg about the local e_R axis,
and (off by default) two random Fourier perturbations. Default families:
offset_planar and offset_3d (circles available on request).
Every start is fitted by least squares to the solver's Fourier basis (order M) from
arclength-uniform samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod
from xcoil.homotopy import boundary as bd

N_FIT = 1024


@dataclass
class Start:
    family: str
    label: str
    params: dict
    x: np.ndarray                   # (C * 3K [+ C - 1],) DOFs, simsopt CurveXYZFourier order [+ currents u]
    points: np.ndarray              # (C, N_FIT, 3) base-coil samples before fitting
    info: dict = field(default_factory=dict)


def _arclength_resample(P, n=N_FIT):
    P = np.asarray(P, float)
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)]); s /= s[-1]
    u = np.arange(n) / n
    Pc = np.vstack([P, P[:1]])
    return np.stack([np.interp(u, s, Pc[:, k]) for k in range(3)], 1)


def _fit(points_c, order, kind="fourier"):
    """(C, n, 3) arclength-uniform samples -> DOF vector."""
    n = points_c.shape[1]
    B0 = fr.scalar_basis_at(kind, np.arange(n) / n, order)[0]
    pinv = np.linalg.pinv(B0)
    return np.concatenate([(pinv @ P).T.reshape(-1) for P in points_c])


def _length(points_c):
    return float(sum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum() for P in points_c))


def _cached(b, key, fn):
    """Per-boundary memo of surface evaluations (the size searches evaluate the same coil planes and
    the same dense surface sample for every trial size; the boundary is immutable)."""
    memo = b.__dict__.get("_starts_memo")
    if memo is None:
        memo = {}
        object.__setattr__(b, "_starts_memo", memo)
    if key not in memo:
        memo[key] = fn()
    return memo[key]


def _surface_points(b, phi, n=512):
    th = 2 * np.pi * np.arange(n) / n
    return np.asarray(b.eval_xyz(th, np.full(n, phi)))


def _surface_normals(b, phi, n=512, h=1e-5):
    return _cached(b, ("normals", float(phi), int(n), float(h)), lambda: _surface_normals_eval(b, phi, n, h))


def _surface_normals_eval(b, phi, n, h):
    th = 2 * np.pi * np.arange(n) / n
    X = np.asarray(b.eval_xyz(th, np.full(n, phi)))
    Xt = (np.asarray(b.eval_xyz(th + h, np.full(n, phi))) - np.asarray(b.eval_xyz(th - h, np.full(n, phi)))) / (2 * h)
    Xp = (np.asarray(b.eval_xyz(th, np.full(n, phi + h))) - np.asarray(b.eval_xyz(th, np.full(n, phi - h)))) / (2 * h)
    N = np.cross(Xp, Xt); N /= np.linalg.norm(N, axis=1, keepdims=True)
    # outward: away from the magnetic-axis estimate (centroid of the cross-section)
    c = X.mean(0)
    if np.mean(np.sum((X - c) * N, 1)) < 0:
        N = -N
    return X, N


def _plane_basis(phi):
    eR = np.array([np.cos(phi), np.sin(phi), 0.0]); ez = np.array([0.0, 0.0, 1.0])
    ephi = np.array([-np.sin(phi), np.cos(phi), 0.0])
    return eR, ephi, ez


# --- families: base-coil point sets as a function of the size parameter ---------------------

def circle_points(b, phis, R1, n=N_FIT):
    s = 2 * np.pi * np.arange(n) / n
    out = []
    for ph in phis:
        eR, _, ez = _plane_basis(ph)
        c = b.R00 * eR
        out.append(c + R1 * (np.cos(s)[:, None] * eR + np.sin(s)[:, None] * ez))
    return np.stack(out)


def offset_planar_points(b, phis, d, n=N_FIT):
    from xcoil.homotopy import seeds as sd
    from xcoil.homotopy import surface_path as sp
    sigma = sp._cross_section_orientation(b)
    R, Z = sd._offset(b, d, sigma)
    out = []
    for ph in phis:
        P = np.stack([R * np.cos(ph), R * np.sin(ph), Z], 1)
        out.append(_arclength_resample(P, n))
    return np.stack(out)


def offset_3d_points(b, phis, d, n=N_FIT):
    out = []
    for ph in phis:
        X, N = _surface_normals(b, ph)
        out.append(_arclength_resample(X + d * N, n))
    return np.stack(out)


MOD_MODES = (("cos", 1), ("sin", 1), ("cos", 2), ("sin", 2))


def modulated_3d_points(b, phis, d, amp, kind="sin", m=1, n=N_FIT, h=1e-5):
    """Toroidally modulated surface-following coils: X(theta, phi) + d n_hat(theta, phi) on
    phi = phi_c + amp f(m theta), f = cos or sin (theta the boundary's poloidal angle; amp in rad)."""
    th = 2 * np.pi * np.arange(n) / n
    f = (np.cos if kind == "cos" else np.sin)(m * th)
    out = []
    for ph0 in phis:
        ph = ph0 + amp * f
        X = np.asarray(b.eval_xyz(th, ph))
        Xt = (np.asarray(b.eval_xyz(th + h, ph)) - np.asarray(b.eval_xyz(th - h, ph))) / (2 * h)
        Xp = (np.asarray(b.eval_xyz(th, ph + h)) - np.asarray(b.eval_xyz(th, ph - h))) / (2 * h)
        N = np.cross(Xp, Xt); N /= np.linalg.norm(N, axis=1, keepdims=True)
        if np.mean(np.sum((X - X.mean(0)) * N, 1)) < 0:              # outward
            N = -N
        out.append(_arclength_resample(X + d * N, n))
    return np.stack(out)


def _min_clearance(b, pts_c, nfp_grid=64):
    """Minimum distance from the base coils to the full boundary (dense sample)."""
    def grid():
        ph = np.linspace(0, 2 * np.pi, 256, endpoint=False); th = np.linspace(0, 2 * np.pi, nfp_grid, endpoint=False)
        PH, TH = np.meshgrid(ph, th, indexing="ij")
        return np.asarray(b.eval_xyz(TH.ravel(), PH.ravel()))
    S = _cached(b, ("clearance_grid", int(nfp_grid)), grid)
    return min(_min_dist(P[::8], S) for P in pts_c)


def _size_for_length(fun, target, lo, hi):
    f = lambda v: _length(fun(v)) - target
    if f(lo) > 0:
        return None
    while f(hi) < 0:
        hi *= 1.5
    return brentq(f, lo, hi, xtol=1e-6)


def _size_for_clearance(fun, b, dcs, lo, hi):
    f = lambda v: _min_clearance(b, fun(v)) - 1.1 * dcs
    if f(lo) > 0:
        return lo
    while f(hi) < 0:
        hi *= 1.5
    return brentq(f, lo, hi, xtol=1e-4)


def _min_dist(A, B):
    """Smallest distance between the point sets A (n, 3) and B (m, 3): |a|^2 + |b|^2 - 2 a.b with
    one matrix product (no (n, m, 3) difference array)."""
    d2 = np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None] - 2.0 * (A @ B.T)
    return float(np.sqrt(max(float(d2.min()), 0.0)))


def _expand_np(P, nfp, stellsym):
    """curves/symmetry.expand in numpy: (C, n, 3) base coils -> (Nphys, n, 3)."""
    C = P.shape[0]
    key = (C, int(nfp), bool(stellsym))
    if key not in _SYM:
        sym = sym_mod.build(C, nfp, stellsym)
        _SYM[key] = (np.asarray(sym.R), np.asarray(sym.coil))
    R, coil = _SYM[key]
    return np.einsum("jab,jkb->jka", R, P[coil])


_SYM = {}


def _min_coil_coil(points_c, nfp, stellsym, n=256):
    """Smallest distance between distinct coils of the full symmetric set (base coils sampled at n
    points, every other coil of the set; symmetry as curves/symmetry.py)."""
    C = points_c.shape[0]
    P = np.asarray(points_c)[:, :: max(1, points_c.shape[1] // n)]
    full = _expand_np(P, nfp, stellsym)                            # (Nphys, n, 3); copy c = base coil c
    if bd.tables_on_device():
        return float(_min_coil_coil_jax(P, full))
    d = np.inf
    for c in range(C):                                             # base coil c against all others
        others = np.concatenate([full[j] for j in range(full.shape[0]) if j != c])
        d = min(d, _min_dist(P[c], others))
    return d


def _linked(points_c, nfp, stellsym, n=128):
    """True if any two coils of the full symmetric set have a nonzero (rounded) Gauss linking number.
    Every pair is a symmetry image of a pair with a base coil, and the symmetries are proper rotations,
    so base coils against all coils suffice."""
    C = points_c.shape[0]
    sym = sym_mod.build(C, nfp, stellsym)
    Q = np.asarray(points_c)[:, :: max(1, points_c.shape[1] // n)]
    G = np.asarray(sym_mod.expand(Q, sym))
    seg = lambda X: np.roll(X, -1, axis=-2) - X
    mid = lambda X: 0.5 * (np.roll(X, -1, axis=-2) + X)
    coil, ident = np.asarray(sym.coil), np.asarray(sym.ident)
    if bd.tables_on_device():
        skip = (coil[None, :] == np.arange(C)[:, None]) & ident[None, :]
        return bool(_max_linking_jax(Q, G, skip) > 0.5)
    for c in range(C):
        a, da = mid(Q[c]), seg(Q[c])
        for j in range(G.shape[0]):
            if coil[j] == c and ident[j]:
                continue
            b, db = mid(G[j]), seg(G[j])
            D = a[:, None] - b[None]
            lk = np.sum(np.einsum("ijk,ijk->ij", D, np.cross(da[:, None], db[None])) /
                        np.linalg.norm(D, axis=-1) ** 3) / (4 * np.pi)
            if abs(lk) > 0.5:
                return True
    return False


@jax.jit
def _min_coil_coil_jax(P, full):
    """_min_coil_coil's loop as one program on the default JAX device (base coil c against every coil j != c)."""
    J = full.shape[0]
    bb = jnp.sum(full * full, -1)

    def one(c):
        a = P[c]
        d2 = jnp.sum(a * a, 1)[:, None, None] + bb[None] - 2.0 * jnp.einsum("ix,jkx->ijk", a, full)
        return jnp.min(jnp.where((jnp.arange(J) == c)[None, :, None], jnp.inf, d2))
    return jnp.sqrt(jnp.maximum(jnp.min(jax.lax.map(one, jnp.arange(P.shape[0]))), 0.0))


@jax.jit
def _max_linking_jax(Q, G, skip):
    """Largest |Gauss linking number| of _linked's pairs as one program on the default JAX device (skip: (C, Nphys)
    the identity copies; a NaN pair counts as unlinked, as in the loop)."""
    seg = lambda X: jnp.roll(X, -1, axis=-2) - X
    mid = lambda X: 0.5 * (jnp.roll(X, -1, axis=-2) + X)
    b, db = mid(G), seg(G)                                         # (Nphys, n, 3)

    def one(args):
        q, sk = args
        a, da = mid(q), seg(q)
        D = a[:, None, None] - b[None]                             # (n, Nphys, n, 3)
        lk = jnp.sum(jnp.einsum("ijkl,ijkl->ijk", D, jnp.cross(da[:, None, None], db[None])) /
                     jnp.linalg.norm(D, axis=-1) ** 3, axis=(0, 2)) / (4 * jnp.pi)
        return jnp.max(jnp.where(sk | jnp.isnan(lk), 0.0, jnp.abs(lk)))
    return jnp.max(jax.lax.map(one, (Q, skip)))


def _admissible_size(fun, lo, v, nfp, stellsym, dmin, steps=24, bisect=16):
    """Largest size in [lo, v] whose coils are unlinked and at least dmin apart, searched down from v
    (None if there is none). Requiring the separation keeps the start away from the crossings where
    a sampled linking integral is unreliable (and gives the solver a start that meets d_cc)."""
    ok = lambda u: _min_coil_coil(fun(u), nfp, stellsym) >= dmin and not _linked(fun(u), nfp, stellsym, n=256)
    if ok(v):
        return v
    grid = np.linspace(v, lo, steps + 1)
    for i in range(1, steps + 1):
        if ok(grid[i]):
            hi, good = grid[i - 1], grid[i]
            for _ in range(bisect):
                mid = 0.5 * (hi + good)
                hi, good = (hi, mid) if ok(mid) else (mid, good)
            return good
    return None


def _cap_admissible(fun, v0, v, dcc, nfp, stellsym, steps=32, bisect=20):
    """Largest size in [v0, v] reachable from v0 with all coils of the symmetric set at least dcc
    apart along the way (coils cannot become linked without touching, so the capped start is
    unlinked). Returns v when the whole path is admissible."""
    ok = lambda u: _min_coil_coil(fun(u), nfp, stellsym) >= dcc
    grid = np.linspace(v0, v, steps + 1)
    for i in range(1, steps + 1):
        if not ok(grid[i]):
            lo, hi = grid[i - 1], grid[i]
            for _ in range(bisect):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            return lo
    return v


def _tilt(points_c, phis, deg):
    out = []
    for P, ph in zip(points_c, phis):
        eR, ephi, ez = _plane_basis(ph)
        c = P.mean(0); a = np.deg2rad(deg)
        # rotation about the e_R axis through the centroid: mixes e_phi and e_z
        Rm = (np.outer(eR, eR) + np.cos(a) * (np.outer(ephi, ephi) + np.outer(ez, ez))
              + np.sin(a) * (np.outer(ez, ephi) - np.outer(ephi, ez)))
        out.append(c + (P - c) @ Rm.T)
    return np.stack(out)


MOD_AMP = 0.25          # modulation amplitude / coil spacing pi / (nfp C) (QUASR's coils: ~0.3-0.4 per mode)


def modulated(b, C, order, cap, d, stellsym=True, kind="fourier", admissible_dcc=None, unlinked=False,
              modes=MOD_MODES, amp=MOD_AMP):
    """Toroidally modulated offset_3d starts on the offset d (generate: the clearing offset, whose
    coils are the farthest apart): for every mode (cos / sin, m) and sign, amplitude amp x the coil spacing
    pi / (nfp C) (reduced to the length cap if longer), capped at the largest amplitude reachable
    from the unmodulated start with all coils at least admissible_dcc apart; with unlinked, an
    inadmissible result is shrunk to the largest admissible amplitude (_admissible_size) or dropped;
    so is one whose amplitude ends below a quarter of the intended one. The searches use 256 points
    per coil."""
    phis = sym_mod.base_toroidal_angles(C, b.nfp, stellsym)
    a0 = amp * np.pi / (b.nfp * C)
    out = []
    for fk, m in modes:
        for sgn in (+1.0, -1.0):
            fun = lambda a, fk=fk, m=m, sgn=sgn, n=256: modulated_3d_points(b, phis, d, sgn * a, fk, m, n=n)
            prm = dict(d=float(d), mode=f"{fk}{m}", sign=sgn)
            a = a0
            if _length(fun(a)) > cap:
                a = _size_for_length(fun, cap, 0.0, a0)
                if a is None:
                    continue
                prm["amp_length_capped"] = float(sgn * a0)
            if admissible_dcc is not None:
                ac = _cap_admissible(fun, 0.0, a, admissible_dcc, b.nfp, stellsym)
                if ac < a:
                    prm["amp_uncapped"] = float(sgn * a)
                a = ac
            if unlinked:
                u = _admissible_size(fun, 0.0, a, b.nfp, stellsym, admissible_dcc or 0.0)
                if u is None:
                    continue
                a = u
            if a < 0.25 * a0:                                 # ~ the unmodulated start
                continue
            P = fun(a, n=N_FIT)
            prm["amp"] = float(sgn * a)
            out.append(Start("modulated_3d", f"mod_{fk}{m}{'+' if sgn > 0 else '-'}", prm, _fit(P, order, kind), P))
    return out


def generate(b, C, order, cap, dcs, stellsym=True, kind="fourier", seed=0,
             families=("offset_planar", "offset_3d"), random_variants=False, free_currents=True,
             admissible_dcc=None, unlinked=False):
    """All starts for one device: returns a list of Start. free_currents: append u = 0 (equal
    currents, C - 1 free current variables; curves/currents). admissible_dcc: cap every
    length-sized start at the largest size reachable from the clearing size with all coils at least
    admissible_dcc apart (None: uncapped); capped starts record d_uncapped.
    unlinked: every start must be admissible - unlinked (no nonzero linking number in the symmetric
    coil set) with all coils at least admissible_dcc apart: an inadmissible start is shrunk to the
    largest admissible size below it (the original recorded as <p>_linked) or dropped if there is
    none; an inadmissible tilted start is dropped."""
    phis = sym_mod.base_toroidal_angles(C, b.nfp, stellsym)
    fams_all = dict(circle=lambda v: circle_points(b, phis, v),
                    offset_planar=lambda v: offset_planar_points(b, phis, v),
                    offset_3d=lambda v: offset_3d_points(b, phis, v))
    fams = {k: fams_all[k] for k in families if k in fams_all}
    if "modulated_3d" in families and "offset_3d" not in fams:
        raise ValueError("modulated_3d needs offset_3d (its base offset)")
    base_d = None
    rc10 = abs(float(b.rc[(b.xm == 1) & (b.xn == 0)][0]))     # minor-radius scale (sign = theta orientation)
    rng = np.random.default_rng(seed)
    starts = []
    for fam, fun in fams.items():
        pname = "R1" if fam == "circle" else "d"
        lo = 0.5 * rc10 if fam == "circle" else 0.2 * dcs
        sizes = {"clear": _size_for_clearance(fun, b, dcs, lo, 4 * rc10)}
        uncapped = {}
        for f in (0.70, 0.85, 1.00):
            v = _size_for_length(fun, f * cap, lo, 4 * rc10)
            if v is not None:
                lab = f"L{int(round(100 * f))}"
                if admissible_dcc is not None and v > sizes["clear"]:
                    vc = _cap_admissible(fun, sizes["clear"], v, admissible_dcc, b.nfp, stellsym)
                    if vc < v:
                        uncapped[lab] = v
                    v = vc
                sizes[lab] = v
        if fam == "circle":
            sizes["script3rc10"] = 3.0 * rc10
        relinked = {}
        if unlinked:
            for lab in list(sizes):
                u = _admissible_size(fun, lo, sizes[lab], b.nfp, stellsym, admissible_dcc or 0.0)
                if u is None:
                    del sizes[lab]
                elif u < sizes[lab]:
                    relinked[lab], sizes[lab] = sizes[lab], u
        if fam == "offset_3d":
            base_d = sizes.get("clear")
        for lab, v in sizes.items():
            P = fun(v)
            prm = {pname: float(v), **({f"{pname}_uncapped": float(uncapped[lab])} if lab in uncapped else {}),
                   **({f"{pname}_linked": float(relinked[lab])} if lab in relinked else {})}
            starts.append(Start(fam, lab, prm, _fit(P, order, kind), P))
        mid = sizes.get("L85")
        if mid is not None:
            P = fun(mid)
            for deg in (+8.0, -8.0):
                Pt = _tilt(P, phis, deg)
                if unlinked and (_min_coil_coil(Pt, b.nfp, stellsym) < (admissible_dcc or 0.0)
                                 or _linked(Pt, b.nfp, stellsym, n=256)):
                    continue
                prm = {pname: float(mid), "tilt_deg": deg,
                       **({f"{pname}_uncapped": float(uncapped["L85"])} if "L85" in uncapped else {})}
                starts.append(Start(fam, f"L85_tilt{deg:+.0f}", prm, _fit(Pt, order, kind), Pt))
            if not random_variants:
                continue
            x0 = _fit(P, order, kind)
            scale = float(np.max(np.abs(P - P.mean(1, keepdims=True))))
            K = 2 * order + 1
            for r in range(2):
                noise = rng.normal(size=x0.size) * 0.02 * scale / np.sqrt(K)
                noise.reshape(C, 3, K)[:, :, 0] = 0.0                    # keep the coil centres
                starts.append(Start(fam, f"L85_rand{r}", {pname: float(mid), "noise": 0.02, "rng": r}, x0 + noise, P))
    if "modulated_3d" in families and base_d is not None:
        starts += modulated(b, C, order, cap, base_d, stellsym, kind, admissible_dcc, unlinked)
    if free_currents:
        for st in starts:
            st.x = np.concatenate([st.x, np.zeros(C - 1)])
    return starts
