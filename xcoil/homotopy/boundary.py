"""Target boundary in Fourier form (VMEC convention), FP64 NumPy, untimed preprocessing.

    R(theta, phi) = sum_k rc_k cos(a_k) + rs_k sin(a_k)
    Z(theta, phi) = sum_k zs_k sin(a_k) + zc_k cos(a_k),     a_k = m_k theta - nfp n_k phi

phi is the cylindrical toroidal angle. Stellarator-symmetric boundaries have rs = zc = 0; the
non-symmetric terms are supported when present. Boundaries are stored as
surface/data/<name>.npz with keys xm, xn, rc, zs[, rs, zc], nfp, stellsym.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "surface", "data")


@dataclass(frozen=True)
class FourierBoundary:
    """A plasma boundary as Fourier series in the cylindrical angle phi:
    R = sum rc cos(m theta - n nfp phi) + rs sin(...), Z = sum zs sin(m theta - n nfp phi) + zc cos(...), with mode
    numbers xm, xn (n in units of nfp; rs = zc = 0 for stellarator symmetry). eval_rz / eval_xyz evaluate it."""
    xm: np.ndarray
    xn: np.ndarray
    rc: np.ndarray
    zs: np.ndarray
    rs: np.ndarray
    zc: np.ndarray
    nfp: int
    stellsym: bool
    name: str = ""

    @property
    def R00(self) -> float:
        sel = (self.xm == 0) & (self.xn == 0)
        return float(self.rc[sel].sum())

    def eval_rz(self, theta, phi):
        """Cylindrical R, Z at broadcastable theta, phi (one jitted program on the default JAX device, equal to the
        numpy sums at rounding level; XCOIL_TABLES=numpy: the numpy sums)."""
        if tables_on_device():
            th, ph = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
            th, ph = th.ravel(), ph.ravel()
            # chunks of points: the (points x modes) intermediates of one call reached tens of GB for large
            # high-order boundaries (nfp 7-8, aspect 12, 48th-order refits)
            step = max(1, int(EVAL_CHUNK_ELEMS // max(1, self.xm.size)))
            Rs, Zs = [], []
            for k in range(0, th.size, step):
                R, Z = _eval_rz_jax(self.xm, self.xn, self.rc, self.rs, self.zs, self.zc, float(self.nfp),
                                    th[k:k + step], ph[k:k + step])
                Rs.append(np.asarray(R)); Zs.append(np.asarray(Z))
            shape = np.broadcast_shapes(np.shape(theta), np.shape(phi))
            return (np.concatenate(Rs).reshape(shape) if Rs else np.zeros(shape),
                    np.concatenate(Zs).reshape(shape) if Zs else np.zeros(shape))
        th = np.asarray(theta, float)[..., None]
        ph = np.asarray(phi, float)[..., None]
        a = self.xm * th - self.nfp * self.xn * ph
        c, s = np.cos(a), np.sin(a)
        R = c @ self.rc + s @ self.rs
        Z = s @ self.zs + c @ self.zc
        return R, Z

    def eval_xyz(self, theta, phi):
        R, Z = self.eval_rz(theta, phi)
        ph = np.broadcast_to(np.asarray(phi, float), R.shape)
        return np.stack([R * np.cos(ph), R * np.sin(ph), Z], -1)


EVAL_CHUNK_ELEMS = int(os.environ.get("XCOIL_EVAL_CHUNK", str(2 ** 26)))  # points x modes per eval_rz call (~0.5 GB)


def tables_on_device():
    """Boundary sums (eval_rz, surface_path tables) and the start checks (starts._linked, _min_coil_coil) on the
    default JAX device (default; XCOIL_TABLES=numpy: the numpy loops, as before)."""
    return os.environ.get("XCOIL_TABLES", "jax") != "numpy"


def _eval_rz_jax(xm, xn, rc, rs, zs, zc, nfp, th, ph):
    return _eval_rz_jit()(xm, xn, rc, rs, zs, zc, nfp, th, ph)


_EVAL_RZ = []


def _eval_rz_jit():
    if not _EVAL_RZ:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def f(xm, xn, rc, rs, zs, zc, nfp, th, ph):
            a = xm * th[:, None] - nfp * xn * ph[:, None]
            c, s = jnp.cos(a), jnp.sin(a)
            return c @ rc + s @ rs, s @ zs + c @ zc
        _EVAL_RZ.append(f)
    return _EVAL_RZ[0]


def from_modes(xm, xn, rc, zs, nfp, stellsym=True, rs=None, zc=None, name=""):
    """FourierBoundary from mode arrays xm, xn (n in units of nfp), rc, zs [m] (rs, zc: non-symmetric terms)."""
    xm = np.asarray(xm, float); xn = np.asarray(xn, float)
    rc = np.asarray(rc, float); zs = np.asarray(zs, float)
    rs = np.zeros_like(rc) if rs is None else np.asarray(rs, float)
    zc = np.zeros_like(rc) if zc is None else np.asarray(zc, float)
    return FourierBoundary(xm, xn, rc, zs, rs, zc, int(nfp), bool(stellsym), name)


# boundaries registered at run time by user_problem.make_problem ("user/<key>"): outside the package, so an installed
# (read-only) xcoil works; XCOIL_USER_DIR overrides the folder
USER_DIR = os.environ.get("XCOIL_USER_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "xcoil", "boundaries")


def path_of(name: str) -> str:
    """File of a named boundary: bundled names in DATA_DIR, "user/<key>" in USER_DIR."""
    if name.startswith("user/"):
        return os.path.join(USER_DIR, f"{name[5:]}.npz")
    return os.path.join(DATA_DIR, f"{name}.npz")


def load(name: str) -> FourierBoundary:
    """A named boundary: a bundled one (landreman_paul_qa, w7x, qh_reactor, cfqs_2b40,
    rotating_ellipse) or one registered by user_problem.make_problem ("user/<key>")."""
    d = np.load(path_of(name))
    get = lambda k: d[k] if k in d.files else None
    return from_modes(d["xm"], d["xn"], d["rc"], d["zs"], int(d["nfp"]),
                      bool(int(d["stellsym"])), get("rs"), get("zc"), name)


def save(b: FourierBoundary, name: str):
    """Store b under `name` (see path_of)."""
    os.makedirs(os.path.dirname(path_of(name)), exist_ok=True)
    np.savez(path_of(name), xm=b.xm, xn=b.xn, rc=b.rc, zs=b.zs,
             rs=b.rs, zc=b.zc, nfp=b.nfp, stellsym=int(b.stellsym))


def from_simsopt(surface, name="", tol=0.0) -> FourierBoundary:
    """Extract all nonzero modes of a simsopt SurfaceRZFourier (sym and non-sym terms)."""
    xm, xn, rc, zs, rs, zc = [], [], [], [], [], []
    for m in range(surface.mpol + 1):
        for n in range(-surface.ntor, surface.ntor + 1):
            if m == 0 and n < 0:
                continue
            r_c = surface.get_rc(m, n); z_s = surface.get_zs(m, n) if (m, n) != (0, 0) else 0.0
            r_s = z_c = 0.0
            if not surface.stellsym:
                r_s = surface.get_rs(m, n) if (m, n) != (0, 0) else 0.0
                z_c = surface.get_zc(m, n)
            if max(abs(r_c), abs(z_s), abs(r_s), abs(z_c)) <= tol:
                continue
            xm.append(m); xn.append(n); rc.append(r_c); zs.append(z_s); rs.append(r_s); zc.append(z_c)
    return from_modes(xm, xn, rc, zs, surface.nfp, surface.stellsym, rs, zc, name)


def fit_grid(R, Z, nfp, name="") -> FourierBoundary:
    """Tensor-grid input -> Fourier boundary by exact 2D DFT.

    R, Z: (ntheta, nphi) on theta_i = 2 pi i/ntheta, phi_j = 2 pi j/(nfp nphi) (one field
    period, consistent poloidal label). Returns all resolvable modes (general, non-symmetric)."""
    R = np.asarray(R, float); Z = np.asarray(Z, float)
    nt, npz = R.shape
    FR = np.fft.fft2(R) / (nt * npz)           # index [p, q] <-> exp(i(p theta + q zeta))
    FZ = np.fft.fft2(Z) / (nt * npz)
    xm, xn, rc, rs, zs, zc = [], [], [], [], [], []
    mmax, nmax = nt // 2, npz // 2
    for m in range(0, mmax):
        for n in range(-nmax + 1, nmax):
            if m == 0 and n < 0:
                continue
            # a = m theta - nfp n phi = m theta - n zeta  -> exp(i a) coefficient pair (p=m, q=-n)
            p, q = m % nt, (-n) % npz
            cR, cZ = FR[p, q], FZ[p, q]
            f = 1.0 if (m == 0 and n == 0) else 2.0   # conjugate pair folded into cos/sin
            rc.append(f * cR.real); rs.append(-f * cR.imag)
            zc.append(f * cZ.real); zs.append(-f * cZ.imag)
            xm.append(m); xn.append(n)
    rc, rs, zs, zc = map(np.asarray, (rc, rs, zs, zc))
    keep = np.maximum.reduce([abs(rc), abs(rs), abs(zs), abs(zc)]) > 1e-15
    stellsym = bool(np.all(abs(rs[keep]) < 1e-14) and np.all(abs(zc[keep]) < 1e-14))
    return from_modes(np.asarray(xm)[keep], np.asarray(xn)[keep], rc[keep], zs[keep], nfp,
                      stellsym, rs[keep], zc[keep], name)
