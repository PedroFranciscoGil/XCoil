"""XCoil for any target surface: load a boundary, build the coil problem, build the starting
coils, solve, check / certify, and write the coils. No simsopt dependency (simsopt is used only,
if installed, to read simsopt JSON surfaces and to write a simsopt BiotSavart file in addition).

Used by xcoil-coils (xcoil/cli.py). The rules:
  * coil quadrature fine enough for the distance limits: nq >= L_coil / (0.06 min(d_cc, d_cs)),
    at least max(128, 16 M), multiple of 32, at most 2048;
  * coil-surface distance grid with toroidal spacing <= d_cs / 4 (128-768 x 64);
  * starts built from the boundary alone: circles, planar offsets of the averaged cross-section,
    3D surface-following offsets (initial_coils, one chosen start; starts, all admissible ones);
  * the step test that rejects trial steps changing a coil linking number.

Surface inputs (load_boundary): a VMEC input file (input.*; RBC / ZBS / NFP), a VMEC wout file
(*.nc, netCDF3 via scipy, or netCDF4 if the netCDF4 package is installed), our .npz boundary format,
the name of a bundled boundary (w7x, landreman_paul_qa, qh_reactor, cfqs_2b40, rotating_ellipse), or - with
simsopt installed - a simsopt JSON file (a surface, or [surfaces, coils] as in QUASR: the outermost
surface; non R, Z surfaces are refitted to one and the fit error is reported).

Certification (solve_all, certify=): "all" certifies every result; "best" ranks the results by
the quick check (unlinked first, then true-limit violation, then objective) and certifies in that
order until one passes; "none" reports only the quick check. The quick check (quick_check) is
cheap and uncertified: flux error on the certification grid, sampled lengths, curvature, MSC,
point-sampled distances, linking numbers.
"""
from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import jax
import jax.numpy as jnp

from xcoil.homotopy import boundary as bd

ARC = dict(arc_pen_thresh=1e-2, arc_pen_weight=1.0, arc_pen_intervals=-1)   # simsopt auglag_qa.py threshold
VIOL_TOL = 1e-3


# --- boundaries -----------------------------------------------------------------------------------

def read_vmec_input(path):
    """Stellarator-symmetric boundary from a VMEC input namelist (RBC(n,m), ZBS(n,m), NFP)."""
    text = open(path).read()
    if re.search(r"\bLASYM\s*=\s*(\.?t|true)", text, re.I):
        raise ValueError("LASYM = T (non-stellarator-symmetric) inputs are not supported")
    nfp = int(re.search(r"\bNFP\s*=\s*(\d+)", text, re.I).group(1))
    num = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?)"
    modes = {}
    for key, n, m, v in re.findall(r"\b(RBC|ZBS)\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*" + num, text, re.I):
        modes.setdefault((int(m), int(n)), [0.0, 0.0])[0 if key.upper() == "RBC" else 1] = float(v.replace("d", "e").replace("D", "e"))
    keys = sorted(modes)
    return bd.from_modes([k[0] for k in keys], [k[1] for k in keys], [modes[k][0] for k in keys],
                         [modes[k][1] for k in keys], nfp, True)


def read_wout(path, s_index=-1):
    """Stellarator-symmetric boundary (last flux surface) from a VMEC wout file."""
    try:
        from scipy.io import netcdf_file
        f = netcdf_file(path, "r", mmap=False)
        get = lambda k: np.array(f.variables[k][()])
    except Exception:                                   # netCDF4 / HDF5 files
        try:
            import netCDF4
        except ImportError as ex:
            raise ImportError(f"{path} is not netCDF3; install the netCDF4 package to read it") from ex
        f = netCDF4.Dataset(path)
        get = lambda k: np.array(f.variables[k][:])
    nfp = int(get("nfp"))
    if "lasym__logical__" in f.variables and int(get("lasym__logical__")):
        raise ValueError("lasym wout files (non-stellarator-symmetric) are not supported")
    xm, xn = get("xm").astype(int), get("xn").astype(int) // nfp     # wout xn includes nfp
    return bd.from_modes(xm, xn, get("rmnc")[s_index], get("zmns")[s_index], nfp, True)


def write_vmec_input(b, path, name="xcoil"):
    """Minimal VMEC-style input file with the boundary (NFP, RBC, ZBS) - readable by read_vmec_input
    and by VMEC / simsopt."""
    lines = ["&INDATA", f"  ! boundary written by xcoil ({name})", "  LASYM = F", f"  NFP = {b.nfp}",
             f"  MPOL = {int(np.max(b.xm)) + 1}", f"  NTOR = {int(np.max(np.abs(b.xn)))}"]
    for m, n, rc, zs in zip(b.xm.astype(int), b.xn.astype(int), b.rc, b.zs):
        lines.append(f"  RBC({n},{m}) = {rc:.16e}   ZBS({n},{m}) = {zs:.16e}")
    open(path, "w").write("\n".join(lines + ["/", ""]))


def refit_rz(surface, modes=24, nfit=112):
    """R, Z Fourier series in the cylindrical angle fitted to any simsopt surface (exact cross
    sections, one field period, nfit x nfit points). Needs simsopt. Returns (boundary, error [m])."""
    from simsopt.geo import SurfaceRZFourier
    nfp = surface.nfp
    rz = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=modes, ntor=modes,
                          quadpoints_phi=np.linspace(0, 1 / nfp, nfit, endpoint=False),
                          quadpoints_theta=np.linspace(0, 1, nfit, endpoint=False))
    rz.least_squares_fit(np.stack([surface.cross_section(p, thetas=nfit) for p in np.array(rz.quadpoints_phi)]))
    err = 0.0
    for ph in (np.arange(8) + 0.37) / (8 * nfp):                     # angles between the fit points (turns)
        a, c = surface.cross_section(ph, thetas=256), rz.cross_section(ph, thetas=4096)
        Ra, Rc = np.hypot(a[:, 0], a[:, 1]), np.hypot(c[:, 0], c[:, 1])
        err = max(err, float(np.min(np.hypot(Ra[:, None] - Rc[None], a[:, 2][:, None] - c[:, 2][None]), 1).max()))
    fb = bd.from_simsopt(rz)
    keep = (np.abs(fb.rc) > 0) | (np.abs(fb.zs) > 0)
    return bd.from_modes(fb.xm[keep], fb.xn[keep], fb.rc[keep], fb.zs[keep], nfp, True), err


def load_boundary(spec):
    """(FourierBoundary, info) from spec: a VMEC / VMEX input file (input.*: RBC / ZBS / NFP), a wout file (*.nc, the
    last closed flux surface), an XCoil .npz boundary, a bundled name (landreman_paul_qa, w7x,
    qh_reactor, cfqs_2b40, rotating_ellipse) or, with simsopt installed, a simsopt JSON file (surfaces not given as
    R, Z series in the cylindrical angle are refitted; info["fit_err_m"] is the fit error). info: source, fit_err_m."""
    if not os.path.exists(spec) and os.path.exists(os.path.join(bd.DATA_DIR, f"{spec}.npz")):
        return bd.load(spec), dict(source="bundled boundary", fit_err_m=0.0)
    if not os.path.exists(spec):
        raise FileNotFoundError(f"{spec}: neither a file nor a bundled boundary")
    base = os.path.basename(spec)
    if spec.endswith(".npz"):
        d = np.load(spec)
        return bd.from_modes(d["xm"], d["xn"], d["rc"], d["zs"], int(d["nfp"]), True), dict(source="npz", fit_err_m=0.0)
    if spec.endswith(".nc"):
        return read_wout(spec), dict(source="VMEC wout file", fit_err_m=0.0)
    if spec.endswith(".json"):
        try:
            from simsopt._core import load
            from simsopt.geo import SurfaceRZFourier
        except ImportError as ex:
            raise ImportError("reading simsopt JSON files needs simsopt (pip install simsopt); "
                              "VMEC input / wout files, .npz and bundled boundaries do not") from ex
        obj = load(spec)
        if isinstance(obj, (list, tuple)):                           # e.g. QUASR: [surfaces, coils]
            obj = obj[0]
        s = obj[-1] if isinstance(obj, (list, tuple)) else obj
        if not s.stellsym:
            raise ValueError("only stellarator-symmetric surfaces are supported")
        if isinstance(s, SurfaceRZFourier):
            return bd.from_simsopt(s), dict(source="simsopt SurfaceRZFourier", fit_err_m=0.0)
        b, err = refit_rz(s)
        return b, dict(source=f"simsopt {type(s).__name__}, refitted to R, Z(cylindrical angle)", fit_err_m=err)
    return read_vmec_input(spec), dict(source="VMEC input file", fit_err_m=0.0)      # input.* and others


def r_max(b):
    th, ph = np.meshgrid(np.linspace(0, 2 * np.pi, 128), np.linspace(0, 2 * np.pi, 256))
    return float(np.max(b.eval_rz(th, ph)[0]))


# --- problem --------------------------------------------------------------------------------------

def settings(b, ncoils, order, length, per_coil_length, kappa, msc, dcc, dcs, nq=None, grid_dist=None,
             size_classes=False):
    """Problem settings (the dict make_problem uses). ncoils: base coils per half period; order: Fourier order;
    length [m]: per coil (per_coil_length=True) or of all base coils together; kappa [1/m]: maximum curvature;
    msc [1/m^2]: maximum mean squared curvature; dcc, dcs [m]: minimum coil-coil and coil-surface distances;
    nq: points per coil (default max(128, 16 order, a multiple of 32 >= L_coil / (0.06 min(dcc, dcs))), at most
    2048); grid_dist: (nphi, ntheta) of the distance grid (default from the boundary size and dcs).
    size_classes: round nq and the distance grid up to the classes of fixed/size_classes.py, so that problems of
    the same structure share compiled programs (for runs over many surfaces)."""
    from xcoil.fixed import size_classes as sc
    L_coil = length if per_coil_length else length / ncoils
    if nq is None:
        nq = max(128, 16 * order, 32 * math.ceil(L_coil / (0.06 * min(dcc, dcs)) / 32))
        nq = min(nq, 2048)
    if grid_dist is None:
        nphi = 2 * b.nfp * math.ceil(4 * 2 * math.pi * r_max(b) / dcs / (2 * b.nfp))
        grid_dist = (min(max(nphi, 128), 768), 64)
    if size_classes:
        nq, grid_dist = sc.nq_class(nq), sc.grid_class(grid_dist, b.nfp)
    return dict(n_coils=int(ncoils), order=int(order), nq_opt=int(nq), kappa_lim=float(kappa), msc_lim=float(msc),
                dcc=float(dcc), dcs=float(dcs), grid_dist=tuple(grid_dist), eps=float(length),
                per_coil_length=bool(per_coil_length))


def make_problem(b, key, arc=None, grid=None, **kw):
    """CoilProblem for boundary b (saved as <XCOIL_USER_DIR>/<key>.npz, default ~/.cache/xcoil/boundaries, and
    registered as a device).
    arc: overrides of the arclength-penalty settings ARC (arc_pen_thresh [m^2], arc_pen_weight,
    arc_pen_intervals), e.g. rescaled for a device of another size; grid: flux grid of the problem
    (default: the config's grid_fine). kw: settings() arguments (ncoils, order, length, per_coil_length, kappa, msc,
    dcc, dcs, nq, grid_dist, size_classes). Returns (prob, settings dict)."""
    from xcoil.explore.problem import CoilProblem
    from xcoil.homotopy.devices import DEVICES, Device
    s = settings(b, **kw)
    name = f"user/{key}"
    bd.save(bd.from_modes(b.xm, b.xn, b.rc, b.zs, b.nfp, True, b.rs, b.zc, name), name)
    DEVICES[f"user_{key}"] = Device(name, s["n_coils"], True, s["kappa_lim"], s["msc_lim"], s["dcc"], s["dcs"],
                                    s["eps"], s["eps"], s["order"], s["grid_dist"], nq_opt=s["nq_opt"],
                                    per_coil_length=s["per_coil_length"], reference="user surface")
    return CoilProblem(f"user_{key}", s["eps"], grid=grid, grid_dist=s["grid_dist"],
                       per_coil_length=s["per_coil_length"], **dict(ARC, **(arc or {}))), s


# --- starting coils -------------------------------------------------------------------------------

INITS = {"circle": "circle", "planar": "offset_planar", "3d": "offset_3d"}
SIZES = ("clear", "L70", "L85", "L100", "half-R0")


def initial_coils(prob, init="3d", size="clear", value=None, tilt=0.0):
    """One starting coil set.
      init   "circle": planar circles of radius R1 about the axis point (R00, 0) in each coil plane
                       (as simsopt's create_equally_spaced_curves);
             "planar": planar offsets, by a distance d, of the toroidally averaged cross-section;
             "3d":     surface-following offsets X(theta, phi_c) + d n(theta, phi_c) of the boundary.
      size   "clear" (smallest size with all coils 1.1 d_cs from the surface), "L70" / "L85" / "L100"
             (70 / 85 / 100 % of the length budget), "half-R0" (circles: R1 = R00 / 2, as
             auglag_qa.py); or value: an explicit d [m] (planar, 3d) or R1 [m] (circle).
      tilt   rotation of every coil plane about its e_R axis [deg].
    Coil planes at phi_c = (c + 1/2) pi / (nfp C). Returns (Start, check): check = admissibility of
    the coils (linked, min coil-coil distance, min coil-surface clearance, total base-coil length)."""
    from xcoil.curves import symmetry as sym_mod
    from xcoil.fixed import starts as stm
    b, C, order = prob.b, prob.cfg.n_coils, prob.cfg.order
    phis = sym_mod.base_toroidal_angles(C, b.nfp, True)
    fun = {"circle": lambda v: stm.circle_points(b, phis, v), "planar": lambda v: stm.offset_planar_points(b, phis, v),
           "3d": lambda v: stm.offset_3d_points(b, phis, v)}[init]
    rc10 = abs(float(b.rc[(b.xm == 1) & (b.xn == 0)][0]))
    lo = 0.5 * rc10 if init == "circle" else 0.2 * prob.lim.dcs
    if value is None:
        if size == "half-R0":
            if init != "circle":
                raise ValueError("size 'half-R0' is for circular coils")
            value = 0.5 * b.R00
        elif size == "clear":
            value = stm._size_for_clearance(fun, b, prob.lim.dcs, lo, 4 * rc10)
        elif size in ("L70", "L85", "L100"):
            value = stm._size_for_length(fun, int(size[1:]) / 100 * start_cap(prob), lo, 4 * rc10)
            if value is None:
                raise ValueError(f"even the smallest {init} coils are longer than {size} of the length budget")
        else:
            raise ValueError(f"unknown size {size!r}: one of {SIZES} or an explicit value")
    pname = "R1" if init == "circle" else "d"
    P = fun(value)
    if tilt:
        P = stm._tilt(P, phis, tilt)
    x = np.concatenate([stm._fit(P, order), np.zeros(C - 1)])
    label = f"{init}_{size or pname + '=' + format(value, 'g')}" + (f"_tilt{tilt:+g}" if tilt else "")
    st = stm.Start(INITS[init], label, {pname: float(value), **({"tilt_deg": tilt} if tilt else {})}, x, P)
    check = dict(linked=bool(stm._linked(P, b.nfp, True, n=256)), d_cc=stm._min_coil_coil(P, b.nfp, True),
                 d_cs=stm._min_clearance(b, P), length=stm._length(P))
    return st, check



def start_cap(prob):
    """Total base-coil length used to size the starting coils (the per-coil cap times C when per_coil_length)."""
    return prob.eps * (prob.cfg.n_coils if prob.lim.per_coil_length else 1)


def warm_start(start, x, label="warm"):
    """A start at the design vector x (e.g. a previous result's r["x"]) with the metadata of `start`: pass it to
    solve_all to continue from a design (warm start), e.g. after changing a limit."""
    import dataclasses
    return dataclasses.replace(start, x=np.asarray(x, float), label=label)

def starts(prob, families=("offset_3d", "offset_planar", "circle")):
    """Every admissible start of the given families (sizes clear, L70, L85, L100, L85 tilted +-8 deg;
    linked or overlapping starts shrunk or dropped)."""
    from xcoil.fixed.starts import generate
    out = generate(prob.b, prob.cfg.n_coils, prob.cfg.order, start_cap(prob), prob.lim.dcs, families=families,
                   admissible_dcc=prob.lim.dcc, unlinked=True)
    for s in out:
        s.label = {"offset_3d": "3d", "offset_planar": "planar", "circle": "circle"}[s.family] + "_" + s.label
    return out


# first admissible start of each family, in this order
PAIR_ORDER = {"planar": ["planar_L100", "planar_clear", "planar_L70", "planar_L85", "planar_L85_tilt+8", "planar_L85_tilt-8"],
              "3d": ["3d_L100", "3d_L85", "3d_clear", "3d_L70", "3d_L85_tilt-8", "3d_L85_tilt+8"]}


def pair(prob):
    """Two starts, one per family: a planar offset and a surface-following (3D) offset - the families
    fail on different devices (e.g. planar on the QH reactor)."""
    by = {s.label: s for s in starts(prob, ("offset_3d", "offset_planar"))}
    out = []
    for fam, order in PAIR_ORDER.items():
        c = [by[k] for k in order if k in by] or [s for k, s in by.items() if k.startswith(fam + "_")]
        if c:
            out.append(c[0])
    return out


# --- quick check, solve, certify ------------------------------------------------------------------

def _quick_fn(x, V, T, eps, off):
    """quick_check's metrics as one program of (x, problem data): shared by every problem of the
    same structure (jitted once in _QUICK)."""
    from xcoil.field import mixed as mx
    from xcoil.curves import fourier as fr, symmetry as sym_mod
    from xcoil.constraints import pointwise as pw
    from xcoil.homotopy import surface_path as sp, problem_h as ph
    from xcoil.fixed.device_lm import _linking_fn
    C = T.sym.C
    S = sp.eval_xyz(T.dist, 1.0)
    sub = max(1, T.cb.B0.shape[0] // 256)
    mask = np.stack([(np.asarray(T.sym.coil) == c) & np.asarray(T.sym.ident) for c in range(C)])
    links = _linking_fn(T)

    def d2(A, B):                                   # squared distances (n, m), no (n, m, 3) tensor
        return jnp.sum(A * A, 1)[:, None] + jnp.sum(B * B, 1)[None] - 2.0 * A @ B.T

    B = mx.field_on(x, V.cbq, V.sym, V.X, V.I0, None)
    q = jnp.abs(jnp.sum(B * V.n, 1)) / jnp.linalg.norm(B, axis=1)
    g, gd, gdd = fr.coils_curves(x, T.cb, C)
    gs = g[:, ::sub]
    G = sym_mod.expand(gs, T.sym)                                            # (J, n, 3)
    n = gs.shape[1]
    dcc = jnp.stack([jnp.min(jnp.where(jnp.repeat(jnp.asarray(mask[c]), n)[None], jnp.inf,
                                       d2(gs[c], G.reshape(-1, 3)))) for c in range(C)])
    dcs = jnp.min(jnp.stack([jnp.min(d2(gs[c], S)) for c in range(C)]))
    dcc, dcs = jnp.sqrt(jnp.maximum(dcc, 0.0)), jnp.sqrt(jnp.maximum(dcs, 0.0))
    p = (jnp.asarray(1.0), eps)
    r = ph.resid(T, x, p)
    return dict(mean_bn_pct=100 * jnp.sum(V.w * q) / jnp.sum(V.w), max_bn_pct=100 * jnp.max(q),
                L=pw.lengths(gd), kappa=jnp.max(pw.curvature(gd, gdd), axis=1), msc=pw.msc(gd, gdd),
                d_cc=jnp.min(dcc), d_cs=dcs, linked=jnp.any(links(x) != 0),
                v=jnp.maximum(0.0, jnp.max(ph.cons(T, x, p) - off)), F=(r @ r) / T.cfg.f_ref)


_QUICK = jax.jit(_quick_fn)


def quick_check(prob, x):
    """Cheap, uncertified metrics of x: mean / max |B.n|/|B| [%] on the certification grid (no fine
    grid for the maximum), objective F, largest true-limit violation v (normalized constraint rows),
    per-coil lengths, max curvature and MSC at the optimization quadrature, point-sampled minimum
    coil-coil and coil-surface distances, and whether any coil pair is linked. One compiled program
    for all problems of the same structure (the problem's data are its inputs)."""
    if not hasattr(prob, "_quick_off"):
        from xcoil.fixed.device_lm import true_violation_offsets
        prob._quick_off = true_violation_offsets(prob)
    r = {k: np.asarray(v) for k, v in _QUICK(jnp.asarray(x, jnp.float64), prob.V, prob.T,
                                             jnp.asarray(prob.eps), prob._quick_off).items()}
    out = {k: float(v) if v.ndim == 0 else v.tolist() for k, v in r.items()}
    out["linked"] = bool(r["linked"])
    return out


def _rank(r):
    q = r["quick"]
    return (q["linked"], q["v"] > VIOL_TOL, q["F"])


def screen_key(q):
    """Screening score of a start after the first stage of its solve (smaller is better): unlinked
    first, then the quick check's mean |B.n|/|B|."""
    return (bool(q["linked"]), q["mean_bn_pct"])


def solve_all(prob, start_list, precision="mixed", target_mean=None, link_test=True, certify="best", warmup=0, log=print,
              screen=None, max_outer=60, compile_first=True, repair=True, max_force=None,
              ordering=None, lp_curvature=None):
    """Solve from every start, then certify (certify "all" | "best" | "none", see the module
    docstring). target_mean (percent) stops each solve at the first outer iteration whose
    certification-grid mean meets it; otherwise the general rule. max_outer caps the outer
    iterations of every start. compile_first=False: no separate compile solve (t_compile 0; the
    first solve's time then includes any compilation - for runs over many surfaces, whose
    programs are shared and mostly compiled already).
    screen=k: screen, then finish - run only the first stage of every start's solve (the first
    subproblem, on the coarse grids), rank the starts by screen_key, and finish only the k best;
    the others are reported with screened_out=True (their first-stage result, not certified).
    repair: a certified-run design that fails only on a distance limit is re-solved with a distance margin sized from
    the shortfall (device_lm.repair_distances).
    max_force (opt-in): dict(F_lim=..., radius=..., margin=0.0, nq=768), the maximum coil force per unit length F_lim
    [N/m] with conductor radius `radius` (constraints/forces.py): rows F / F_lim - (1 - margin) <= 0 on every base coil
    in the solve, and the certificate's "force" entry (max per coil on nq points; certified only if within F_lim).
    ordering (opt-in): dict(gap_deg=1.0, kappa=20.0, beta=500.0, n_seg=16, axis=None), the toroidal ordering rows
    (constraints/ordering.py) about a fixed reference axis (default: the boundary's geometric axis), and the
    certificate's "ordering" entry (certified only if no pair crosses and every gap reaches gap_deg).
    lp_curvature (opt-in): dict(J_max=..., p=2.0, threshold=0.0, nq=4096), the Lp-curvature rows J_c / J_max - 1
    <= 0 per coil (constraints/lp_curvature.py) and the certificate's "lp_curvature" entry (on nq points).
    The distance repair re-solves with these rows too.
    Returns (results, compile s)."""
    from xcoil.fixed.device_lm import build_solver, make_monitor
    from xcoil.fixed.device_lm import MAIN
    budget = dict(MAIN, precision=precision, link_test=link_test)
    tgt = None if target_mean is None else (float(target_mean), None)
    extras = []
    if ordering:
        from xcoil.constraints import ordering as od
        oc = dict(dict(gap_deg=1.0, kappa=20.0, beta=500.0, n_seg=16, axis=None), **ordering)
        extras.append(od.ordering_extra(prob, oc["axis"], oc["gap_deg"], oc["kappa"], oc["beta"], oc["n_seg"]))
    if lp_curvature:
        from xcoil.constraints import lp_curvature as lpc
        lc = dict(dict(p=2.0, threshold=0.0, nq=4096), **lp_curvature)
        extras.append(lpc.lp_extra(prob, lc["J_max"], lc["p"], lc["threshold"]))
    if max_force:
        from xcoil.constraints import forces as fo
        mf = dict(dict(margin=0.0, nq=768), **max_force)
        extras.append(fo.force_extra(prob, mf["F_lim"], mf["radius"], mf["margin"]))
    extra = _join_extras(extras)
    solve = build_solver(prob, budget, target=tgt, monitor=make_monitor(prob) if tgt else None, max_outer=max_outer,
                         extra=extra)
    t_compile = 0.0
    if compile_first:
        t0 = time.perf_counter(); jax.block_until_ready(solve(start_list[0].x)); t_compile = time.perf_counter() - t0
    for _ in range(warmup):
        jax.block_until_ready(solve(start_list[0].x))
    res = []

    def report(s, t, x, info, extra=None):
        q = quick_check(prob, x)
        res.append(dict(start=s.label, params=s.params, t=t, x=np.asarray(x), quick=q, cert=None,
                        info={k: np.asarray(v).tolist() for k, v in info.items()}, **(extra or {})))
        log(f"  {s.label:18s} {t:7.2f} s  outer {int(info['outer']):2d}  mean {q['mean_bn_pct']:.4f} %  "
            f"L {sum(q['L']):.3f}  kappa {max(q['kappa']):.3f}  MSC {max(q['msc']):.3f}  d_cc {q['d_cc']:.4f}  "
            f"d_cs {q['d_cs']:.4f}  v {q['v']:.1e}" + ("  LINKED" if q["linked"] else ""))
    if screen:
        first = []
        for s in start_list:
            t0 = time.perf_counter()
            with solve.precision:
                c = solve.stage1(jnp.asarray(s.x, jnp.float64)); jax.block_until_ready(c)
            first.append((s, c, time.perf_counter() - t0, quick_check(prob, np.asarray(c[0]))))
        first.sort(key=lambda f: screen_key(f[3]))
        log(f"  screening: first stage of {len(first)} starts in {sum(f[2] for f in first):.2f} s; finishing "
            + ", ".join(f[0].label for f in first[:screen]))
        for s, c, t1, q in first[:screen]:
            t0 = time.perf_counter()
            with solve.precision:
                c = solve.stage2(c); jax.block_until_ready(c)
            report(s, t1 + time.perf_counter() - t0, c[0], dict(outer=c[8], stopped=c[9]))
        for s, c, t1, q in first[screen:]:
            res.append(dict(start=s.label, params=s.params, t=t1, x=np.asarray(c[0]), quick=q, cert=None,
                            info=dict(outer=1), screened_out=True))
    else:
        for s in start_list:
            t0 = time.perf_counter(); x, info = solve(s.x); jax.block_until_ready(x); t = time.perf_counter() - t0
            report(s, t, x, info)
    done = [r for r in res if not r.get("screened_out")]
    order = done if certify == "all" else sorted(done, key=_rank) if certify == "best" else []
    def full_cert(x):
        """The certificate with the opt-in limits' entries (certified only if all of them are ok)."""
        c = prob.certify(x)
        if max_force:
            c["force"] = fo.certify_force(x, prob, mf["F_lim"], mf["radius"], mf["nq"])
        if lp_curvature:
            c["lp_curvature"] = lpc.certify_lp(x, prob, lc["J_max"], lc["p"], lc["threshold"], lc["nq"])
        if ordering:
            c["ordering"] = od.certify_ordering(x, prob, oc["axis"], oc["kappa"], oc["gap_deg"])
        c["certified"] = bool(c["certified"] and all(c[k]["ok"] for k in ("force", "lp_curvature", "ordering")
                                                     if k in c))
        return c

    for r in order:
        t0 = time.perf_counter(); r["cert"] = full_cert(r["x"]); r["t_cert"] = time.perf_counter() - t0
        if repair and not r["cert"]["certified"]:     # certificate feedback on the distance limits
            from xcoil.fixed.device_lm import repair_distances, distance_shortfall, build_solver
            t0 = time.perf_counter()
            if extra is None:
                x2, c2, m = repair_distances(prob, r["x"], r["cert"], budget, max_outer=max_outer)
            else:                                     # the same repair, with the opt-in rows in the re-solve
                x2, c2, m = r["x"], r["cert"], None
                s = distance_shortfall(prob, r["cert"])
                if s > 0 and not r["cert"].get("linked"):
                    m = round(1.5 * s + 2e-3, 4)
                    b = {k: v for k, v in budget.items() if k not in ("coarse_grid", "coarse_dist")}
                    b.update(inner_first=int(b.get("inner", 50)), first_prog=0.0, rho0=1e3, dist_margin=m)
                    xr, _ = build_solver(prob, b, max_outer=max_outer, extra=extra)(np.asarray(r["x"]))
                    cr = full_cert(np.asarray(xr))
                    if cr["certified"] or cr["mean_bn_pct"] < r["cert"]["mean_bn_pct"]:
                        x2, c2 = np.asarray(xr), cr
                    else:
                        m = None
            if m is not None:
                r["x"], r["cert"], r["repair_margin"] = x2, c2, m
                r["t_repair"] = time.perf_counter() - t0
                log(f"  repair {r['start']:18s} distance margin {m:.4f} ({r['t_repair']:.1f} s)")
        c = r["cert"]
        log(f"  certificate {r['start']:18s} {'CERTIFIED' if c['certified'] else ('linked' if c['linked'] else 'not certified')}"
            f"  mean {c['mean_bn_pct']:.4f} %  max {c['max_bn_pct']:.3f} %  ({r['t_cert']:.1f} s)")
        if certify == "best" and c["certified"]:
            break
    return res, t_compile


def _join_extras(extras):
    """One device_lm.Extra with the constraint rows of several (None: no extra rows)."""
    if not extras:
        return None
    if len(extras) == 1:
        return extras[0]
    from xcoil.fixed.device_lm import Extra
    g = lambda x: jnp.concatenate([e.g(x) for e in extras])
    jg = lambda x: jnp.concatenate([e.jg(x) for e in extras], 0)
    return Extra(lambda x: jnp.zeros((0,), x.dtype), g, jr=lambda x: jnp.zeros((0, x.shape[0]), x.dtype), jg=jg)


def best(results):
    """The certified result with the smallest mean, else the best finished one by the quick check."""
    cert = [r for r in results if r["cert"] and r["cert"]["certified"]]
    if cert:
        return min(cert, key=lambda r: r["cert"]["mean_bn_pct"])
    return min([r for r in results if not r.get("screened_out")] or results, key=_rank)


# --- output ---------------------------------------------------------------------------------------

def force_limit_in_problem(prob, F_user, total_current):
    """A coil-force limit F_user [N/m] meant for coils whose base coils carry `total_current` [A] together, converted
    to the solver's current normalization (base coils at |I0| = 1e6 A each): |dF/dl| scales as I^2, so
    F_lim = F_user (C |I0| / total_current)^2. Pass the result as solve_all(max_force=dict(F_lim=...))."""
    C, I0 = prob.T.sym.C, abs(float(prob.T.I0))
    return float(F_user) * (C * I0 / float(total_current)) ** 2


def force_to_user(prob, F_problem, total_current):
    """Inverse of force_limit_in_problem: forces [N/m] of the solver's normalization at `total_current` [A]."""
    C, I0 = prob.T.sym.C, abs(float(prob.T.I0))
    return float(F_problem) * (float(total_current) / (C * I0)) ** 2


def physical_coils(prob, x, total_current=None, n=256):
    """(points (Nphys, n, 3) [m], currents (Nphys,) [A] with the symmetry sign, base currents (C,))."""
    from xcoil.curves import fourier as fr, symmetry as sym_mod, currents as cur
    T = prob.T
    cb = fr.make_basis(prob.cfg.basis, n, prob.cfg.order)
    x = jnp.asarray(np.asarray(x, float))
    P = np.asarray(sym_mod.expand(fr.coils_curves(x, cb, T.sym.C)[0], T.sym))
    Ib = np.asarray(cur.base_currents(x, T.cb, T.sym.C, T.I0))
    scale = 1.0 if total_current is None else total_current / Ib.sum()
    return P, np.asarray(T.sym.sign) * Ib[np.asarray(T.sym.coil)] * scale, Ib * scale


def write_coils_json(prob, x, path, total_current=None, extra=None):
    """Coils as JSON: base curves as CurveXYZFourier coefficients in simsopt's order (for each of x,
    y, z: c0, s1, c1, ..., sM, cM), base currents [A], and the symmetry (nfp copies, each with its
    stellarator-flipped mirror of opposite current - simsopt's coils_via_symmetries order)."""
    C, M = prob.cfg.n_coils, prob.cfg.order
    K = 3 * (2 * M + 1)
    _, _, Ib = physical_coils(prob, x, total_current)
    x = np.asarray(x, float)
    json.dump(dict(format="xcoil coils v1", nfp=prob.b.nfp, stellsym=True, order=M,
                   dof_order="per coil: x, y, z blocks, each [c0, s1, c1, ..., sM, cM] (simsopt CurveXYZFourier)",
                   symmetry="for k in range(nfp): for flip in (False, True): for each base coil "
                            "(rotation 2 pi k / nfp; flip (x, y, z) -> (x, -y, -z) with opposite current)",
                   base_coils=[dict(dofs=x[c * K:(c + 1) * K].tolist(), current=float(Ib[c])) for c in range(C)],
                   **(extra or {})), open(path, "w"), indent=1)


def write_makegrid(prob, x, path, total_current=None, n=256):
    """All physical coils in the MAKEGRID coils-file format (points [m], current [A] per point)."""
    P, I, _ = physical_coils(prob, x, total_current, n)
    lines = [f"periods {prob.b.nfp}", "begin filament", "mirror NIL"]
    for j, (pts, cur) in enumerate(zip(P, I)):
        for p in pts:
            lines.append(f"{p[0]:.12e} {p[1]:.12e} {p[2]:.12e} {cur:.12e}")
        p = pts[0]
        lines.append(f"{p[0]:.12e} {p[1]:.12e} {p[2]:.12e} {0.0:.12e} 1 coil{j + 1}")
    open(path, "w").write("\n".join(lines + ["end", ""]))


def write_vtk(prob, x, prefix):
    """prefix_surface.vts (full torus StructuredGrid: B.n/|B|, |B|, B.n) and prefix_coils.vtu (all
    physical coils as closed polylines: base-coil index and current per coil and per point)."""
    from xcoil.fixed.export_vtk import Exporter
    if not hasattr(prob, "_exporter"):
        prob._exporter = Exporter(prob)
    return prob._exporter(x, prefix, fmt="xml")


def write_simsopt(prob, x, path, total_current=None):
    """simsopt BiotSavart JSON of the coils, if simsopt is installed (returns False otherwise)."""
    try:
        from simsopt.geo import CurveXYZFourier
        from simsopt.field import Current, coils_via_symmetries, BiotSavart
    except ImportError:
        return False
    C, M = prob.cfg.n_coils, prob.cfg.order
    K = 3 * (2 * M + 1)
    _, _, Ib = physical_coils(prob, x, total_current)
    curves = []
    for c in range(C):
        cv = CurveXYZFourier(prob.cfg.nq_opt, M)
        cv.x = np.asarray(x, float)[c * K:(c + 1) * K]
        curves.append(cv)
    BiotSavart(coils_via_symmetries(curves, [Current(float(i)) for i in Ib], prob.b.nfp, True)).save(path)
    return True
