"""xcoil-coils: coils for a target plasma boundary, from the command line.

Stage-two coil optimization with XCoil: given a target plasma boundary, find
stellarator-symmetric coils (shapes and currents, total current fixed) whose field is tangent to
it, under hard limits on coil length, coil-coil and coil-surface distance, pointwise curvature and
mean squared curvature (MSC). In the spirit of simsopt's auglag_qa.py example, whose problem is
the default here, but self-contained: it needs xcoil, jax, numpy and scipy, not simsopt.

XCoil solves an augmented Lagrangian on the squared normal field, with a Levenberg-Marquardt
inner solver on the Gauss-Newton normal equations, compiled with JAX (CPU or GPU), from starting
coils built from the boundary alone. Results can be certified in double precision: dense flux
grids (the maximum also on a 3x finer grid), exact minimum distances, curvature from 4096 samples,
Gauss linking numbers.

Usage (docs/cli.md and docs/tutorials):
    xcoil-coils                                               # the problem of simsopt's auglag_qa.py
    xcoil-coils --init circle --size half-R0                  # auglag_qa.py's initial coils
    xcoil-coils --surface input.my_device --ncoils 5 --order 6 --length 30 --total-length
    xcoil-coils --surface w7x --length 38 --total-length --cc 0.3 --cs 0.35 --curvature 2.5 --msc 1.5 \\
                --order 12 --init all
    xcoil-coils --help
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------
# Parameters (the defaults are the problem of simsopt's auglag_qa.py; all can be changed on the
# command line, see --help)
# ---------------------------------------------------------------------------------------------
SURFACES = ["landreman_paul_qa"]  # target surface file(s) or bundled name(s)
NCOILS = 4                  # base coils per half period
ORDER = 4                   # Fourier order of each coil
LENGTH_TARGET = 4           # length limit [m]: per coil (PER_COIL_LENGTH) or of all base coils together
PER_COIL_LENGTH = True
CC_THRESHOLD = 0.1          # minimum coil-coil distance [m]
CS_THRESHOLD = 0.15         # minimum coil-surface distance [m]
CURVATURE_THRESHOLD = 3     # maximum curvature [1/m], pointwise
MSC_THRESHOLD = 4           # maximum mean squared curvature [1/m^2], per coil
INIT = "3d"                 # initial coils: "circle", "planar", "3d" or "all"
SIZE = "clear"              # "clear", "L70", "L85", "L100" (share of the length budget) or "half-R0" (circles)
OFFSET = None               # explicit offset distance d [m] for "planar" / "3d" (overrides SIZE)
RADIUS = None               # explicit circle radius R1 [m] for "circle" (overrides SIZE)
TILT = 0.0                  # rotation of the coil planes about their radial axis [deg]
CERTIFY = "best"            # "best", "all" or "none"
TARGET_MEAN = None          # e.g. 0.6: stop at the first iterate with mean |B.n|/|B| <= 0.6 % (else: converge)
PRECISION = "mixed"         # "mixed" (FP64 geometry, FP32 Biot-Savart kernels, FP64 accumulation) or "fp64"
TOTAL_CURRENT = 3e5         # sum of the base-coil currents [A] in the outputs (the flux error does not depend on it)
OUT_DIR = "./output_xcoil/"


def build_parser():
    """The argparse parser of xcoil-coils (every option is documented in docs/reference.md)."""
    ap = argparse.ArgumentParser(prog="xcoil-coils", description=__doc__.split("Usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surface", nargs="+", default=SURFACES,
                    help="target surface(s): VMEC input (input.*), wout (*.nc), .npz, a bundled name (w7x, "
                         "landreman_paul_qa, qh_reactor, cfqs_2b40, rotating_ellipse) or, with simsopt installed, a simsopt JSON file")
    ap.add_argument("--ncoils", type=int, default=NCOILS, help="base coils per half period")
    ap.add_argument("--order", type=int, default=ORDER, help="Fourier order of the coils")
    ap.add_argument("--length", type=float, default=LENGTH_TARGET, help="length limit [m]")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-coil", dest="per_coil", action="store_true", default=PER_COIL_LENGTH, help="length limit per coil (default)")
    g.add_argument("--total-length", dest="per_coil", action="store_false", help="length limit on all base coils together")
    ap.add_argument("--cc", type=float, default=CC_THRESHOLD, help="minimum coil-coil distance [m]")
    ap.add_argument("--cs", type=float, default=CS_THRESHOLD, help="minimum coil-surface distance [m]")
    ap.add_argument("--curvature", type=float, default=CURVATURE_THRESHOLD, help="maximum curvature [1/m]")
    ap.add_argument("--msc", type=float, default=MSC_THRESHOLD, help="maximum mean squared curvature [1/m^2]")
    ap.add_argument("--init", default=INIT, choices=["circle", "planar", "3d", "pair", "all"], help="initial coils "
                    "(pair: one planar and one 3D start)")
    ap.add_argument("--size", default=SIZE, choices=["clear", "L70", "L85", "L100", "half-R0"],
                    help="size of the initial coils (ignored with --offset / --radius)")
    ap.add_argument("--offset", type=float, default=OFFSET, help="offset distance d [m] (planar, 3d)")
    ap.add_argument("--radius", type=float, default=RADIUS, help="circle radius R1 [m] (circle)")
    ap.add_argument("--tilt", type=float, default=TILT, help="tilt of the coil planes [deg]")
    ap.add_argument("--certify", default=CERTIFY, choices=["best", "all", "none"],
                    help="best: certify in order of the quick check until one passes; all: every result; none: quick check only")
    ap.add_argument("--target-mean", type=float, default=TARGET_MEAN, help="stop at mean |B.n|/|B| <= this [%%]")
    ap.add_argument("--precision", default=PRECISION, choices=["mixed", "fp64"],
                    help="mixed: FP64 geometry, FP32 Biot-Savart kernels, FP64 accumulation (default, fast); fp64: all FP64")
    ap.add_argument("--nq", type=int, default=None, help="points per coil (default: a rule resolving the distance "
                    "limits, nq >= L_coil / (0.06 min(cc, cs)); smaller is faster, e.g. 128 for the default problem)")
    ap.add_argument("--total-current", type=float, default=TOTAL_CURRENT, help="sum of base-coil currents in the outputs [A]")
    ap.add_argument("--out", default=OUT_DIR, help="output directory (one subfolder per surface)")
    ap.add_argument("--no-link-test", dest="link_test", action="store_false", help="allow steps that change coil linking")
    ap.add_argument("--screen", type=int, default=None, help="with several starts (--init pair / all): run only the "
                    "first stage of every start, then finish the SCREEN best (e.g. 1)")
    ap.add_argument("--max-outer", type=int, default=60, help="outer iterations per start at most")
    ap.add_argument("--max-force", type=float, default=None, help="opt-in: maximum coil force per unit length [N/m] "
                    "(with --total-current as the coils' total current); certified on 768 points per coil")
    ap.add_argument("--force-radius", type=float, default=0.02, help="conductor radius of the force's self-field "
                    "regularization [m]")
    ap.add_argument("--force-margin", type=float, default=0.005, help="relative margin of the force rows in the solve "
                    "(covers the gap between the solver's points and the certificate's 768)")
    ap.add_argument("--lp-curvature-max", type=float, default=None, help="opt-in: limit J_max on each coil's Lp "
                    "curvature J = (1/p) int max(kappa - threshold, 0)^p dl")
    ap.add_argument("--lp-p", type=float, default=2.0, help="exponent p of the Lp curvature")
    ap.add_argument("--lp-threshold", type=float, default=0.0, help="curvature threshold of the Lp curvature [1/m]")
    ap.add_argument("--ordering", action="store_true", help="opt-in: toroidal ordering rows (neighbouring coils may "
                    "not cross in the (phi, theta) projection about the boundary's geometric axis)")
    ap.add_argument("--ordering-gap", type=float, default=1.0, help="minimum toroidal gap of the ordering rows [deg]")
    return ap


def parse(argv=None):
    return build_parser().parse_args(argv)


def report(tag, q):
    print(f"{tag}: mean |B.n|/|B| {q['mean_bn_pct']:.4f} %, max {q['max_bn_pct']:.3f} % (certification grid); "
          f"objective {q['F']:.3e}; largest limit violation {q['v']:.1e}")
    print(f"{tag}: lengths {[round(v, 4) for v in q['L']]} (sum {sum(q['L']):.4f}); max curvature "
          f"{[round(v, 3) for v in q['kappa']]}; MSC {[round(v, 3) for v in q['msc']]}")
    print(f"{tag}: min coil-coil {q['d_cc']:.4f} m, min coil-surface {q['d_cs']:.4f} m, "
          f"{'LINKED coils' if q['linked'] else 'no linked coils'}")


def run_surface(spec, a):
    import jax
    jax.config.update("jax_enable_x64", True)
    from xcoil.fixed import user_problem as up

    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(spec).name if os.path.exists(spec) else spec).strip("_")
    out = os.path.join(a.out, key + f"_ncoils{a.ncoils}_order{a.order}_length{a.length:g}{'pc' if a.per_coil else 'tot'}_"
                       f"cc{a.cc:g}_cs{a.cs:g}_curv{a.curvature:g}_msc{a.msc:g}") + "/"
    os.makedirs(out, exist_ok=True)
    print("=" * 120)
    print(f"Surface: {spec}")
    b, info = up.load_boundary(spec)
    print(f"  {info['source']}, nfp {b.nfp}, {len(b.xm)} modes" + (f", refit error {info['fit_err_m']:.1e} m" if info["fit_err_m"] else ""))
    if info["fit_err_m"] > 1e-3:
        print("  WARNING: the refitted surface deviates from the input by more than 1 mm (strongly shaped surface)")
    prob, st = up.make_problem(b, key, ncoils=a.ncoils, order=a.order, length=a.length, per_coil_length=a.per_coil,
                               kappa=a.curvature, msc=a.msc, dcc=a.cc, dcs=a.cs, nq=a.nq)
    print(f"  problem: {prob.n} unknowns, {prob.m} constraint rows, {st['nq_opt']} points per coil, "
          f"distance grid {st['grid_dist'][0]}x{st['grid_dist'][1]}; device {jax.devices()[0]}")

    if a.init in ("all", "pair"):
        starts = up.starts(prob) if a.init == "all" else up.pair(prob)
        print(f"  {len(starts)} admissible starts: {', '.join(s.label for s in starts)}")
    else:
        value = a.radius if a.init == "circle" else a.offset
        st0, chk = up.initial_coils(prob, a.init, None if value is not None else a.size, value, a.tilt)
        starts = [st0]
        pname, v = next(iter(st0.params.items()))
        print(f"  initial coils {st0.label}: {pname} = {v:.4f} m, total base-coil length {chk['length']:.3f} m, "
              f"min coil-coil {chk['d_cc']:.4f} m (limit {a.cc:g}), min coil-surface {chk['d_cs']:.4f} m (limit {a.cs:g})"
              + (", LINKED" if chk["linked"] else ", unlinked"))
        if chk["linked"]:
            print("  WARNING: the initial coils are linked; the result cannot be certified. Use a smaller offset or "
                  "radius, another --init, or --init all.")
    up.write_vtk(prob, starts[0].x, out + "init")
    report(f"Initial ({starts[0].label})", up.quick_check(prob, starts[0].x))

    t0 = time.time()
    max_force = None
    if a.max_force:
        F_lim = up.force_limit_in_problem(prob, a.max_force, a.total_current)
        max_force = dict(F_lim=F_lim, radius=a.force_radius, margin=a.force_margin)
        print(f"  max coil force {a.max_force:g} N/m at total current {a.total_current:g} A "
              f"(= {F_lim:.4g} N/m in the solver's current normalization), conductor radius {a.force_radius:g} m")
    ordering = dict(gap_deg=a.ordering_gap) if a.ordering else None
    lp = dict(J_max=a.lp_curvature_max, p=a.lp_p, threshold=a.lp_threshold) if a.lp_curvature_max else None
    results, t_compile = up.solve_all(prob, starts, a.precision, a.target_mean, a.link_test, a.certify,
                                      screen=a.screen if len(starts) > 1 else None, max_outer=a.max_outer,
                                      max_force=max_force, ordering=ordering, lp_curvature=lp)
    t_all = time.time() - t0
    r = up.best(results)
    report(f"Final ({r['start']})", r["quick"])
    c = r["cert"]
    print("-" * 120)
    if c is None:
        print(f"Best start {r['start']} (by the quick check): NOT certified (--certify none)")
    else:
        print(f"Best start {r['start']}: {'CERTIFIED' if c['certified'] else 'NOT certified'}  mean |B.n|/|B| = "
              f"{c['mean_bn_pct']:.4f} %, max = {c['max_bn_pct']:.4f} %; length {c['L']:.3f} m (longest coil "
              f"{c['L_max_coil']:.3f} m), max curvature {c['max_kappa']:.3f}, max MSC {c['msc_max']:.3f}, min coil-coil "
              f"{c['d_cc']:.4f} m, min coil-surface {c['d_cs']:.4f} m, currents {c['I_rel_min']:.2f}-{c['I_rel_max']:.2f} x mean")
    if c is not None and "force" in c:
        f = c["force"]
        print(f"  coil force: max {up.force_to_user(prob, max(f['max_force']), a.total_current):.4g} N/m "
              f"(limit {a.max_force:g} N/m) {'ok' if f['ok'] else 'EXCEEDED'}")
    if c is not None and "lp_curvature" in c:
        lc = c["lp_curvature"]
        print(f"  Lp curvature: max {max(lc['lp_curvature']):.4g} (limit {lc['J_max']:g}) {'ok' if lc['ok'] else 'EXCEEDED'}")
    if c is not None and "ordering" in c:
        o = c["ordering"]
        print(f"  ordering: smallest gap {min(o['gap_window_deg']):.2f} deg, penalized gap "
              f"{min(o['gap_penalized_deg']):.2f} deg (minimum {a.ordering_gap:g}) {'ok' if o['ok'] else 'VIOLATED'}")
    n_cert = sum(bool(x["cert"] and x["cert"]["certified"]) for x in results)
    print(f"  {len(results)} start(s), {n_cert} certified (of {sum(x['cert'] is not None for x in results)} checked); "
          f"solve times {min(x['t'] for x in results):.2f}-{max(x['t'] for x in results):.2f} s "
          f"(compilation {t_compile:.0f} s, total {t_all:.0f} s)")

    up.write_vtk(prob, r["x"], out + "optimized")
    up.write_coils_json(prob, r["x"], out + "coils_optimized.json", a.total_current,
                        extra=dict(surface=spec, certified=bool(c and c["certified"])))
    up.write_makegrid(prob, r["x"], out + f"coils.{key}", a.total_current)
    simsopt_file = up.write_simsopt(prob, r["x"], out + "biot_savart_optimized.json", a.total_current)
    json.dump(dict(surface=spec, boundary=info, settings={k: list(v) if isinstance(v, tuple) else v for k, v in st.items()},
                   device=str(jax.devices()[0]), precision=a.precision, link_test=a.link_test, certify=a.certify,
                   t_compile=t_compile, best=r["start"],
                   runs=[dict(start=x["start"], params=x["params"], t=x["t"], info=x["info"], quick=x["quick"],
                              cert=x["cert"], x=x["x"].tolist()) for x in results]),
              open(out + "summary.json", "w"), indent=1)
    print("Output directory:", out, "(init_surface.vts, init_coils.vtu, optimized_surface.vts, optimized_coils.vtu, "
          "coils_optimized.json, coils." + key + ", summary.json"
          + (", biot_savart_optimized.json" if simsopt_file else "") + ")")
    return dict(surface=spec, best=r["start"], certified=bool(c and c["certified"]),
                mean=(c or r["quick"])["mean_bn_pct"], n_cert=n_cert, n=len(results), out=out)


def main(argv=None):
    """Entry point of xcoil-coils."""
    args = parse(argv)
    summary = []
    for spec in args.surface:
        try:
            summary.append(run_surface(spec, args))
        except Exception as ex:                                   # keep going with the other surfaces
            import traceback
            traceback.print_exc()
            summary.append(dict(surface=spec, error=repr(ex)))
    if len(summary) > 1:
        print("=" * 120)
        for r in summary:
            print(f"{r['surface']}: " + (f"ERROR {r['error']}" if "error" in r else
                  f"best {r['best']} {'certified' if r['certified'] else 'NOT certified'} mean {r['mean']:.4f} % "
                  f"({r['n_cert']}/{r['n']} certified) -> {r['out']}"))
    print("FINISHED")
    return 0 if all(not r.get("error") for r in summary) else 1


if __name__ == "__main__":
    sys.exit(main())
