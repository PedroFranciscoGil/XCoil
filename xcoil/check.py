"""xcoil-check: report the installation (versions, JAX devices and precision, compile cache, bundled boundaries,
optional packages) and run one tiny coil solve. Ends with OK when everything works."""
import argparse
import importlib.util
import os
import sys
import time


def build_parser():
    """The argparse parser of xcoil-check."""
    ap = argparse.ArgumentParser(prog="xcoil-check", description=__doc__)
    ap.add_argument("--no-solve", action="store_true", help="only the report, no test solve")
    return ap


def main(argv=None):
    """Entry point of xcoil-check."""
    a = build_parser().parse_args(argv)
    import jax
    import numpy as np
    import scipy
    import xcoil
    from xcoil import config
    from xcoil.homotopy import boundary as bd
    from importlib import metadata
    ok = True
    print(f"xcoil       {metadata.version('xcoil')} ({os.path.dirname(xcoil.__file__)})")
    print(f"python      {sys.version.split()[0]}")
    print(f"jax         {jax.__version__}, numpy {np.__version__}, scipy {scipy.__version__}")
    print(f"devices     {[str(d) for d in jax.devices()]}")
    print(f"x64         {bool(jax.config.jax_enable_x64)} (precision mode {config.PRECISION})")
    print(f"cache       {config.COMPILE_CACHE}")
    names = sorted(f[:-4] for f in os.listdir(bd.DATA_DIR) if f.endswith(".npz"))
    print(f"boundaries  {', '.join(names)}")
    for extra in ("simsopt", "netCDF4"):
        print(f"{extra:11s} {'installed' if importlib.util.find_spec(extra) else 'not installed (optional)'}")
    if not jax.config.jax_enable_x64:
        print("ERROR: JAX x64 is off (import xcoil before any JAX computation, or set JAX_ENABLE_X64=1)")
        ok = False
    if not a.no_solve:
        from xcoil.fixed import user_problem as up
        b, _ = up.load_boundary("landreman_paul_qa")
        prob, _ = up.make_problem(b, "xcoil_check", ncoils=2, order=2, length=8.0, per_coil_length=True, kappa=5.0,
                                  msc=10.0, dcc=0.1, dcs=0.15, nq=32)
        start, _ = up.initial_coils(prob, init="circle", size="half-R0")
        t = time.time()
        res, _ = up.solve_all(prob, [start], certify="none", max_outer=3, log=lambda *x, **k: None)
        q = res[0]["quick"]
        good = np.isfinite(q["mean_bn_pct"])
        print(f"test solve  {'ok' if good else 'FAILED'}: 2 coils, 3 outer iterations, mean |B.n|/|B| "
              f"{q['mean_bn_pct']:.3f} % ({time.time() - t:.0f} s incl. compilation)")
        ok = ok and good
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
