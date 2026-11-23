"""Tutorial 3: your own target boundary - a VMEC input file, a wout file, raw Fourier arrays, a simsopt or DESC surface
(docs/tutorials/03_own_boundary.md, docs/boundaries.md).

    python examples/tutorial_03_own_boundary.py          # a minute on a laptop CPU
"""
import os

import numpy as np

from xcoil.fixed import user_problem as up
from xcoil.homotopy import boundary as bd

HERE = os.path.dirname(os.path.abspath(__file__))

# (a) a VMEC / VMEX input file (RBC / ZBS lines and NFP are read; everything else is ignored)
b1, info1 = up.load_boundary(os.path.join(HERE, "input.LandremanPaul2021_QA"))
print("(a)", info1["source"], "nfp", b1.nfp, len(b1.xm), "modes")

# (b) raw Fourier arrays: R = sum rc cos(m theta - n nfp phi), Z = sum zs sin(m theta - n nfp phi)
xm = np.array([0, 0, 1, 1, 1]); xn = np.array([0, 1, 0, 1, -1])
rc = np.array([1.0, 0.05, 0.15, 0.02, 0.0]); zs = np.array([0.0, -0.05, 0.15, 0.02, 0.0])
b2 = bd.from_modes(xm, xn, rc, zs, nfp=3, stellsym=True, name="my_rotating_ellipse")
print("(b) R00 =", b2.R00, "nfp", b2.nfp)

# (c) write it as a VMEC input file and read it back (the file is a plain RBC / ZBS list)
up.write_vmec_input(b2, "input.my_rotating_ellipse")
b3, info3 = up.load_boundary("input.my_rotating_ellipse")
print("(c)", info3["source"], "max |dR|", float(np.max(np.abs(np.asarray(b3.eval_rz(0.3, 0.2)) - np.asarray(b2.eval_rz(0.3, 0.2))))))

# (d) wout files: up.load_boundary("wout_x.nc") takes the last closed flux surface (netCDF3 via scipy; netCDF4 with
#     the netcdf extra). simsopt surfaces / JSON files (including QUASR devices) and DESC: see docs/boundaries.md.

# solve on the new boundary: 3 coils per half period, sizes scaled to its minor radius
prob, st = up.make_problem(b3, "tutorial3", ncoils=3, order=4, length=1.6, per_coil_length=True, kappa=10.0,
                           msc=30.0, dcc=0.05, dcs=0.08, nq=64)
start, chk = up.initial_coils(prob, init="3d", size="clear")
r = up.best(up.solve_all(prob, [start])[0])
print(f"certified {r['cert']['certified']}, mean |B.n|/|B| {r['cert']['mean_bn_pct']:.3f} %")
