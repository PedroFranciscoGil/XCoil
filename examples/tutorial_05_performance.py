"""Tutorial 5: performance - compile once and reuse, precision, points per coil, GPU, compile cache
(docs/tutorials/05_performance.md).

    python examples/tutorial_05_performance.py           # a few minutes on a laptop CPU
"""
import jax

from xcoil.fixed import user_problem as up

print("JAX devices:", jax.devices())
b, _ = up.load_boundary("landreman_paul_qa")
kw = dict(ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15,
          grid_dist=(128, 64))
quiet = lambda *a, **k: None

# solve_all first solves once to compile (t_compile = that first solve, compilation included), then solves again:
# the difference is the compile time
for nq, prec in ((64, "mixed"), (64, "fp64"), (128, "mixed")):
    prob, _ = up.make_problem(b, f"tutorial5_{nq}_{prec}", nq=nq, **kw)
    start, _ = up.initial_coils(prob, init="circle", size="half-R0")
    res, t_first = up.solve_all(prob, [start], precision=prec, certify="none", log=quiet)
    t = res[0]["t"]
    print(f"nq {nq:3d} {prec:5s}: first solve {t_first:5.1f} s (compilation ~{max(t_first - t, 0):4.1f} s), "
          f"solve {t:5.2f} s, mean |B.n|/|B| {res[0]['quick']['mean_bn_pct']:.4f} %")

# compiled programs are shared by structure: another boundary with the same field periods, number of coils, order,
# points per coil, grids and settings reuses them (the boundary and the limit values are program inputs), so its
# first solve compiles nothing new. CFQS has nfp 2, like LP QA.
b2, _ = up.load_boundary("cfqs_2b40")
S = b2.R00
prob2, _ = up.make_problem(b2, "tutorial5_cfqs", ncoils=4, order=4, length=4.0 * S, per_coil_length=True,
                           kappa=3.0 / S, msc=4.0 / S ** 2, dcc=0.1 * S, dcs=0.15 * S, nq=64, grid_dist=(128, 64))
start2, _ = up.initial_coils(prob2, init="circle", size="half-R0")
res2, t_first2 = up.solve_all(prob2, [start2], certify="none", log=quiet)
print(f"CFQS, same structure as the first LP QA problem: first solve {t_first2:.1f} s, solve {res2[0]['t']:.2f} s "
      f"(no new compilation)")
