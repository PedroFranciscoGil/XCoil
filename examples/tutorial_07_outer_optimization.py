"""Tutorial 7: XCoil inside an outer optimization - the gradient of the optimal coil error with respect to the
boundary (envelope theorem), checked against re-solves (docs/tutorials/07_outer_optimization.md). The full
single-stage optimizer built on this is QSS (github.com/PedroFranciscoGil/qss-filaments).

    python examples/tutorial_07_outer_optimization.py    # a few minutes on a laptop CPU
"""
import numpy as np
from scipy.optimize import nnls

from xcoil.fixed import user_problem as up
from xcoil.fixed.device_lm import build_solver
from xcoil.fixed.device_lm import MAIN
from xcoil.homotopy import boundary as bd

kw = dict(ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
quiet = lambda *a, **k: None
b0, _ = up.load_boundary("landreman_paul_qa")
sel = int(np.where((b0.xm == 1) & (b0.xn == 1))[0][0])          # the boundary coefficient we vary: rc(1, 1)


def boundary(h):
    """The LP QA boundary with rc(1, 1) changed by h."""
    rc = np.array(b0.rc, float); rc[sel] += h
    return bd.from_modes(b0.xm, b0.xn, rc, b0.zs, b0.nfp, True, name="lpqa_h")


def problem(h):
    return up.make_problem(boundary(h), f"tutorial7_{h:+.0e}", **kw)[0]


def tight_solve(p, x0):
    """A converged solve of F itself: no IRLS reweighting (solve_all's default steers towards the L1 field ratio,
    so its optimum is not a KKT point of F), stall tolerance 1e-9, limits to 1e-7 (finite differences of optimal
    values need the optimum much more accurately than solve_all's 1 % objective stall)."""
    solve = build_solver(p, dict(MAIN, irls=0.0), max_outer=200, viol_tol=1e-7, ftol=1e-9)
    x, info = solve(np.asarray(x0))
    return np.asarray(x)


# 1. the coil optimum at the base boundary: x*, F* = F(x*), and the multipliers z* of the active limits
prob = problem(0.0)
start, _ = up.initial_coils(prob, init="circle", size="half-R0")
r = up.best(up.solve_all(prob, [start], log=quiet)[0])
x = tight_solve(prob, r["x"])
F0, gF = prob.F_grad(x, 1.0)
g, A = prob.g_jac(x, 1.0)
act = np.where(g > -1e-3)[0]
z, _ = nnls(A[act].T, -gF)                      # stationarity: grad F + A_act^T z = 0, z >= 0
print(f"F* = {F0:.6e}, {act.size} near-active limits, stationarity |grad F + A^T z| / |grad F| = "
      f"{np.linalg.norm(gF + A[act].T @ z) / np.linalg.norm(gF):.1e}")


# 2. envelope theorem: dF*/ds = d/ds [F(x*; s) + z*^T g(x*; s)] at FIXED coils and multipliers - no re-solve
def lagrangian(h):
    p = problem(h)
    return p.F(x, 1.0) + z @ p.g(x, 1.0)[act]


h = 1e-4
G_env = (lagrangian(h) - lagrangian(-h)) / (2 * h)
print(f"dF*/d rc(1,1) by the envelope theorem: {G_env:+.5e}")

# 3. check: re-solve the coils at s +- h (warm-started from x*) and difference the optimal values
for h in (1e-3, 3e-4):
    F = [problem(s).F(tight_solve(problem(s), x), 1.0) for s in (h, -h)]
    G_fd = (F[0] - F[1]) / (2 * h)
    print(f"  re-solve finite difference, h = {h:g}: {G_fd:+.5e} (relative difference {abs(G_env - G_fd) / abs(G_fd):.1e})")
