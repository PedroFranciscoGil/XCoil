"""XCoil from Python: the steps of xcoil_coils.py, for your own scripts.

    python example_api.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent

import jax                                                 # noqa: E402

jax.config.update("jax_enable_x64", True)
from xcoil.fixed import user_problem as up              # noqa: E402

# 1. target surface: a VMEC input file, a wout file, a bundled name (e.g. "w7x"), ...
b, info = up.load_boundary(str(HERE / "input.LandremanPaul2021_QA"))

# 2. coils and limits (the problem of simsopt's auglag_qa.py)
prob, settings = up.make_problem(b, "lpqa_api", ncoils=4, order=4, length=4.0, per_coil_length=True,
                                 kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=128)

# 3. starting coils: one family and size (or up.starts(prob) for all admissible ones)
start, check = up.initial_coils(prob, init="circle", size="half-R0")
print("starting coils:", start.label, start.params, check)

# 4. solve and certify (certify: "best", "all" or "none")
results, t_compile = up.solve_all(prob, [start], precision="mixed", certify="best")
r = up.best(results)
print("quick check:", {k: r["quick"][k] for k in ("mean_bn_pct", "max_bn_pct", "v", "linked")})
print("certified:", r["cert"]["certified"], "mean |B.n|/|B| [%]:", r["cert"]["mean_bn_pct"],
      "max [%]:", r["cert"]["max_bn_pct"])

# 5. outputs
up.write_coils_json(prob, r["x"], "lpqa_coils.json", total_current=3e5)   # CurveXYZFourier coefficients + currents
up.write_makegrid(prob, r["x"], "coils.lpqa", total_current=3e5)          # MAKEGRID coils file
up.write_vtk(prob, r["x"], "lpqa")                                        # lpqa_surface.vts, lpqa_coils.vtu
