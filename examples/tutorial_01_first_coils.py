"""Tutorial 1: first coils for the Landreman-Paul QA boundary, certified and exported (docs/tutorials/01_first_coils.md).

    python examples/tutorial_01_first_coils.py           # about a minute on a laptop CPU
"""
from xcoil.fixed import user_problem as up

# 1. the target boundary: a bundled one by name (or a VMEC input / wout file, see tutorial 3)
b, info = up.load_boundary("landreman_paul_qa")
print("boundary:", info["source"], "nfp", b.nfp)

# 2. the coil problem: 4 coils per half period, Fourier order 4, and the limits of simsopt's auglag_qa.py example
prob, settings = up.make_problem(b, "tutorial1", ncoils=4, order=4, length=4.0, per_coil_length=True,
                                 kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
print(f"problem: {prob.n} unknowns, {prob.m} constraint rows, {settings['nq_opt']} points per coil")

# 3. starting coils: circles of radius R0/2 around the magnetic axis estimate
start, check = up.initial_coils(prob, init="circle", size="half-R0")
print("start:", start.label, "linked" if check["linked"] else "unlinked")

# 4. solve, then certify the best result independently in double precision
results, t_compile = up.solve_all(prob, [start], certify="best")
r = up.best(results)
c = r["cert"]
print(f"certified: {c['certified']}  mean |B.n|/|B| = {c['mean_bn_pct']:.4f} %  max = {c['max_bn_pct']:.3f} %")
print(f"limits: longest coil {c['L_max_coil']:.3f} m (4), max curvature {c['max_kappa']:.3f} (3), "
      f"max MSC {c['msc_max']:.3f} (4), coil-coil {c['d_cc']:.4f} m (0.1), coil-surface {c['d_cs']:.4f} m (0.15)")

# 5. export: coefficients + currents (JSON), MAKEGRID coils file, ParaView files
up.write_coils_json(prob, r["x"], "tutorial1_coils.json", total_current=3e5)
up.write_makegrid(prob, r["x"], "coils.tutorial1", total_current=3e5)
up.write_vtk(prob, r["x"], "tutorial1")
print("wrote tutorial1_coils.json, coils.tutorial1, tutorial1_surface.vts, tutorial1_coils.vtu")
