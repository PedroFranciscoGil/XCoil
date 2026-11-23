"""Tutorial 2: engineering limits - total length, curvature, distances, toroidal ordering, coil force - and reading
the certificate (docs/tutorials/02_limits.md).

    python examples/tutorial_02_limits.py                # about 2 minutes on a laptop CPU
"""
from xcoil.constraints import forces
from xcoil.fixed import user_problem as up

b, _ = up.load_boundary("landreman_paul_qa")

# 1. a TOTAL length budget (18 m for the 4 base coils, shared freely) instead of 4 m per coil, and the toroidal
#    ordering rows (neighbouring coils may not cross; theta about the boundary's fixed geometric axis)
prob, _ = up.make_problem(b, "tutorial2", ncoils=4, order=4, length=18.0, per_coil_length=False,
                          kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
start, _ = up.initial_coils(prob, init="circle", size="half-R0")
r = up.best(up.solve_all(prob, [start], ordering=dict(gap_deg=1.0))[0])
c = r["cert"]
print("\nCertificate (true limits, dense sampling, double precision):")
for k in ("certified", "linked", "v", "mean_bn_pct", "max_bn_pct", "L", "L_max_coil", "max_kappa", "msc_max",
          "d_cc", "d_cs", "I_rel_min", "I_rel_max"):
    print(f"  {k:12s} {c[k]}")
print("  ordering    ", "ok" if c["ordering"]["ok"] else "VIOLATED", "gaps [deg]",
      [round(g, 2) for g in c["ordering"]["gap_window_deg"]])

# 2. add a maximum coil force, 10 % below what this design carries at a total current of 1.25 MA, and continue
#    from the design above (warm start) with the ordering rows kept
I_total = 1.25e6
F_now = max(up.force_to_user(prob, F, I_total) for F in forces.certify_force(r["x"], prob, 1.0, 0.02)["max_force"])
F_max = 0.9 * F_now
print(f"\nmax force of the design above: {F_now:.4g} N/m at {I_total:g} A; limit for the next solve {F_max:.4g} N/m")
mf = dict(F_lim=up.force_limit_in_problem(prob, F_max, I_total), radius=0.02, margin=0.005)
r2 = up.best(up.solve_all(prob, [up.warm_start(start, r["x"])], max_outer=30, max_force=mf,
                          ordering=dict(gap_deg=1.0))[0])
c2 = r2["cert"]
print(f"\nwith the force limit: certified {c2['certified']}, mean |B.n|/|B| {c2['mean_bn_pct']:.4f} %")
print(f"  max force {max(up.force_to_user(prob, F, I_total) for F in c2['force']['max_force']):.4g} N/m "
      f"(limit {F_max:.4g}), ok {c2['force']['ok']}; ordering ok {c2['ordering']['ok']}")
