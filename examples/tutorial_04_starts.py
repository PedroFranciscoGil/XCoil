"""Tutorial 4: starting coils - families and sizes, several starts, screening, warm starts - and picking the best
certified design (docs/tutorials/04_starts.md).

    python examples/tutorial_04_starts.py                # a few minutes on a laptop CPU
"""
from xcoil.fixed import user_problem as up

b, _ = up.load_boundary("landreman_paul_qa")
prob, _ = up.make_problem(b, "tutorial4", ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0,
                          msc=4.0, dcc=0.1, dcs=0.15, nq=64)

# 1. every admissible start: three families (3D offsets, planar offsets, circles) at several sizes; starts that
#    would be linked or closer than the coil-coil limit are dropped or shrunk
starts = up.starts(prob)
print(len(starts), "admissible starts:", ", ".join(s.label for s in starts))

# 2. screen: run the first stage of every start (cheap), finish only the 2 most promising, certify all finished ones
results, t_compile = up.solve_all(prob, starts, screen=2, certify="all", max_outer=40)
for r in results:
    c = r["cert"]
    print(f"  {r['start']:18s} " + ("screened out" if r.get("screened_out") else
          f"{'CERTIFIED' if c['certified'] else 'not certified'} mean {c['mean_bn_pct']:.4f} %"))
best = up.best(results)                     # the certified result with the smallest mean |B.n|/|B|
print("best:", best["start"], round(best["cert"]["mean_bn_pct"], 4), "%")

# 3. warm start: tighten a limit and continue from the best design instead of starting over
prob2, _ = up.make_problem(b, "tutorial4_tight", ncoils=4, order=4, length=3.8, per_coil_length=True, kappa=3.0,
                           msc=4.0, dcc=0.1, dcs=0.15, nq=64)
start = next(s for s in starts if s.label == best["start"])
r2 = up.best(up.solve_all(prob2, [up.warm_start(start, best["x"])], max_outer=30)[0])
print(f"length 3.8 m from the 4 m design: certified {r2['cert']['certified']}, mean {r2['cert']['mean_bn_pct']:.4f} %")
