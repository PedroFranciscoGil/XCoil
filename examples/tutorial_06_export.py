"""Tutorial 6: exporting coils - JSON (CurveXYZFourier coefficients), MAKEGRID, ParaView VTK, simsopt - and reading
them back (docs/tutorials/06_export.md).

    python examples/tutorial_06_export.py                # a minute on a laptop CPU
"""
import json

import numpy as np

from xcoil.fixed import user_problem as up

b, _ = up.load_boundary("landreman_paul_qa")
prob, _ = up.make_problem(b, "tutorial6", ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0,
                          msc=4.0, dcc=0.1, dcs=0.15, nq=64)
start, _ = up.initial_coils(prob, init="circle", size="half-R0")
r = up.best(up.solve_all(prob, [start])[0])

I_total = 3e5                                       # the base coils carry 300 kA together in every export
up.write_coils_json(prob, r["x"], "tutorial6_coils.json", total_current=I_total, extra=dict(note="tutorial 6"))
up.write_makegrid(prob, r["x"], "coils.tutorial6", total_current=I_total)
up.write_vtk(prob, r["x"], "tutorial6")              # tutorial6_surface.vts (|B.n|/|B| on the boundary), _coils.vtu
written = up.write_simsopt(prob, r["x"], "tutorial6_biot_savart.json", total_current=I_total)
print("simsopt BiotSavart file:", "written" if written else "skipped (simsopt not installed)")

# all physical coils as points and currents, e.g. for your own field-line tracer
P, I_phys, I_base = up.physical_coils(prob, r["x"], total_current=I_total, n=128)
print("physical coils:", P.shape, "currents [kA]:", np.round(I_base / 1e3, 2), "sum", round(I_base.sum() / 1e3, 1))

# read the JSON back: per base coil the CurveXYZFourier coefficients (simsopt order) and the current
J = json.load(open("tutorial6_coils.json"))
print("JSON keys:", sorted(J)[:8], "...; base coils:", len(J["base_coils"]), "order", J["order"], "nfp", J["nfp"])
