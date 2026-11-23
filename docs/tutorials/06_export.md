# Tutorial 6: exporting coils

Every format XCoil writes, and how to read the coils back. Script:
[`examples/tutorial_06_export.py`](../../examples/tutorial_06_export.py).

```bash
python examples/tutorial_06_export.py
```

```python
I_total = 3e5                                      # the base coils carry 300 kA together in every export
up.write_coils_json(prob, r["x"], "tutorial6_coils.json", total_current=I_total, extra=dict(note="tutorial 6"))
up.write_makegrid(prob, r["x"], "coils.tutorial6", total_current=I_total)
up.write_vtk(prob, r["x"], "tutorial6")
up.write_simsopt(prob, r["x"], "tutorial6_biot_savart.json", total_current=I_total)   # needs simsopt
P, I_phys, I_base = up.physical_coils(prob, r["x"], total_current=I_total, n=128)
```

| file / call | content | use it in |
|---|---|---|
| `write_coils_json` | per base coil: `CurveXYZFourier` coefficients in simsopt's order (for x, y, z: c₀, s₁, c₁, …, s_M, c_M) and the current [A]; `nfp`, `order`, the symmetry; `extra` keys | simsopt (below), your code |
| `write_makegrid` | every physical coil as points [m] with the current per point, MAKEGRID coils-file format | VMEC/VMEX free boundary (MAKEGRID → mgrid), field-line tracers |
| `write_vtk` | `tutorial6_surface.vts`: the boundary (full torus) with B·n/\|B\|, \|B\|, B·n; `tutorial6_coils.vtu`: all coils as polylines with base-coil index and current | ParaView |
| `write_simsopt` | a simsopt `BiotSavart` JSON (returns False if simsopt is not installed) | simsopt `load` |
| `physical_coils` | in memory: points (Nphys, n, 3), signed currents (Nphys,), base currents (C,) | your own code |

The field error does not depend on `total_current`; forces do (∝ I²).

## Output

```
  circle_half-R0       17.06 s  outer 10  mean 0.5028 %  L 16.001  kappa 2.910  MSC 3.880  d_cc 0.1557  d_cs 0.2265  v 8.4e-05
  certificate circle_half-R0     CERTIFIED  mean 0.5028 %  max 4.996 %  (9.8 s)
simsopt BiotSavart file: skipped (simsopt not installed)
physical coils: (16, 128, 3) currents [kA]: [69.48 71.66 76.7  82.15] sum 300.0
JSON keys: ['base_coils', 'dof_order', 'format', 'nfp', 'note', 'order', 'stellsym', 'symmetry'] ...; base coils: 4 order 4 nfp 2
```

16 physical coils (4 base coils × 2 field periods × 2 for stellarator symmetry). The 4 base currents differ (69–82 kA)
because the currents are free, and they add up to the 300 kA given.

## In simsopt

```python
import json, numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.field import Current, coils_via_symmetries, BiotSavart

J = json.load(open("tutorial6_coils.json"))
curves = []
for c in J["base_coils"]:
    cv = CurveXYZFourier(256, J["order"]); cv.x = np.array(c["dofs"]); curves.append(cv)
coils = coils_via_symmetries(curves, [Current(c["current"]) for c in J["base_coils"]], J["nfp"], True)
bs = BiotSavart(coils)
```
