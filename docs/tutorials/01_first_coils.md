# Tutorial 1: first coils

Coils for the Landreman–Paul precise QA boundary (nfp 2), the problem of simsopt's `auglag_qa.py` example: solve,
certify, export. Script: [`examples/tutorial_01_first_coils.py`](../../examples/tutorial_01_first_coils.py).

```bash
python examples/tutorial_01_first_coils.py
```

## The steps

```python
from xcoil.fixed import user_problem as up

b, info = up.load_boundary("landreman_paul_qa")
prob, settings = up.make_problem(b, "tutorial1", ncoils=4, order=4, length=4.0, per_coil_length=True,
                                 kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
```

1. **The boundary**: a bundled one by name. [Tutorial 3](03_own_boundary.md) loads your own.
2. **The problem**: 4 coils per half field period (16 coils in total, by symmetry), each a Fourier series of order 4.
   The limits: at most 4 m per coil, curvature at most 3 m⁻¹, mean squared curvature at most 4 m⁻², coils at least
   0.1 m apart and at least 0.15 m from the plasma. `nq=64` points per coil is enough here and much faster than the
   conservative default. The key `"tutorial1"` names the problem.
3. **Starting coils**: circles of radius R0/2 around the magnetic axis, as in `auglag_qa.py`:
   `start, check = up.initial_coils(prob, init="circle", size="half-R0")`. `check["linked"]` warns if the starting
   coils are linked; a linked start cannot give a certified design.
4. **Solve and certify**: `results, t_compile = up.solve_all(prob, [start], certify="best")`, then
   `r = up.best(results)`. `r["x"]` is the design, `r["cert"]` its certificate.
5. **Export**: `write_coils_json`, `write_makegrid` and `write_vtk` (see [tutorial 6](06_export.md)).

## Output

```
boundary: bundled boundary nfp 2
problem: 111 unknowns, 776 constraint rows, 64 points per coil
start: circle_half-R0 unlinked
  circle_half-R0        9.42 s  outer 10  mean 0.5028 %  L 16.001  kappa 2.910  MSC 3.880  d_cc 0.1557  d_cs 0.2265  v 8.4e-05
  certificate circle_half-R0     CERTIFIED  mean 0.5028 %  max 4.996 %  (8.3 s)
certified: True  mean |B.n|/|B| = 0.5028 %  max = 4.996 %
limits: longest coil 4.000 m (4), max curvature 2.914 (3), max MSC 3.880 (4), coil-coil 0.1548 m (0.1), coil-surface 0.2251 m (0.15)
wrote tutorial1_coils.json, coils.tutorial1, tutorial1_surface.vts, tutorial1_coils.vtu
```

- **111 unknowns**: 4 coils × 3 coordinates × 9 Fourier coefficients (order 4), plus 3 current variables (the 4
  currents are free with a fixed sum). **776 constraint rows**: the limits at every point of every coil.
- **The solve line**: 10 outer (augmented-Lagrangian) iterations. `mean` is the field error. `L` is the total length of
  the 4 base coils. `v` is the largest limit violation, which must be at most 10⁻³.
- **The certificate** re-measures everything in double precision on dense samples: CERTIFIED. The length limit is
  binding (4.000 m), and so is the MSC almost (3.88 with the solver's 3 % margin). The distances have room.
- `mean |B.n|/|B| = 0.5028 %` is the field error of this design. The maximum, 5 %, sits where the coils pass closest
  to the plasma.

Open `tutorial1_surface.vts` and `tutorial1_coils.vtu` in ParaView to see the coils and the field error on the
boundary.

The same problem from the command line: `xcoil-coils --init circle --size half-R0 --nq 64` ([cli.md](../cli.md)).
Next: [tutorial 2](02_limits.md), more engineering limits.
