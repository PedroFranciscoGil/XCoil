# Python API

The functions most users need, by task. Every signature and default is in [reference.md](reference.md#python-api).

```python
from xcoil.fixed import user_problem as up
```

## Boundary → problem

| call | returns |
|---|---|
| `up.load_boundary(spec)` | `(b, info)`: a `FourierBoundary` from a bundled name, a VMEC input, a wout, an `.npz` or a simsopt JSON ([boundaries.md](boundaries.md)) |
| `up.make_problem(b, key, ncoils, order, length, per_coil_length, kappa, msc, dcc, dcs, nq=None, ...)` | `(prob, settings)`: the coil problem (`prob.n` unknowns, `prob.m` constraint rows) |

`key` names the problem (and its stored boundary): use a new key for every boundary.

## Starting coils

| call | returns |
|---|---|
| `up.initial_coils(prob, init="3d", size="clear", value=None, tilt=0.0)` | `(start, check)`: one start. `init`: `"circle"`, `"planar"`, `"3d"`; `size`: `"clear"`, `"L70"`, `"L85"`, `"L100"`, `"half-R0"`; or `value` (offset or radius [m]). `check["linked"]` warns about linked coils |
| `up.starts(prob)` | every admissible start of the three families at several sizes (linked or overlapping ones dropped or shrunk) |
| `up.pair(prob)` | one planar and one 3D start |
| `up.warm_start(start, x)` | a start at the design `x` (continue from a previous result) |

A start has `.label`, `.params`, `.x` (the design vector).

## Solve and certify

```python
results, t_compile = up.solve_all(prob, starts, precision="mixed", certify="best", max_outer=60,
                                  target_mean=None, screen=None, link_test=True,
                                  max_force=None, ordering=None, lp_curvature=None)
r = up.best(results)
```

`results` has one dict per start:

| key | content |
|---|---|
| `start`, `params` | the start's label and size |
| `x` | the design vector: coil coefficients (simsopt `CurveXYZFourier` order per coil), then C − 1 current variables |
| `t` | solve time [s] |
| `quick` | cheap metrics: `mean_bn_pct`, `max_bn_pct`, `F`, `v` (largest limit violation), `L` (per coil), `kappa`, `msc`, `d_cc`, `d_cs`, `linked` |
| `cert` | the certificate (or `None` if not certified): `certified`, `linked`, `v`, `mean_bn_pct`, `max_bn_pct`, `L` (total), `L_max_coil`, `max_kappa`, `msc_max`, `d_cc`, `d_cs`, `I_rel_min`, `I_rel_max`, `f_B_hi`, and `force` / `ordering` / `lp_curvature` when those are on |
| `info` | solver details: `outer` iterations, stop reason |
| `screened_out` | with `screen=k`: True for starts not finished |

`certify`: `"best"` certifies in the order of the quick check until one passes, `"all"` every result, `"none"`
nothing. `prob.certify(x)` certifies any design; `up.quick_check(prob, x)` gives the quick metrics.

## Opt-in constraints

```python
from xcoil.constraints import forces, ordering, lp_curvature

mf = dict(F_lim=up.force_limit_in_problem(prob, 2e5, total_current=1.25e6), radius=0.02, margin=0.005)
od = dict(gap_deg=1.0)                       # axis=None: the boundary's geometric axis, fixed
lp = dict(J_max=0.5, p=2.0, threshold=2.0)
results, _ = up.solve_all(prob, starts, max_force=mf, ordering=od, lp_curvature=lp)
```

The pieces are public too: `forces.coil_forces`, `forces.certify_force`, `ordering.rows`,
`ordering.axis_from_boundary`, `ordering.certify_ordering`, `lp_curvature.lp_curvature`, `lp_curvature.certify_lp`.

## Export

| call | writes |
|---|---|
| `up.write_coils_json(prob, x, path, total_current=None, extra=None)` | base coils as `CurveXYZFourier` coefficients, currents [A], symmetry |
| `up.write_makegrid(prob, x, path, total_current=None, n=256)` | all coils, MAKEGRID coils-file format |
| `up.write_vtk(prob, x, prefix)` | `prefix_surface.vts` (B·n/\|B\| on the boundary) and `prefix_coils.vtu` for ParaView |
| `up.write_simsopt(prob, x, path, total_current=None)` | a simsopt `BiotSavart` JSON (if simsopt is installed; returns False otherwise) |
| `up.physical_coils(prob, x, total_current=None, n=256)` | (points (Nphys, n, 3), signed currents (Nphys,), base currents (C,)) in memory |

`total_current` is the sum of the base-coil currents in the export; the field error does not depend on it.

## Lower level

`xcoil.fixed.device_lm.build_solver(prob, budget, extra=...)` builds the compiled solver `solve(x0) -> (x, info)`
that `solve_all` uses (`budget`: [reference.md](reference.md#solver-budget)); `build_batched_solver` solves many starts
of one problem in one program; `Extra` adds your own constraint rows g(x) ≤ 0 (JAX functions of x). The problem
object exposes `prob.F(x, 1.0)`, `prob.F_grad(x, 1.0)`, `prob.g(x, 1.0)`, `prob.g_jac(x, 1.0)` (objective and
constraint rows at t = 1, the target boundary) and `prob.T` (its tables: `T.sym`, `T.cb`, `T.I0`).
