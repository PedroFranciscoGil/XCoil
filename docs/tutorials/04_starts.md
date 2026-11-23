# Tutorial 4: starting coils, many starts, warm starts

The coil problem is not convex: the design you get depends on the starting coils. This tutorial builds every
admissible start, screens them, keeps the best certified design, then continues from it after tightening a limit.
Script: [`examples/tutorial_04_starts.py`](../../examples/tutorial_04_starts.py).

```bash
python examples/tutorial_04_starts.py
```

## Start families

All starts are built from the boundary alone:

| family (`init`) | coils |
|---|---|
| `"3d"` (`offset_3d`) | the boundary offset outward by a distance d, cut at the coils' toroidal angles: surface-following coils |
| `"planar"` (`offset_planar`) | planar offsets of the toroidally averaged cross-section |
| `"circle"` | circles around the magnetic-axis estimate (simsopt's `create_equally_spaced_curves`) |

| size | meaning |
|---|---|
| `"clear"` | the smallest size with every coil 1.1 d_cs from the plasma |
| `"L70"`, `"L85"`, `"L100"` | 70 / 85 / 100 % of the length budget |
| `"half-R0"` | circles of radius R0/2 |
| `value=` | an explicit offset d or radius [m] |

`up.starts(prob)` builds all of them, plus tilted variants (±8°), and drops or shrinks the ones whose coils would be
linked or closer than the coil–coil limit.

## Screen, finish, pick

```python
starts = up.starts(prob)
results, t_compile = up.solve_all(prob, starts, screen=2, certify="all", max_outer=40)
best = up.best(results)                      # the certified result with the smallest field error
```

`screen=2` runs only the first stage of every start (the first subproblem on coarse grids, about a tenth of a solve).
It ranks the starts by field error after that stage and finishes the best 2; the rest are reported as
`screened_out`. Without `screen`, every start is solved fully. `certify="all"` certifies every finished result.

## Warm start

After changing a limit, continue from the previous design instead of starting over:

```python
prob2, _ = up.make_problem(b, "tutorial4_tight", ..., length=3.8, ...)   # 3.8 m per coil instead of 4
r2 = up.best(up.solve_all(prob2, [up.warm_start(start, best["x"])], max_outer=30)[0])
```

## Output

```
19 admissible starts: 3d_clear, 3d_L70, 3d_L85, 3d_L100, 3d_L85_tilt+8, 3d_L85_tilt-8, planar_clear, ...
  screening: first stage of 19 starts in 25.77 s; finishing circle_L70, planar_L85_tilt-8
  circle_L70           25.02 s  outer 10  mean 0.5028 %  L 16.001  kappa 2.910  MSC 3.880  d_cc 0.1548  d_cs 0.2263  v 8.0e-05
  planar_L85_tilt-8    26.60 s  outer 10  mean 0.5028 %  L 16.001  kappa 2.910  MSC 3.880  d_cc 0.1585  d_cs 0.2260  v 8.4e-05
  certificate circle_L70         CERTIFIED  mean 0.5028 %  max 5.009 %  (21.2 s)
  certificate planar_L85_tilt-8  CERTIFIED  mean 0.5028 %  max 4.996 %  (20.7 s)
  circle_L70         CERTIFIED mean 0.5028 %
  planar_L85_tilt-8  CERTIFIED mean 0.5028 %
  circle_L100        screened out
  ...
best: circle_L70 0.5028 %
  warm                 25.00 s  outer  9  mean 1.2744 %  L 15.203  kappa 2.858  MSC 3.881  d_cc 0.1625  d_cs 0.1861  v 5.0e-04
  certificate warm               CERTIFIED  mean 1.2744 %  max 10.680 %  (20.4 s)
length 3.8 m from the 4 m design: certified True, mean 1.2744 %
```

Two very different starts (circles, tilted planar coils) reach the same 0.5028 % design on this boundary. On harder
boundaries they do not, which is why several starts are worth their cost. The warm start reaches the new optimum in
9 outer iterations. Taking 5 % off the length budget raises the field error from 0.50 % to 1.27 %: the length limit is
the binding one here.

From the command line: `xcoil-coils --init all --screen 2` ([cli.md](../cli.md#4-choose-the-initial-coils----init---size)).
