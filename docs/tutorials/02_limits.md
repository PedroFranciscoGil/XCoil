# Tutorial 2: engineering limits and the certificate

A total length budget, the toroidal ordering constraint and a coil-force limit, and how to read the certificate.
Script: [`examples/tutorial_02_limits.py`](../../examples/tutorial_02_limits.py) (about 2 minutes on a laptop CPU;
the force rows dominate). Every limit is defined in [constraints.md](../constraints.md).

```bash
python examples/tutorial_02_limits.py
```

## 1. Total length and ordering

```python
prob, _ = up.make_problem(b, "tutorial2", ncoils=4, order=4, length=18.0, per_coil_length=False,
                          kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
start, _ = up.initial_coils(prob, init="circle", size="half-R0")
r = up.best(up.solve_all(prob, [start], ordering=dict(gap_deg=1.0))[0])
```

- `per_coil_length=False`: the 4 base coils share 18 m freely, instead of 4 m each.
- `ordering=dict(gap_deg=1.0)`: neighbouring coils may not cross in the (φ, θ) projection. θ is measured about the
  boundary's geometric axis, fixed.

```
  circle_half-R0        8.86 s  outer  7  mean 0.0977 %  L 18.001  kappa 2.910  MSC 3.880  d_cc 0.1010  d_cs 0.2916  v 6.2e-05
  repair circle_half-R0     distance margin 0.0287 (29.3 s)
  certificate circle_half-R0     not certified  mean 0.0976 %  max 0.562 %  (13.0 s)

Certificate (true limits, dense sampling, double precision):
  certified    False
  linked       False
  v            0.007471472484257835
  mean_bn_pct  0.09755590152268187
  max_bn_pct   0.5617047546050797
  L            18.000029056518333
  L_max_coil   4.587016121556397
  max_kappa    2.9420416558487905
  msc_max      3.8799999176578224
  d_cc         0.09925285275157422
  d_cs         0.2906447225633826
  I_rel_min    0.9760963774563515
  I_rel_max    1.0385750188062415
  ordering     ok gaps [deg] [6.27, 7.53, 12.05, 6.36, 21.12]
```

**Reading the certificate.** Two more metres of coil cut the field error from 0.50 % to 0.098 %. But the design is
*not certified*: `v` = 0.0075 is the largest violation, and the value that fails is `d_cc` = 0.0993 m against the
0.1 m limit (0.7 % short). Everything else holds: the length (18.000 m), the curvature (2.94 ≤ 3), the MSC
(3.880 ≤ 4), the coil–surface distance and the ordering. The currents stay within 4 % of each other.

Why: the solver measured the coil–coil distance between 64 points per coil, and the longest coil is 4.6 m long.
The exact distance between the polylines is a little smaller. `solve_all` already tried one automatic repair (a
larger distance margin) and got closer, but not enough. The fixes, in order: more points per coil (the default
`nq`, which follows the coil length and the distance limits, would have caught it), or a warm start, as below.

## 2. A coil-force limit

```python
I_total = 1.25e6                                        # the coils' real total current [A]
F_now = max(up.force_to_user(prob, F, I_total) for F in forces.certify_force(r["x"], prob, 1.0, 0.02)["max_force"])
F_max = 0.9 * F_now
mf = dict(F_lim=up.force_limit_in_problem(prob, F_max, I_total), radius=0.02, margin=0.005)
r2 = up.best(up.solve_all(prob, [up.warm_start(start, r["x"])], max_outer=30, max_force=mf,
                          ordering=dict(gap_deg=1.0))[0])
```

The force per unit length scales with the current squared, so a limit in N/m only means something together with the
current: `force_limit_in_problem` converts it to the solver's normalization, and `force_to_user` converts back.
`radius` is the conductor radius of the self-force. `margin` keeps the solve 0.5 % inside the limit, because the
certificate checks 768 points per coil and the solve checks 64.

```
max force of the design above: 3.233e+05 N/m at 1.25e+06 A; limit for the next solve 2.91e+05 N/m
  warm                 36.49 s  outer  6  mean 0.0981 %  L 18.000  kappa 2.910  MSC 3.878  d_cc 0.1165  d_cs 0.2784  v 1.9e-05
  certificate warm               CERTIFIED  mean 0.0981 %  max 0.545 %  (15.1 s)

with the force limit: certified True, mean |B.n|/|B| 0.0981 %
  max force 2.902e+05 N/m (limit 2.91e+05), ok True; ordering ok True
```

A 10 % lower force costs almost nothing in field error (0.0976 % → 0.0981 %). The coils moved apart (d_cc 0.117 m),
and the design is now certified: force, ordering and every other limit hold.

## The other opt-in limits

- **Lp curvature**: `solve_all(..., lp_curvature=dict(J_max=0.5, p=2.0, threshold=2.0))` limits how far each coil's
  curvature exceeds 2 m⁻¹, in an integral sense.

On the command line: `xcoil-coils --total-length --length 18 --ordering --max-force 2.9e5 --total-current 1.25e6`
([cli.md](../cli.md)).
