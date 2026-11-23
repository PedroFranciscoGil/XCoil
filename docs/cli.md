# The `xcoil-coils` command

`xcoil-coils` designs coils for one or several target boundaries from the command line: starting coils, solve,
certificate, and every output file. Every option is in [reference.md](reference.md#xcoil-coils); this page explains
them by task. `python examples/xcoil_coils.py` is the same command.

Its defaults are the problem of SIMSOPT's `auglag_qa.py` example: the Landreman–Paul precise QA boundary, 4 coils per
half period of Fourier order 4, 4 m per coil, coil–coil ≥ 0.1 m, coil–surface ≥ 0.15 m, curvature ≤ 3 m⁻¹,
MSC ≤ 4 m⁻².

## 1. First run

```bash
xcoil-coils --nq 128
```

About a minute on a laptop CPU (the first solve also compiles). It prints the starting coils, the solve, and the result:

```
  initial coils 3d_clear: d = 0.1651 m, total base-coil length 9.423 m, min coil-coil 0.2545 m (limit 0.1), min coil-surface 0.1650 m (limit 0.15), unlinked
  3d_clear             22.95 s  outer 10  mean 0.5028 %  L 16.001  kappa 2.910  MSC 3.880  d_cc 0.1552  d_cs 0.2263  v 7.9e-05
  certificate 3d_clear           CERTIFIED  mean 0.5028 %  max 5.011 %  (17.5 s)
Best start 3d_clear: CERTIFIED  mean |B.n|/|B| = 0.5028 %, max = 5.0107 %; length 16.001 m (longest coil 4.000 m), max curvature 2.911, max MSC 3.880, min coil-coil 0.1547 m, min coil-surface 0.2252 m, currents 0.93-1.10 x mean
```

`mean |B.n|/|B|` is the area-weighted mean of the normal field over the field strength on the target surface (the
field error); `max` its maximum. SIMSOPT is **not** needed; with SIMSOPT installed XCoil also reads SIMSOPT JSON
surfaces (e.g. QUASR devices) and writes a SIMSOPT `BiotSavart` file.

## 2. Choose the target surface — `--surface`

| input | example |
|---|---|
| VMEC input file (`RBC(n,m)`, `ZBS(n,m)`, `NFP`) | `--surface input.my_device` |
| VMEC `wout` file (last flux surface) | `--surface wout_my_device.nc` |
| bundled boundaries | `--surface w7x`, `landreman_paul_qa`, `qh_reactor`, `cfqs_2b40`, `rotating_ellipse` |
| XCoil `.npz` boundary | `--surface my_boundary.npz` |
| SIMSOPT JSON (needs SIMSOPT) | `--surface serial0755658.json` (a QUASR device; its outermost surface) |

Only stellarator-symmetric surfaces are supported. Surfaces that are not given as R, Z Fourier series
in the cylindrical angle (SIMSOPT's `SurfaceXYZTensorFourier`, as in QUASR) are refitted to one; the
fit error is printed, with a warning if it exceeds 1 mm.

Several surfaces in one call are solved one after another, each into its own output folder, with the
same settings: `--surface a.json b.json input.c`. For surfaces that need different limits, run the
script once per surface.

## 3. Set the coils and the limits

| option | meaning | default |
|---|---|---|
| `--ncoils` | base coils per half field period | 4 |
| `--order` | Fourier order of each coil | 4 |
| `--length` | length limit [m] | 4 |
| `--per-coil` / `--total-length` | the limit applies to each coil / to all base coils together | per coil |
| `--cc` | minimum coil–coil distance [m] | 0.1 |
| `--cs` | minimum coil–surface distance [m] | 0.15 |
| `--curvature` | maximum curvature, pointwise [1/m] | 3 |
| `--msc` | maximum mean squared curvature per coil [1/m²] | 4 |
| `--target-mean` | stop as soon as the mean \|B·n\|/\|B\| [%] reaches this (else: converge) | off |
| `--max-force`, `--force-radius`, `--force-margin` | opt-in maximum coil force per unit length [N/m] at `--total-current`, conductor radius [m], solver margin | off, 0.02, 0.005 |
| `--lp-curvature-max`, `--lp-p`, `--lp-threshold` | opt-in Lp-curvature limit per coil | off, 2, 0 |
| `--ordering`, `--ordering-gap` | opt-in toroidal ordering (no crossing of neighbouring coils) and its minimum gap [deg] | off, 1 |

All coils of one field period are copies of the base coils by stellarator symmetry; the currents are
free, with their sum fixed. The curvature and MSC limits are enforced 3 % inside the given values, so
that the certificate (which checks the given values on dense samples) passes.

## 4. Choose the initial coils — `--init`, `--size`

The starting coils are built from the boundary alone:

| `--init` | starting coils |
|---|---|
| `circle` | planar circles of radius R1 about the magnetic-axis estimate, in each coil plane (as SIMSOPT's `create_equally_spaced_curves`) |
| `planar` | planar offsets, at a distance d, of the toroidally averaged cross-section |
| `3d` (default) | surface-following offsets, at a distance d, of the full 3D boundary |
| `pair` | one planar and one 3D start (the two families fail on different devices, e.g. planar on the QH reactor); the better certified design is kept |
| `all` | every family and size below (up to 19 starts); the best certified design is kept |

| `--size` | size of the starting coils |
|---|---|
| `clear` (default) | the smallest that clears the coil–surface limit |
| `L70`, `L85`, `L100` | coils using 70 / 85 / 100 % of the length budget |
| `half-R0` | circles with R1 = R0/2 (as `auglag_qa.py`) |

or give the size directly: `--offset 0.25` (d in m, `planar` / `3d`) or `--radius 0.4` (R1 in m,
`circle`). `--tilt 5` rotates every coil plane by 5° about its radial axis.

Before solving, the script reports the starting coils' size, length and distances, and whether any
two coils are **linked**. Linked starting coils cannot give a certified design (the solver keeps the
coils' topology): choose a smaller size, another family, or `--init all`, which keeps only admissible
starts (unlinked, coils at least `--cc` apart; larger ones are shrunk, impossible ones dropped).

```bash
xcoil-coils --init circle --size half-R0        # auglag_qa.py's starting coils
xcoil-coils --init planar --size L85
xcoil-coils --init 3d --offset 0.25 --tilt 5
xcoil-coils --init pair                         # one planar + one 3D start
xcoil-coils --init all                          # all starts, keep the best
xcoil-coils --init all --screen 1               # all starts screened, only the best one finished
```

With several starts (`pair`, `all`), `--screen N` runs only the first stage of every start's solve
(the first subproblem, on coarse grids - roughly a tenth of a solve) and finishes only the N most
promising ones, ranked by their |B·n|/|B| after that stage. This is experimental: it usually picks the best start or one within a few
percent, at a fraction of the cost of solving every start. `--max-outer` caps the outer iterations of every start (default 60).

## 5. Certification — `--certify`

The solver works with sampled coils and smooth (soft-minimum) distances. The **certificate** re-measures
a design independently in double precision: \|B·n\|/\|B\| on a finer surface grid (the maximum also on a
3× finer grid), exact minimum distances on dense samples, curvature from 4096 samples per coil
refined around its peaks, MSC, length, and the Gauss linking number of every coil pair. A design is
*certified* if all limits hold to 10⁻³ (relative) and no coils are linked. Certify every design you
report or use.

| `--certify` | what is certified |
|---|---|
| `best` (default) | the results in order of the quick check (unlinked first, then smallest limit violation, then objective) until one passes |
| `all` | every result |
| `none` | nothing: only the quick check (\|B·n\|/\|B\| on the certification grid, sampled lengths, curvature, MSC, distances, linking); the result is reported as *not certified* |

A certificate takes a few seconds on a GPU and about 20–30 s on a laptop CPU.

## 6. Outputs

In `output_xcoil/<surface>_<settings>/`:

| file | contents |
|---|---|
| `optimized_surface.vts` | target surface (full torus, VTK StructuredGrid) coloured by B·n/\|B\| (`Bn_over_B`), \|B\| (`modB`) and B·n (`Bn`) |
| `optimized_coils.vtu` | all coils (VTK UnstructuredGrid of closed polylines) with their base-coil index (`idx`, `base_coil`) and current (`current`, `current_rel`) |
| `init_surface.vts`, `init_coils.vtu` | the same for the starting coils |
| `coils_optimized.json` | base coils as `CurveXYZFourier` coefficients (SIMSOPT's order) and currents [A], with the symmetry |
| `coils.<surface>` | all coils in the MAKEGRID coils-file format |
| `summary.json` | settings, every start's quick check and certificate, and its coefficients |
| `biot_savart_optimized.json` | SIMSOPT `BiotSavart` file (only if SIMSOPT is installed) |

The currents are scaled so that the base coils carry `--total-current` (default 3·10⁵ A) together;
\|B·n\|/\|B\| does not depend on it. To use the coils in SIMSOPT:

```python
import json, numpy as np
from simsopt.geo import CurveXYZFourier
from simsopt.field import Current, coils_via_symmetries, BiotSavart
J = json.load(open("output_xcoil/<run>/coils_optimized.json"))
curves = []
for c in J["base_coils"]:
    cv = CurveXYZFourier(256, J["order"]); cv.x = np.array(c["dofs"]); curves.append(cv)
coils = coils_via_symmetries(curves, [Current(c["current"]) for c in J["base_coils"]], J["nfp"], True)
bs = BiotSavart(coils)
```

## 7. More examples

```bash
# W7-X: 4 coils of order 12, 38 m in total, total length 38 m
xcoil-coils --surface w7x --order 12 --length 38 --total-length --cc 0.3 --cs 0.35 \
       --curvature 2.5 --msc 1.5 --init all

# a QUASR device (download serial0755658.json from the QUASR database; needs simsopt) with one coil per half period
xcoil-coils --surface serial0755658.json --ncoils 1 --order 16 --length 8.25 --total-length \
       --cc 0.094 --cs 0.368 --curvature 4.5 --msc 3.2 --init all

# your own VMEC input file, a quick exploratory run
xcoil-coils --surface input.my_device --ncoils 5 --order 6 --length 30 --total-length \
       --nq 128 --certify none
```

## 8. Speed

* Each target surface is compiled once (5–40 s); then each solve takes seconds on a CPU and well
  under a second to a few seconds on a GPU.
* `--nq` sets the points per coil. The default resolves the distance limits on long coils
  (nq ≥ coil length / (0.06 · min(cc, cs))) and is conservative: the default problem needs 672
  points with it, while `--nq 128` gives the same certified design several times faster.
* On a GPU (`xcoil[cuda]`) the whole solve runs on the device with JAX's own linear algebra.

## 9. When a result is not certified

* **Linked coils**: the starting coils were linked, or (rarely) coils crossed; try another `--size`
  (smaller), another `--init`, or `--init all`.
* **A limit slightly violated** (`v` above 10⁻³): try another start, or increase `--nq`.
* **Iteration limit** (60 outer iterations): the limits may be too tight for the number of coils and
  their length; relax them, add coils, or raise the Fourier order.
* The flux error is a local optimum of the starting coils; different starts can give different
  designs, which is what `--init all` explores.

## Scope

Stellarator-symmetric coils and surfaces, fixed target surface, filamentary coils as Fourier series.
The engineering limits are in [constraints.md](constraints.md). Surfaces must be representable as R, Z Fourier series in the cylindrical angle (very
strongly elongated surfaces that fold back toroidally are not).
