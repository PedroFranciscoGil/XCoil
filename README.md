<p align="center">
  <img src="docs/images/xcoil-logo.png" alt="XCoil" width="520">
</p>

# XCoil

XCoil designs the coils of a stellarator for a given plasma boundary: fast, differentiable, certified coil
optimization in JAX. It finds filament coils, shapes and currents, whose magnetic field is
tangent to the boundary, under hard engineering limits (length, curvature, distances, force, ordering, no linking),
in one compiled program on a CPU or a GPU. Every result can be **certified**: re-measured independently in double
precision against the true limits.

---

## 1. Install

```bash
pip install "xcoil @ git+https://github.com/PedroFranciscoGil/XCoil"
```

That installs XCoil with `jax`, `numpy` and `scipy` for the CPU (Python 3.11+). Pin a release for reproducible runs
(`...XCoil@v2.0.0`).

| you want | install |
|---|---|
| an NVIDIA GPU (Linux, CUDA 12) | `pip install "xcoil[cuda] @ git+https://github.com/PedroFranciscoGil/XCoil"` |
| simsopt surfaces, simsopt coil files, the cross-check tests | add the `simsopt` extra: `xcoil[simsopt]` |
| netCDF4 `wout` files (netCDF3 works without) | add the `netcdf` extra |
| to develop XCoil | `git clone` the repository, then `pip install -e ".[test]"` |

Check the installation (versions, devices, a tiny test solve; ends with `OK`):

```bash
xcoil-check
```

## 2. First coils in 2 minutes

```bash
xcoil-coils --init circle --size half-R0 --nq 128
```

This solves the coil problem of simsopt's `auglag_qa.py` example on the Landreman–Paul precise QA boundary: 4 coils
per half field period, Fourier order 4, at most 4 m per coil, curvature ≤ 3 m⁻¹, mean squared curvature ≤ 4 m⁻²,
coil–coil ≥ 0.1 m, coil–surface ≥ 0.15 m. The end of the output:

```
Best start circle_half-R0: CERTIFIED  mean |B.n|/|B| = 0.5027 %, max = 4.9975 %; length 16.001 m (longest coil 4.000 m), ...
  1 start(s), 1 certified (of 1 checked); solve times 22.89-22.89 s (compilation 31 s, total 71 s)
Output directory: ./output_xcoil/landreman_paul_qa_ncoils4_order4_length4pc_cc0.1_cs0.15_curv3_msc4/ (...)
```

`mean |B.n|/|B|` is the field error (the area-weighted mean of the normal field over the field strength on the
boundary): the smaller, the better the coils make this plasma. `CERTIFIED` means every limit holds when re-measured
on dense samples in double precision. The output folder holds:

| file | what it is |
|---|---|
| `coils_optimized.json` | the coils: Fourier coefficients of every base coil (simsopt `CurveXYZFourier` order), currents, symmetry |
| `coils.<surface>` | all coils in the MAKEGRID coils-file format (for VMEC free boundary, field-line tracers) |
| `optimized_coils.vtu`, `optimized_surface.vts` | ParaView files: the coils, and the boundary coloured by B·n/\|B\| |
| `init_coils.vtu`, `init_surface.vts` | the same for the starting coils |
| `summary.json` | settings, every start's result and certificate |
| `biot_savart_optimized.json` | a simsopt `BiotSavart` file (only with simsopt installed) |

The same from Python, in [tutorial 1](docs/tutorials/01_first_coils.md):

```python
from xcoil.fixed import user_problem as up

b, info = up.load_boundary("landreman_paul_qa")
prob, settings = up.make_problem(b, "lpqa", ncoils=4, order=4, length=4.0, per_coil_length=True,
                                 kappa=3.0, msc=4.0, dcc=0.1, dcs=0.15, nq=64)
start, check = up.initial_coils(prob, init="circle", size="half-R0")
results, t_compile = up.solve_all(prob, [start])
r = up.best(results)
print(r["cert"]["certified"], r["cert"]["mean_bn_pct"])
up.write_coils_json(prob, r["x"], "coils.json", total_current=3e5)
```

## 3. How it works

```
  target boundary ──┐
  (bundled name,    │      ┌───────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
  VMEC input, wout, ├────► │ coil problem  ├───► │     solve     ├───► │   certify    ├───► │    export    │
  simsopt / DESC)   │      │ make_problem  │     │   solve_all   │     │ (FP64, dense,│     │ JSON, MAKEGRID│
  coil settings ────┤      │ + starting    │     │ augmented     │     │  true limits,│     │ VTK, simsopt │
  (coils, order)    │      │   coils       │     │ Lagrangian +  │     │  linking)    │     │              │
  limits ───────────┘      └───────────────┘     │ Gauss-Newton  │     └──────────────┘     └──────────────┘
                                                 └───────────────┘
```

1. **Problem**: the boundary, the number of coils per half period and their Fourier order fix the unknowns (the
   coils' Fourier coefficients and their currents, with the total current fixed); the limits become constraint rows
   g ≤ 0. Starting coils are built from the boundary alone (offsets of the surface, planar offsets, circles).
2. **Solve**: minimize the squared normal field on the boundary under the limits with an augmented Lagrangian whose
   subproblems are solved by Levenberg–Marquardt steps on the Gauss–Newton normal equations, all on the JAX device.
3. **Certify**: re-measure the design independently in double precision: the field error on a finer grid, exact
   distances, densely sampled curvature, lengths, and the linking number of every coil pair.
4. **Export** the coils for simsopt, VMEC (MAKEGRID), ParaView, or your own code.

[docs/concepts.md](docs/concepts.md) has the formulation, the solver and what certification checks.

## 4. What matters

| setting | effect |
|---|---|
| **coils per half period** (`ncoils`) | more coils: lower field error, harder engineering (closer coils) |
| **Fourier order** (`order`) | more shaping per coil: lower field error, more unknowns and curvature to control |
| **length limit** (`length`, per coil or total) | usually the most important limit: longer coils follow the plasma better |
| **curvature, MSC, distances** | the engineering limits; tight ones raise the field error. See [constraints.md](docs/constraints.md) |
| **force, ordering** (opt-in) | maximum force per unit length; no crossing of neighbouring coils |
| **starting coils** (`init`, `size`) | the problem is non-convex: different starts reach different designs. Try several (`--init all`) and keep the best certified one |
| **warm vs cold start** | after changing a limit, continue from the previous design (`warm_start`) instead of starting over |
| **CPU vs GPU** | the same results; a GPU solves in well under a second to a few seconds, a laptop CPU in seconds to a minute. The first solve of a problem structure also compiles (5–40 s) |
| **points per coil** (`nq`) | the default resolves the distance limits conservatively; 64–128 is much faster for typical problems |
| **flux target** (`target_mean`) | stop as soon as the field error reaches a target instead of converging |

## 5. Documentation

| page | content |
|---|---|
| [docs/tutorials/](docs/tutorials/README.md) | 7 tutorials with runnable scripts: first coils, engineering limits, your own boundary, starts and warm starts, performance, export, XCoil inside an outer optimization |
| [docs/concepts.md](docs/concepts.md) | the problem, the solver, certification, symmetry, units |
| [docs/boundaries.md](docs/boundaries.md) | every way to give a target boundary |
| [docs/constraints.md](docs/constraints.md) | every objective and constraint: definition, parameters, defaults, certificate, cost |
| [docs/cli.md](docs/cli.md) | the `xcoil-coils` command: every option, outputs, examples |
| [docs/api.md](docs/api.md) | the Python API by task |
| [docs/reference.md](docs/reference.md) | every option, function, setting and environment variable (generated from the code) |
| [docs/troubleshooting.md](docs/troubleshooting.md) | not certified, infeasible limits, slow, memory, compile cache, x64 |

## 6. Tests

```bash
pytest -m "not slow and not bigmem"
```

The CPU suite of the CI (each file in its own process there). `pytest -m slow` adds the end-to-end tutorials and the
long solves; `pytest -m bigmem` the tests that build dense autodiff Jacobians of production-size problems (5–10 GB of
RAM). Tests that compare with simsopt run only when simsopt is installed.

## License and authors

MIT, Copyright (c) 2026 Pedro Francisco Gil and Alan A. Kaptanoglu ([LICENSE](LICENSE), [AUTHORS.md](AUTHORS.md)).
