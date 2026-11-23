# Troubleshooting

**Start with `xcoil-check`.** It reports the versions, the JAX devices, whether x64 is on, the compile cache, the
bundled boundaries and the optional packages, and runs a tiny solve. It ends with `OK`.

## Not certified

The certificate says which limit fails: look at `cert["v"]` (largest violation) and the values next to the limits.

| symptom | what to do |
|---|---|
| `linked` | the starting coils were linked (`check["linked"]` of `initial_coils`) or coils crossed: a smaller `size`, another `init`, or `starts(prob)` (only admissible starts) |
| `v` slightly above 10⁻³ on a distance | `solve_all` already re-solves once with a larger distance margin (repair); otherwise more points per coil (`nq`) or another start |
| `v` slightly above 10⁻³ on curvature or MSC | more points per coil (`nq` ≥ 16 × order): the curvature can peak between the solver's points |
| force or Lp curvature just over | a larger `margin` (force) or a slightly lower `J_max`: the solve checks the problem's points, the certificate 768 / 4096 |
| ordering not ok | starting coils crossed in (φ, θ): start from ordered coils (every `starts` start is) |
| stopped at `max_outer` | the limits may be infeasible for this many coils of this length: relax a limit, add coils, raise the order, or raise `max_outer` |

## Infeasible limits

There is no general test for feasibility. Signs: the violation `v` stalls well above 10⁻³ while the penalty grows,
from every start. Typical causes: a length budget too short for the coil–surface distance (coils must reach around
the plasma at d_cs), coil–coil distances too large for the number of coils (C coils per half period share a toroidal
gap of π/(nfp C) at the inboard side), limits copied from a device of another size ([boundaries.md](boundaries.md#size-and-limits)).
Relax one limit at a time, warm-starting from the previous design (`warm_start`).

## Slow on a CPU

- The first solve of a problem structure compiles (5–40 s). Later solves with the same structure reuse it, and with a
  compile cache so do later processes (below).
- Points per coil: the default `nq` is conservative; `nq=64` or `128` is several times faster for typical problems.
- Force rows (C·nq rows with dense Jacobians) take minutes on a CPU: develop without them, add them at the end, or use
  a GPU.
- `--init all` solves up to 19 starts: `screen=1` finishes only the most promising.

## Out of GPU memory

- Lower `nq` or the distance grid (`make_problem(..., grid_dist=(nphi, ntheta))`).
- Large boundaries with many modes: `XCOIL_EVAL_CHUNK` (smaller chunks).
- JAX preallocates 75 % of the GPU: `XLA_PYTHON_CLIENT_PREALLOCATE=false` makes it allocate on demand.

## Compile cache

XCoil turns on JAX's persistent compilation cache when it is imported: in `JAX_COMPILATION_CACHE_DIR` if set, else in
`~/.cache/xcoil/jax`. `XCOIL_COMPILE_CACHE=/dir` picks another folder, `XCOIL_COMPILE_CACHE=off` turns it off. The
cache is for the whole process (your other JAX programs are cached too). Delete the folder if it grows too large;
entries for other JAX versions are never used.

## x64

XCoil needs JAX's 64-bit mode; importing `xcoil` (any module) switches it on. If JAX ran a computation before xcoil
was imported, some arrays may already be 32-bit: import xcoil first, or set `JAX_ENABLE_X64=1`. `xcoil-check` reports
it.

## Import problems

- `ModuleNotFoundError: xcoil` after `pip install`: check that the same Python runs both (`python -m pip install ...`).
- simsopt JSON surfaces need the `simsopt` extra; netCDF4 wout files need the `netcdf` extra.
