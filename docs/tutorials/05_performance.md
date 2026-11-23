# Tutorial 5: performance

Where the time goes, and how to make XCoil fast: compilation and its reuse, points per coil, precision, the GPU and
the compile cache. Script: [`examples/tutorial_05_performance.py`](../../examples/tutorial_05_performance.py).

```bash
python examples/tutorial_05_performance.py
```

## Output (laptop CPU)

```
JAX devices: [CpuDevice(id=0)]
nq  64 mixed: first solve  16.7 s (compilation ~ 0.3 s), solve 16.41 s, mean |B.n|/|B| 0.5028 %
nq  64 fp64 : first solve  20.0 s (compilation ~ 0.0 s), solve 20.50 s, mean |B.n|/|B| 0.5028 %
nq 128 mixed: first solve  26.0 s (compilation ~ 4.4 s), solve 21.59 s, mean |B.n|/|B| 0.5027 %
CFQS, same structure as the first LP QA problem: first solve 18.3 s, solve 17.70 s (no new compilation)
```

`solve_all` first solves once to compile (`t_compile`: that first solve, compilation included), then solves again.
Here the persistent compile cache was off, but most of the compilation was already in memory: the earlier problems
of the same structure had compiled it.

## What matters

| lever | effect |
|---|---|
| **compilation** | the first solve of a problem *structure* (field periods, coils, order, points per coil, grids, opt-in rows) compiles: 5–40 s. Every later problem with the same structure reuses the programs, whatever its boundary and limit values (CFQS above). Keep `nq` and `grid_dist` fixed across a scan to profit |
| **compile cache** | compiled programs are stored on disk (`~/.cache/xcoil/jax`, or `XCOIL_COMPILE_CACHE`), so a new process skips compilation too |
| **points per coil** `nq` | the cost grows with nq. The default resolves the distance limits conservatively (up to hundreds of points); 64–128 is several times faster for typical problems. Check the certificate (tutorial 2 shows what too few points do) |
| **precision** | `"mixed"` (default): FP32 Biot–Savart kernels with FP64 geometry and accumulation, the same designs; on GPUs several times faster than `"fp64"` |
| **GPU** | install `xcoil[cuda]`. Each solve takes well under a second to a few seconds. The certificate is FP64; consumer GPUs are slow in FP64, so it is a larger share of the time there |
| **many starts** | `screen=k` finishes only the k most promising ([tutorial 4](04_starts.md)); `device_lm.build_batched_solver` solves B starts of one problem in one program |
| **opt-in rows** | force rows (C·nq dense rows) dominate on a CPU; ordering and Lp-curvature rows are cheap |
| **target** | `target_mean=` stops as soon as the field error reaches a target |
