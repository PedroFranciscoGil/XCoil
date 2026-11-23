# Tutorial 7: XCoil inside an outer optimization

Single-stage and quasi-single-stage stellarator optimization change the plasma boundary s while keeping it
reachable by coils. That needs the optimal coil error F*(s) and its gradient. The **envelope theorem** gives the
gradient without re-solving the coils:

    dF*/ds = ∂/∂s [ F(x*; s) + z*ᵀ g(x*; s) ]        at fixed coils x* and multipliers z*

This tutorial computes it for one boundary coefficient and checks it against re-solved coils. Script:
[`examples/tutorial_07_outer_optimization.py`](../../examples/tutorial_07_outer_optimization.py). QSS
([qss-filaments](https://github.com/PedroFranciscoGil/qss-filaments)) is the full optimizer built on this: it uses XCoil's constraints with its own coil-solver loop and the
envelope gradient of the coil subproblem inside an augmented-Lagrangian outer loop over the plasma boundary.

```bash
python examples/tutorial_07_outer_optimization.py
```

## The steps

1. **A converged optimum of F itself.** Finite differences of optimal values need the optimum far more accurately
   than `solve_all`'s default stall test (1 % change of the objective). `solve_all` also reweights the field residual
   towards the L1 field ratio (IRLS), so its optimum is not a KKT point of F. Solve F directly:

   ```python
   from xcoil.fixed.device_lm import build_solver
   from xcoil.fixed.device_lm import MAIN
   solve = build_solver(prob, dict(MAIN, irls=0.0), max_outer=200, viol_tol=1e-7, ftol=1e-9)
   x, info = solve(start_x)
   ```

2. **Multipliers** of the active limits from stationarity, ∇F + Aₐᵀ z = 0, z ≥ 0 (non-negative least squares on the
   rows with g > −10⁻³):

   ```python
   F0, gF = prob.F_grad(x, 1.0)
   g, A = prob.g_jac(x, 1.0)
   act = np.where(g > -1e-3)[0]
   z, _ = nnls(A[act].T, -gF)
   ```

3. **The envelope gradient**: the Lagrangian F + zᵀg at the fixed x* on problems built for s ± h, differenced. No
   re-solve; the boundary enters only through the problem tables.
4. **The check**: re-solve the coils at s ± h (warm-started from x*) and difference the optimal values.

## Output

```
F* = 5.334126e-01, 10 near-active limits, stationarity |grad F + A^T z| / |grad F| = 1.8e-04
dF*/d rc(1,1) by the envelope theorem: -2.47496e+01
  re-solve finite difference, h = 0.001: -2.47544e+01 (relative difference 1.9e-04)
  re-solve finite difference, h = 0.0003: -2.46749e+01 (relative difference 3.0e-03)
```

The envelope gradient matches the re-solved optima to 2 × 10⁻⁴. It costs two Lagrangian evaluations instead of two
coil solves. For many boundary coefficients, differentiate the Lagrangian with JAX instead of finite differences:
the problem tables are JAX functions of the boundary (QSS does this).

Pitfalls:

- With `solve_all`'s default (IRLS on) the stationarity residual is about 40 % and the "gradient" is off by 20 %.
- A limit that becomes active or inactive between s − h and s + h makes F* non-differentiable there.
- F is normalized by B_ref, which make_problem computes from each problem's starting coils. Compare F only between
  problems built the same way, as here.
