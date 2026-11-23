# Concepts

## The problem

Given a plasma boundary Σ (a closed toroidal surface), find coils whose magnetic field B is tangent to Σ, so that Σ
is a flux surface of the coils' field. XCoil solves the fixed-boundary ("stage-two") coil problem

    minimize    f_B(x) = Σ_q (w_q / W) (B(x_q) · n_q / B_ref)²         (+ an arclength-variation penalty)
    subject to  g_j(x) ≤ 0   for every engineering limit j

- x: the design vector. For every **base coil** c = 1 … C (the coils of one half field period), the
  `CurveXYZFourier` coefficients of its curve γ_c(s) = (x(s), y(s), z(s)), s ∈ [0, 1): for each of x, y, z, the
  coefficients (c₀, s₁, c₁, …, s_M, c_M) of order M. These are followed by C − 1 **current variables** u, with base
  currents I_c = I₀ (1 + (P u)_c), where P = [I; −1ᵀ]. The currents are free and their sum is fixed: the total current
  only sets the field strength, which the objective divides out.
- x_q, n_q, w_q: points, unit normals and area weights of a grid on Σ (`grid_fine`, per field period, folded by
  symmetry). B_ref is a fixed normalization: the area-weighted RMS of |B| from the starting coils.
- g: the constraint rows, normalized so that g ≤ 0 is feasible and g = 0.1 means 10 % past the limit. The full list
  is in [constraints.md](constraints.md).

The quantity reported to users is the **field error** mean |B·n|/|B| [%]: the area-weighted mean over Σ of
|B·n| / |B|, and its maximum. It is not the objective itself, but it is what the certificate measures.

## Symmetry

A stellarator with nfp field periods and stellarator symmetry is built from 2·nfp copies of the C base coils. Every
base coil is rotated by 2πk/nfp, and mirrored by (x, y, z) → (x, −y, −z) with its current reversed (simsopt's
`coils_via_symmetries`). "4 coils per half period" on an nfp = 2 device means 16 physical coils. The boundary's nfp
sets the coils' nfp. Only stellarator-symmetric boundaries are supported.

## The solver

`solve_all` (and `device_lm.build_solver`, the program it compiles) solves the problem with a
**Powell–Hestenes–Rockafellar augmented Lagrangian**:

    Φ(x; λ, ρ) = F(x) + (ρ/2) Σ_j max(0, g_j(x) + λ_j/ρ)²

Each outer iteration minimizes Φ for fixed multipliers λ and penalty ρ by **Levenberg–Marquardt** steps on the
Gauss–Newton normal equations (JᵀJ + Aₐᵀ Aₐ ρ + damping), with exact Biot–Savart Jacobians and the exact curvature
of the constraint rows per coil. Then λ ← max(0, λ + ρ g), and ρ grows when the violation stops decreasing.

The first subproblem runs on coarse grids. The whole solve consists of two compiled JAX programs with no host
round-trips. It stops when the true limits hold to `viol_tol` = 10⁻³ and the objective changes by less than
`ftol` = 1 % between outer iterations, or after `max_outer` outer iterations.

- **Precision**: `precision="mixed"` (default) computes geometry in FP64 and the Biot–Savart kernels in FP32 with
  FP64 accumulation, which is fast on GPUs. `"fp64"` is all double precision. The certificate is always FP64.
- **Margins**: the solve applies the curvature and MSC limits 3 % inside the given values and the distance limits
  1 % inside. The distances in the solve are soft minima (a conservative lower bound of the true minimum). So a
  converged design passes the certificate, which checks the given values exactly.
- **Compiled once per structure**: the programs take the problem's data (boundary tables, limit values) as inputs.
  Any problem with the same structure reuses one compiled program: same number of coils, order, points per coil,
  grids and switches. The first solve of a structure compiles (5–40 s); later ones do not.
- **Non-convex**: different starting coils can reach different local optima. `starts(prob)` builds a family of
  starts; `solve_all(prob, starts, certify="all")` solves them all; `best` keeps the certified one with the lowest
  field error.

## Certification

The solve works with sampled coils, soft-minimum distances and margins. The **certificate** (`prob.certify(x)`, done
by `solve_all` according to `certify=`) re-measures a design independently, in FP64, against the TRUE limits:

| quantity | how it is measured |
|---|---|
| mean, max \|B·n\|/\|B\| | full torus, `verify_grid` points per field period; the maximum also on a 3× finer grid |
| length | quadrature with 768 points per coil |
| maximum curvature | 4096 samples per coil, then golden-section refinement around the 4 highest peaks |
| mean squared curvature | 4096 samples per coil (arclength-weighted) |
| coil–coil distance | exact segment-to-segment distances between 1024-segment polylines |
| coil–surface distance | exact point-to-triangle distances to the triangulated boundary |
| linking | Gauss linking number of every pair of physical coils |
| force, ordering (if on) | 768 points per coil (`cert["force"]`, `cert["ordering"]`) |

The violation is v = max over the limits of max(0, (value − limit)/limit). A design is **certified** when v ≤ 10⁻³
(0.1 %), no pair of coils is linked, and nothing is NaN. Certify every design you report or use.

## Units

Lengths in metres, curvature in 1/m, MSC in 1/m², currents in A, forces per unit length in N/m, the field in T.
The field error is dimensionless (reported in %). Limits scale with the device: a boundary twice as large needs twice
the lengths and distances, half the curvature and a quarter of the MSC for the same design.

The solver normalizes the base-coil currents to 1 MA each. Exports rescale them to the total current you give
(`total_current=`). The field error does not depend on that choice, but forces do (∝ I²):
`force_limit_in_problem` converts a force limit given at your real current.

## Using XCoil inside an outer optimization

The optimal field error F*(s) depends on the boundary coefficients s. By the envelope theorem, at a solution x* with
multipliers z* of the active limits:

    dF*/ds = ∂/∂s [ F(x*; s) + z*ᵀ g(x*; s) ]      (coils and multipliers fixed)

This costs no extra solve. It needs a KKT point of F itself: solve with `irls=0` in the budget (`solve_all`'s default
IRLS reweighting steers towards the L1 field ratio instead). [Tutorial 7](tutorials/07_outer_optimization.md) checks
it against re-solves (agreement 2 × 10⁻⁴).
QSS ([qss-filaments](https://github.com/PedroFranciscoGil/qss-filaments)) uses it to optimize the plasma and the coils together.
