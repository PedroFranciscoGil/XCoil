# Objectives and constraints

Every term XCoil optimizes or constrains. Constraint rows are normalized and read **g ≤ 0** (feasible); g = 0.1 means
10 % past the limit. "Solve" is what the optimizer enforces (with margins), "certificate" is what `prob.certify` and
`solve_all(certify=...)` check against the true limit (see [concepts.md](concepts.md#certification)).

Summary:

| | term | default | switch |
|---|---|---|---|
| O1 | field error f_B (squared normal field) | always | - |
| O2 | arclength-variation penalty | on in `make_problem` (T = 0.01 m²) | `make_problem(arc=...)` |
| O4 | MSC penalty | off | `HConfig.w_msc_final` |
| C1 | coil length (per coil or total) | on | `make_problem(length=, per_coil_length=)` |
| C2 | maximum curvature | on | `kappa=` |
| C3 | mean squared curvature | on | `msc=` |
| C4 | coil–coil distance | on | `dcc=` |
| C5 | coil–surface distance | on | `dcs=` |
| C6 | no linking | on | `solve_all(link_test=)` |
| C7 | currents (fixed total, free split) | on | - |
| C8 | arclength-variation constraint | off | `HConfig.arc_var_lim` |
| C9 | maximum coil force | off | `solve_all(max_force=...)` |
| C10 | toroidal ordering about a fixed axis | off | `solve_all(ordering=...)` |
| C11 | Lp curvature | off | `solve_all(lp_curvature=...)` |
| S1 | field-error target (stopping rule) | off | `solve_all(target_mean=)` |

Not in XCoil: bounds on individual currents, torsion.

---

## O1. Field error f_B

- **Definition**: f_B = Σ_q (w_q/W) (B(x_q)·n_q / B_ref)² on the flux grid (`grid_fine` per field period), B_ref the
  area-weighted RMS of |B| from the starting coils (fixed). The objective is F = f_B / f_ref (+ O2, O4).
- **Reported**: mean \|B·n\|/\|B\| [%] (area-weighted) and its maximum, by the quick check and the certificate.
- **Certificate**: on the full torus with `verify_grid` points per field period; the maximum also on a 3× finer grid.
- **Cost**: the main cost of a solve (Biot–Savart from every coil point to every grid point, and its Jacobian).

## O2. Arclength-variation penalty

- **Definition**: residual rows √w · max(0, Var_c − T) per base coil, Var_c the variance of |γ'| over intervals of
  the coil (simsopt `ArclengthVariation`); keeps the parametrization near uniform speed. A gauge term: it barely
  changes the coil shapes.
- **Parameters**: `arc_pen_thresh` T [m²] (default in `make_problem` 0.01, simsopt `auglag_qa.py`), `arc_pen_weight`
  w (1), `arc_pen_intervals` (−1: 2·order intervals). Override with `make_problem(..., arc=dict(arc_pen_thresh=...))`;
  `arc=dict(arc_pen_thresh=0)` switches it off. Rescale T by (size)² for devices of another size.
- **Certificate**: reported (`arc_pen_*` entries), not a limit.

## O4. MSC penalty

- **Definition**: residual rows √w_msc κ_ck / κ_lim (a penalty on curvature, not the limit C3). Off (`w_msc_final = 0`).
  Prefer the hard limit C3.

## C1. Coil length

- **Definition**: per coil (`per_coil_length=True`, simsopt `auglag_qa.py` convention): g_c = (L_c − ε)/ε for every
  base coil; total (`per_coil_length=False`): g = (Σ_c L_c − ε)/ε over the C base coils. L_c = mean over the
  quadrature of |γ'|.
- **Parameters**: `length` ε [m], `per_coil_length` (`--per-coil` / `--total-length` on the command line).
- **Defaults**: CLI 4 m per coil.
- **Certificate**: 768-point quadrature, no margin (`L`, `L_max_coil` in the certificate).
- **Pitfall**: usually the binding limit: the field error falls steeply with the length budget.

## C2. Maximum curvature

- **Definition**: pointwise, g_ck = (κ_ck − (1 − β) κ_lim)/κ_lim at every quadrature point k of every coil c,
  κ = |γ' × γ''| / |γ'|³, β = 0.03 (`beta_curv`, the solve's margin).
- **Parameters**: `kappa` κ_lim [1/m]. **CLI default** 3.
- **Certificate**: 4096 samples per coil plus golden-section refinement of the 4 highest peaks, no margin
  (`max_kappa`).
- **Pitfall**: between quadrature points the curvature can peak higher; the 3 % margin covers it. With very few
  points per coil (`nq` < 16 × order) raise the margin or `nq`.

## C3. Mean squared curvature

- **Definition**: per coil, g_c = (MSC_c − (1 − β) MSC_lim)/MSC_lim, MSC_c the arclength-weighted mean of κ²
  (simsopt `MeanSquaredCurvature`), β = 0.03.
- **Parameters**: `msc` [1/m²]. **CLI default** 4.
- **Certificate**: 4096 samples per coil (`msc_max`).

## C4. Coil–coil distance

- **Definition**: per coil point, g_ck = ((1 + β_d) d_cc − softmin_cc(c, k))/d_cc, the soft minimum
  −(1/α) log Σ exp(−α d) over every point of every other physical coil (all field periods and stellarator images;
  the coil's own copy excluded). The soft minimum is a lower bound of the true minimum. β_d = `dist_margin` (0.01).
- **Parameters**: `dcc` [m]; α = `softmin_alpha`/d_cc (100/d_cc); `dist_segments` n > 0: one row per segment.
- **CLI default** 0.1 m.
- **Certificate**: exact segment-to-segment distances between 1024-segment polylines (`d_cc`).
- **Cost**: C·nq rows with dense Jacobians; only the K most nearly active rows get Jacobian rows when few are active
  (`cc_rows`).

## C5. Coil–surface distance

- **Definition**: per coil point, g_ck = ((1 + β_d) d_cs − softmin_cs(c, k))/d_cs over the full-torus distance grid of
  the boundary (`grid_dist`).
- **Parameters**: `dcs` [m]. **CLI default** 0.15 m. `cs_window` K: soft-min over the K nearest grid points.
- **Certificate**: exact point-to-triangle distances to the triangulated boundary (`d_cs`).
- **Pitfall**: a coarse distance grid under-resolves the boundary between grid points; the default grid size follows
  from the boundary size and d_cs. If a design fails the certificate only on d_cs, `solve_all` re-solves it once with
  a larger distance margin (repair).

## C6. No linking

- **Definition**: no pair of physical coils may be linked (Gauss linking number 0). Not a constraint row: trial steps
  that change a linking number are rejected (`link_test=True`), so the solve keeps the topology of its starting coils.
- **Switch**: `solve_all(link_test=False)` (CLI `--no-link-test`) allows such steps.
- **Certificate**: linking number of every physical pair, 512 points per coil (`linked`). A linked design is never
  certified.
- **Pitfall**: linked starting coils give linked results: `initial_coils` reports `check["linked"]`; `starts` drops or
  shrinks linked starts.

## C7. Currents

- **Definition**: free base currents I_c = I₀ (1 + (P u)_c) with Σ_c I_c fixed (C − 1 variables u, unbounded). Exports
  rescale them to `total_current`.
- **Reported**: `I_rel_min`, `I_rel_max` (base currents over their mean) in the certificate.
- **Pitfall**: no bounds: a coil can end with a small or reversed current. Check `I_rel_min`.

## C8. Arclength-variation constraint

- **Definition**: per coil, Var_k |γ'_ck| ≤ `arc_var_lim` [m²] (simsopt `ArclengthVariation`, full intervals), a hard
  row instead of the penalty O2. Off (0). Set it through the problem's `HConfig` (not exposed by `make_problem`).

## C9. Maximum coil force

- **Definition**: per coil point, g_ck = F_ck / F_lim − (1 − margin), F = |I_c t × (B_mutual + B_self,reg)| the force
  per unit length: B_mutual the Biot–Savart field of every other physical coil, B_self,reg the regularized self-field
  of a round conductor of radius a (δ = a²/√e; Landreman, Hurwitz & Antonsen), t the unit tangent (simsopt
  `coil_force` physics, `xcoil/constraints/forces.py`).
- **Switch**: `solve_all(prob, starts, max_force=dict(F_lim=..., radius=..., margin=0.005, nq=768))`; CLI
  `--max-force N/m --force-radius m --force-margin`. F_lim is in the solver's normalization (base coils at 1 MA each):
  convert a limit at your real total current with `up.force_limit_in_problem(prob, F, total_current)`, and back with
  `up.force_to_user`. `forces.normalized_limit` converts a per-turn limit at another device scale.
- **Certificate**: maximum per coil on `nq` (768) points, `cert["force"]` = {max_force, F_lim, violation, ok}; the
  design is certified only if ok. The solve checks the force at the problem's quadrature points, the certificate at 768:
  keep a margin (0.5 % at 64 points per coil).
- **Cost**: C·nq rows with dense Jacobians (forward mode): fast on a GPU, minutes on a laptop CPU.
- **Pitfall**: forces scale as I²: a limit only means something with the current it was given for.

## C10. Toroidal ordering about a fixed reference axis

- **Definition**: neighbouring coils (and each end coil with its stellarator image across φ = 0 and φ = π/nfp) must
  not cross in the (φ, θ) projection. φ is a coil point's cylindrical angle, θ its poloidal angle about a **fixed
  reference axis** R_ax(φ) = Σ aₙ cos(n nfp φ), Z_ax(φ) = Σ bₙ sin(n nfp φ). For each pair (left L, right R) and every
  point k of the left coil, E_k = softmin_k' [φ_R,k' − φ_L,k + κ (1 − cos(θ_L,k − θ_R,k'))] (a lower bound of the hard
  minimum) and the row is (gap − E_k) / (π/(nfp C)), soft-maxed over `n_seg` segments: (C + 1)·n_seg rows.
- **Reference axis**: by default the boundary's geometric axis, its m = 0 modes (aₙ = rc(0,n), bₙ = −zs(0,n), the
  poloidal mean of R and Z at each φ), fixed: the rows depend on the coils only, not on the surface. Pass another as
  `axis=(a, b)` (`ordering.axis_from_boundary`, `axis_from_modes`, `axis_from_points`).
- **Switch**: `solve_all(prob, starts, ordering=dict(gap_deg=1.0, kappa=20.0, beta=500.0, n_seg=16, axis=None))`;
  CLI `--ordering --ordering-gap deg`. `n_seg` must divide the points per coil.
- **Certificate**: `cert["ordering"]`: per pair the windowed gap (smallest φ_R − φ_L over point pairs within 0.03 rad
  in θ; negative = crossing) and the penalized gap [deg]; ok when nothing crosses and every penalized gap reaches
  `gap_deg`; the design is certified only if ok.
- **Pitfall**: it prevents crossing, not linking (C6). Starting coils that cross must first untangle: start from
  ordered coils (all `starts` are).

## C11. Lp curvature

- **Definition**: per base coil, g_c = J_c / J_max − 1 with J_c = (1/p) ∫ max(κ − κ₀, 0)^p dl (simsopt
  `LpCurveCurvature`; `xcoil/constraints/lp_curvature.py`). With κ₀ > 0 only the parts curved more than κ₀ count: it
  limits sharp bends with C rows instead of the C·nq pointwise rows of C2.
- **Switch**: `solve_all(prob, starts, lp_curvature=dict(J_max=..., p=2.0, threshold=0.0, nq=4096))`; CLI
  `--lp-curvature-max --lp-p --lp-threshold`. J_max in 1/m^(p−1) (κ in 1/m).
- **Certificate**: J_c on 4096 points per coil, `cert["lp_curvature"]` = {lp_curvature, J_max, violation, ok}
  (ok: violation ≤ 10⁻³); certified only if ok.
- **Cost**: C rows, cheap.

## S1. Field-error target

- `solve_all(target_mean=0.6)` (CLI `--target-mean`) stops a solve at the first outer iteration whose mean
  \|B·n\|/\|B\| on the certification grid is ≤ 0.6 % with the limits met; otherwise the solve converges (|ΔF| ≤ 1 % of F
  between outers with the limits met) or stops after `max_outer` outer iterations.
