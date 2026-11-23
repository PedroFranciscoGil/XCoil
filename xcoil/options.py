"""Descriptions of every setting of XCoil, used to generate docs/reference.md (tools/gen_reference.py).

tests/test_reference.py fails when a public function has no docstring, a command-line option has no help text, an
HConfig field, solver-budget key or XCOIL_* environment variable is not described here, or docs/reference.md is out
of date.
"""

# public API: (module, names) documented in docs/reference.md and docs/api.md
PUBLIC_API = [
    ("xcoil.fixed.user_problem", ["load_boundary", "read_vmec_input", "read_wout", "write_vmec_input", "refit_rz",
                                  "settings", "make_problem", "initial_coils", "starts", "pair", "warm_start",
                                  "quick_check", "solve_all", "best", "force_limit_in_problem",
                                  "force_to_user", "physical_coils", "write_coils_json", "write_makegrid",
                                  "write_vtk", "write_simsopt"]),
    ("xcoil.homotopy.boundary", ["FourierBoundary", "from_modes", "load", "save", "from_simsopt", "fit_grid"]),
    ("xcoil.constraints.forces", ["coil_forces", "max_force", "force_extra", "certify_force", "normalized_limit",
                                  "regularization_circ"]),
    ("xcoil.constraints.lp_curvature", ["lp_curvature", "lp_rows", "lp_extra", "certify_lp"]),
    ("xcoil.constraints.ordering", ["axis_from_boundary", "axis_from_modes", "axis_from_points", "rows", "report",
                                    "ordering_extra", "certify_ordering", "pair_names"]),
    ("xcoil.fixed.device_lm", ["build_solver", "build_batched_solver", "Extra", "make_monitor"]),
]

# the coil problem's settings (xcoil.homotopy.config.HConfig) read by solve_all / build_solver / certification;
# user_problem.make_problem sets the ones marked "make_problem"
HCONFIG_DOC = {
    "surface": ("", "Name of the target boundary (bundled name or user/<key>); set by make_problem."),
    "n_coils": ("", "Base coils per half field period (make_problem ncoils)."),
    "stellsym": ("", "Stellarator symmetry of the coil set (always True in make_problem)."),
    "kappa_lim": ("1/m", "Maximum pointwise curvature (make_problem kappa)."),
    "msc_lim": ("1/m^2", "Maximum mean squared curvature per coil (make_problem msc)."),
    "dcc": ("m", "Minimum coil-coil distance (make_problem dcc)."),
    "dcs": ("m", "Minimum coil-surface distance (make_problem dcs)."),
    "eps_lo": ("m", "Lower end of the length-cap range of the homotopy tables."),
    "eps_hi": ("m", "Upper end of the length-cap range (the length limit at t = 1); make_problem sets both to the length limit."),
    "basis": ("", "Coil representation: 'fourier' (simsopt CurveXYZFourier) or 'bspline'."),
    "order": ("", "Fourier order of every coil (make_problem order)."),
    "n_ctrl": ("", "B-spline control points (basis 'bspline' only)."),
    "nq_opt": ("", "Quadrature points per coil in the solve (make_problem nq; default rule in settings())."),
    "grid_fine": ("", "(nphi, ntheta) flux grid on the boundary for |B.n| (per field period)."),
    "grid_dist": ("", "(nphi, ntheta) full-torus grid of the coil-surface distance soft-min."),
    "path": ("", "Boundary path of the homotopy tables; 'linear' (the fixed-boundary solve uses t = 1)."),
    "t_h": ("", "Homotopy parameter at which the length cap reaches its target (1 in the fixed-boundary solve)."),
    "disp_guard": ("", "Largest coil displacement per continuation step as a fraction of min(dcc, dcs)."),
    "f_ref": ("", "Reference value of the flux objective: the objective is |r|^2 / f_ref (scaling only)."),
    "lagrangian_hessian": ("", "Second-order term of the LM model: 'exact_blockdiag' (default), 'gn' or 'blockdiag'."),
    "beta_curv": ("", "Relative margin of the curvature and MSC rows in the solve (3 %: the certificate passes)."),
    "beta_d": ("", "Relative margin of the distance rows; overridden by the solver budget's dist_margin."),
    "front_param": ("", "Which limit the homotopy tables ramp: 'length' (the length cap) or 'kappa' (the curvature limit)."),
    "front_L": ("m", "Fixed length cap when front_param is 'kappa'."),
    "w_msc_final": ("", "Weight of an MSC penalty in the objective (0: off)."),
    "softmin_alpha": ("1/m", "Sharpness of the soft-min of the distance rows (conservative: below the true min)."),
    "dist_segments": ("", "Distance rows per coil: 0 = one per quadrature point, n = soft-min over n segments."),
    "verify_grid": ("", "(nphi, ntheta) per field period of the certificate's triangulated boundary."),
    "nq_verify": ("", "Quadrature points per coil of the certificate (length, distances)."),
    "viol_tol": ("", "Certified when every true limit holds to this relative violation (1e-3)."),
    "precision": ("", "Precision of the homotopy tables ('fp64'); the solve's precision is solve_all(precision=)."),
    "seed": ("", "Random seed of randomized start variants."),
    "per_coil_length": ("", "True: the length limit applies to every coil; False: to all base coils together."),
    "arc_var_lim": ("m^2", "Per-coil arclength-variation constraint (0: off)."),
    "arc_pen_thresh": ("m^2", "Arclength-variation penalty threshold T (make_problem: simsopt auglag default 0.01)."),
    "arc_pen_weight": ("", "Weight of the arclength-variation penalty in the objective."),
    "arc_pen_intervals": ("", "Arclength-variation intervals: 0 every point, -1 2*order intervals, n>0 n intervals."),
    "beta_L": ("", "Initial length cap of the homotopy: (1 + beta_L) times the seed's length."),
    "w_arc": ("", "Weight of the arclength-uniformity regularizer (a gauge term in units of f_ref; 1e-3)."),
    "w_L0": ("", "Weight of the length regularizer, fading as (1 - t)^2 (0 at the target boundary, t = 1)."),
    "w_msc0": ("", "Weight of an MSC regularizer fading as (1 - t)^2 (0 at t = 1)."),
    "msc_final_ramp": ("", "How w_msc_final enters: 't2' (ramped in as t^2) or 'none' (constant)."),
}

# solver budget: the dict passed to device_lm.build_solver (solve_all uses device_lm.MAIN; device_lm.DEFAULTS
# fills the keys it does not set)
BUDGET_DOC = {
    "precision": "'mixed' (FP64 geometry, FP32 Biot-Savart kernels, FP64 accumulation; default of solve_all) or "
                 "'fp64'.",
    "inner": "Levenberg-Marquardt steps per outer (augmented-Lagrangian) iteration.",
    "inner_first": "LM steps of the first outer iteration (the first subproblem, on the coarse grids).",
    "first_prog": "Stop the first subproblem when 10 steps gain less than this fraction of its objective.",
    "first_rel": "Stop the first subproblem at |grad| <= first_rel |grad_0| (off by default).",
    "omega_first": "Stationarity tolerance of the first subproblem (None: 1/rho0; 0: run all inner_first steps).",
    "rho0": "Initial penalty parameter of the augmented Lagrangian (10).",
    "pen_curv": "Add the curvature of the arclength-variation penalty to the LM model (1: on).",
    "pen_w_first": "Arclength-penalty weight of the first subproblem only.",
    "coarse_grid": "'nphi x ntheta' flux grid of the first subproblem (e.g. '16x32'; empty: no coarse stage).",
    "coarse_dist": "'nphi x ntheta' distance grid of the first subproblem.",
    "coarse_nq": "Points per coil of the first subproblem (default: the problem's).",
    "cc_rows": "Coil-coil Jacobian rows: only the K most nearly active rows when few are active (0: all).",
    "near_rows": "Near-active row window: rows with c > -near enter the LM model (0: active rows only).",
    "damp": "LM damping: 'fourier' (mu diag((1 + m)^2), default) or 'levenberg' (mu I).",
    "rho_rule": "Penalty update: 'guard' (default), 'conv', 'lancelot' or 'algencan' (see device_lm._al_opts).",
    "rho_fac": "Penalty growth factor (10).",
    "rho_max": "Penalty cap.",
    "rho_tau": "Required decrease factor of the violation measure V for 'conv' / 'algencan' (0.25).",
    "rho_stall": "Stalled outers before 'guard' raises the penalty (3).",
    "irls": "One IRLS reweighting per outer towards the L1 field ratio sum w|B.n| / sum w|B| (eps; 0: off).",
    "dist_margin": "Relative margin of the coil-coil and coil-surface rows in the solve (0.01).",
    "cs_window": "Coil-surface soft-min over the K nearest distance-grid points of each coil point (0: all).",
    "ftol": "Stop when the violation is <= viol_tol and |F_k - F_{k-1}| <= ftol F_k.",
    "link_test": "Reject trial steps that change a coil-pair linking number (True).",
}

# environment variables
ENV_DOC = {
    "XCOIL_COMPILE_CACHE": "Persistent JAX compilation cache folder, or 'off' (default: JAX_COMPILATION_CACHE_DIR "
                           "if set, else ~/.cache/xcoil/jax). Set before importing xcoil.",
    "XCOIL_PRECISION": "'fp64' (default; enables JAX x64), 'fp32' or 'mixed' for the homotopy tables.",
    "XCOIL_TABLES": "'jax' (default): boundary tables and start checks on the JAX device; 'numpy': on the host "
                    "(equal at rounding level).",
    "XCOIL_EVAL_CHUNK": "Points x modes per boundary evaluation chunk (default 2^26; bounds memory for big boundaries).",
    "XCOIL_USER_DIR": "Folder of boundaries registered by make_problem (default ~/.cache/xcoil/boundaries).",
    "XCOIL_REF_DIR": "Tests only: folder with the reference solutions some tests compare against (not shipped).",
}
