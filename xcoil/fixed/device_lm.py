"""XCoil's coil solver: PHR augmented Lagrangian with Levenberg-Marquardt inner steps, on the JAX device.

The augmented Lagrangian F + (rho/2) sum max(0, g + lam/rho)^2 is minimized for fixed (lam, rho) by
Levenberg-Marquardt steps on the Gauss-Newton normal equations (exact Biot-Savart Jacobians, exact
per-coil curvature of the constraint rows, Nielsen damping update, Jacobians only at accepted points).
One outer iteration - the whole inner loop, including rejected trials - is a single jitted program
(lax.while_loop), so on a GPU nothing leaves the device inside an outer iteration.

build_solver(prob, budget) -> solve(x0) -> (x, info) is what user_problem.solve_all runs; DEFAULTS
and MAIN are its budget. build_batched_solver solves many starts of one problem in one program.
Extra adds opt-in constraint rows (forces, ordering, Lp curvature, or your own).
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from xcoil.homotopy import problem_h as ph
from xcoil.homotopy import surface_path as sp
from xcoil.constraints import pointwise as pw
from xcoil.field import residuals as rs

RHO_MAX = 1e8


def _cast(tree, dt):
    return jax.tree_util.tree_map(
        lambda a: a.astype(dt) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) else a, tree)


N_LINK_TEST = 256                       # points per coil of the step test's linking numbers


def _linking_fn(T, n=N_LINK_TEST):
    """x -> rounded Gauss linking numbers (C, N_phys) of every base coil with every physical coil
    (the identity copy of the base coil itself set to 0). Every coil pair is a symmetry image of
    such a pair, and the symmetries are proper rotations, which preserve linking numbers, so this
    covers all pairs. One fused reduction per base coil, written component-wise."""
    from xcoil.curves import fourier as fr
    from xcoil.curves import symmetry as sym_mod
    sym, C = T.sym, T.sym.C
    cbl = fr.make_basis(T.cfg.basis, n, T.cfg.order if T.cfg.basis == "fourier" else T.cfg.n_ctrl)
    self_mask = np.stack([(np.asarray(sym.coil) == c) & np.asarray(sym.ident) for c in range(C)])   # (C, J)

    def links(x):
        g, gd, _ = fr.coils_curves(x, cbl, C)
        G, Gd = sym_mod.expand(g, sym), sym_mod.expand(gd, sym)              # (J, n, 3)
        rows = []
        for c in range(C):
            dx = g[c, :, 0][None, :, None] - G[:, None, :, 0]
            dy = g[c, :, 1][None, :, None] - G[:, None, :, 1]
            dz = g[c, :, 2][None, :, None] - G[:, None, :, 2]
            a, b = gd[c][None, :, None, :], Gd[:, None, :, :]
            cx = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
            cy = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
            cz = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            r2 = dx * dx + dy * dy + dz * dz
            r2 = jnp.where(self_mask[c][:, None, None], 1.0, r2)                # no self term
            rows.append(jnp.sum((cx * dx + cy * dy + cz * dz) * lax.rsqrt(r2) ** 3, axis=(1, 2)))
        Lk = jnp.stack(rows) / (n * n * 4 * jnp.pi)
        return jnp.where(jnp.asarray(self_mask), 0.0, jnp.rint(Lk))

    return links


def irls_weights(T, x, t, eps=1e-2):
    """Per-point weights q of the flux residuals for one IRLS step towards the L1 ratio sum w |B.n| / sum w |B|: with
    r_i = sqrt(w_i / W) (B.n)_i / B_ref, sum_i w_i |B.n|_i ~ sum_i sqrt(w_i) |r_i| ~ sum_i (q_i r_i)^2 near x for
    q_i^2 = sqrt(w_i) / max(|r_i|, eps max|r|); q scaled so that |q r| = |r| at x (the objective's size is kept)."""
    X, n, w = ph.surface(T, t)
    r = rs.magnetic_residual(x, T.cb, T.sym, X, n, w, T.B_ref, T.I0)
    a = jnp.abs(r)
    q2 = jnp.sqrt(w) / jnp.maximum(a, eps * jnp.max(a))
    q = jnp.sqrt(q2)
    return q * jnp.sqrt(jnp.sum(r * r) / jnp.sum((q * r) ** 2))


def _mode_weights(T, n, p=2):
    """Per-variable damping weights (1 + m)^p of the Fourier mode m of each coil coefficient (1 for currents and
    B-spline bases): Levenberg damping in the scaled variables z_i = (1 + m_i)^(p/2) x_i (p = 2: the "fourier" damping)."""
    d = np.ones(n)
    if T.cb.kind == "fourier":
        K = T.cb.K; nc = T.sym.C * 3 * K
        m = (np.arange(nc) % K + 1) // 2
        d[:nc] = (1.0 + m) ** 2 if p == 2 else (1.0 + m) ** float(p)
    return d


def _damp_ok(damp):
    """"levenberg" (H + mu I) | "fourier" (H + mu diag((1 + m)^2))."""
    return damp in ("levenberg", "fourier")


class Extra:
    """Constraint rows added to a solve (build_solver(extra=...)): r(x) extra least-squares residuals (the merit
    gets |r|^2 more), g(x) extra constraints g <= 0 (true limit: offset 0), flux_w the weight of the problem's own
    flux residual rows (0: those rows are left out, the field enters only through r and g; then irls must be 0).
    r and g are JAX functions of the full x; jr / jg their Jacobians (default jax.jacrev). The opt-in limits of
    xcoil.constraints (forces.force_extra, ordering.ordering_extra, lp_curvature.lp_extra) are Extras."""

    def __init__(self, r, g, flux_w=1.0, jr=None, jg=None):
        self.r, self.g, self.flux_w = r, g, float(flux_w)
        self.jr, self.jg = jr or jax.jacrev(r), jg or jax.jacrev(g)

    def m(self, n):
        """Number of constraint rows for a design vector of size n."""
        return int(jax.eval_shape(self.g, jax.ShapeDtypeStruct((n,), jnp.float64)).shape[0])


def _with_extra(extra, T, s, rJ, J32, r_only, g_only, A_of, A32):
    """_build's residual / Jacobian / constraint functions with extra's rows appended (flux rows x flux_w; with
    flux_w = 0 the flux rows are not computed: only the regularizer rows of mx.stacked remain)."""
    Jr, Jg, fw = extra.jr, extra.jg, extra.flux_w
    nf = lambda p: ph.surface(T, p[0])[0].shape[0]   # flux rows come first in mx.stacked

    if fw == 0.0:
        def reg(x, p):
            return s * rs.regularizer_residual(x, p[0], T.cb, T.sym.C, T.cfg, T.L_ref)

        def regJ(x, p):
            return s * rs.regularizer_jacobian(x, p[0], T.cb, T.sym.C, T.cfg, T.L_ref)

        rJ = lambda x, p: (reg(x, p), regJ(x, p))
        J32 = regJ
        r_only = reg
        wr = lambda r, x, p: jnp.concatenate([r, extra.r(x).astype(r.dtype)])
        wJ = lambda J, x, p: jnp.concatenate([J, Jr(x).astype(J.dtype)])
    else:
        def wr(r, x, p):
            k = nf(p)
            return jnp.concatenate([r[:k] * fw, r[k:], extra.r(x).astype(r.dtype)])

        def wJ(J, x, p):
            k = nf(p)
            return jnp.concatenate([J[:k] * fw, J[k:], Jr(x).astype(J.dtype)])

    def rJ2(x, p):
        r, J = rJ(x, p)
        return wr(r, x, p), wJ(J, x, p)

    return (rJ2, lambda x, p: wJ(J32(x, p), x, p), lambda x, p: wr(r_only(x, p), x, p),
            lambda x, p: jnp.concatenate([g_only(x, p), extra.g(x).astype(x.dtype)]),
            lambda x, p, h, c, S=None: jnp.concatenate([A_of(x, p, h, c, S), Jg(x).astype(x.dtype)]),
            lambda x, p: jnp.concatenate([A32(x, p), Jg(x).astype(x.dtype)]))


def _build(T, c_f, precision="fp64", cc_K=0, pen_curv=False, link_test=False, jit=True, near=0.0,
           damp="levenberg", extra=None, cs_window=0):
    """precision:
      fp64     everything FP64;
      fp32jac  Jacobians J, A in naive FP32 (tables and x cast), residuals / constraints FP64;
      mixed    FP64 geometry and separations, FP32 kernels, FP64 accumulation / outputs for the
               flux residual and J (field/mixed.py) and the distance soft-mins in g and A
               (pointwise kdt); length / curvature / MSC rows stay FP64.
    Gradients, normal equations and the merit are FP64 in every mode; at an accepted step the
    trial point's r, g are reused, so the Jacobians are the only extra evaluations.
    cc_K > 0 (fp64 / mixed): only the cc_K coil-coil rows with the largest h = max(0, g + lam/rho)
    get Jacobian rows (the others enter A^T h and A_a^T A_a with weight 0, so this is exact);
    if more than cc_K coil-coil rows have h > 0 the full block is evaluated instead.
    pen_curv: add the arclength-penalty curvature (field/residuals.arc_penalty_curvature) to the
    Gauss-Newton matrix, so the LM model sees the penalty on the reparametrization directions.
    link_test: a trial step that changes any coil-pair linking number (Gauss integral, _linking_fn)
    counts as a failed step (damping increased, as for a failed factorization): coils cannot pass
    through each other, so an unlinked start stays unlinked.
    near > 0 (budget["near_rows"]; warmstart/lm_lab): rows with c = g + lam/rho > -near (inactive but
    within `near` of switching on) enter the Gauss-Newton matrix as rho A_i^T A_i with zero gradient
    contribution, so the LM model bounds the linearized penalty from above and steps no longer walk
    into rows that switch on inside them; the cc_rows top-k ranks by c. near = 0: the code path above.
    damp: "levenberg" (H + mu I) or "fourier" (H + mu diag((1 + m)^2), m the Fourier mode of each coil
    coefficient: Levenberg damping in the mode-scaled variables); mu0 = 1e-3 max diag H in both.
    outer(..., q=None): q (flux rows) multiplies the flux residuals and their Jacobian rows - the IRLS
    weights of irls_weights; the returned F is then the weighted sum of squares.
    extra (None: the code paths above): rows of another problem added to this one (see Extra): extra.r(x)
    appended to the residuals, extra.g(x) to the constraints (Jacobians by jax.jacrev, all columns incl.
    the currents), the flux residuals multiplied by extra.flux_w.
    cs_window = K > 0: inside an outer iteration the coil-surface soft-min (values and Jacobian) runs over the K
    distance-grid points nearest to each coil point at the outer's start instead of the whole grid; the constraints
    returned at the outer's end (multiplier update, stopping test) are evaluated on the whole grid."""
    assert _damp_ok(damp), damp
    links = _linking_fn(T) if link_test else None
    from xcoil.field import mixed as mx
    s = jnp.sqrt(c_f)
    lowJ = precision in ("fp32jac", "mixed")
    T32 = _cast(T, jnp.float32) if lowJ else None

    def rJ(x, p):                         # FP64: the fused kernels of field/mixed.py with kdt=None
        t = p[0]
        r, J = mx.stacked(x, t, T.cb, T.sym, ph.surface(T, t), T.B_ref, T.I0, T.cfg, T.L_ref, kdt=None)
        return s * r, s * J

    def J32(x, p):
        if precision == "mixed":
            t = p[0]
            _, J = mx.stacked(x, t, T.cb, T.sym, ph.surface(T, t), T.B_ref, T.I0, T.cfg, T.L_ref)
            return s * J
        _, J = ph.resid_jac(T32, x.astype(jnp.float32), _cast(p, jnp.float32))
        return s * J.astype(x.dtype)

    def A32(x, p):
        if precision == "mixed":
            S = sp.eval_xyz(T.dist, p[0])
            return pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, ph.cap_at(T, p), jnp.float32)
        _, A = ph.cons_jac(T32, x.astype(jnp.float32), _cast(p, jnp.float32))
        return A.astype(x.dtype)

    C, nq = T.sym.C, T.cb.B0.shape[0]
    bl = pw.blocks_for(C, nq, T.lim)
    cc_sl = bl["cc"]
    kdt = jnp.float32 if precision == "mixed" else None

    def A_of(x, p, h, c, S=None):
        if precision == "fp32jac":
            return A32(x, p)
        S = sp.eval_xyz(T.dist, p[0]) if S is None else S
        cap = ph.cap_at(T, p)
        full = lambda _: pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, cap, kdt)
        if not cc_K or T.lim.n_seg:
            return full(None)
        rank = c if near else h                        # near: rank by c, near-active rows get Jacobian rows
        ok, rows = jnp.asarray(True), {}
        for name, K, sl in (("cc_rows", cc_K, cc_sl),):
            if K:
                K = min(K, sl.stop - sl.start)         # small coil sets: fewer rows than the requested K
                _, idx = lax.top_k(rank[sl], K)
                rows[name] = (idx // nq, idx % nq)
                ok = ok & (jnp.sum(h[sl] > 0) <= K)
        part = lambda _: pw.constraint_jacobian(x, T.cb, T.sym, S, T.lim, cap, kdt, **rows)
        return lax.cond(ok, part, full, None)

    def r_only(x, p):
        t = p[0]
        return s * mx.stacked_residual(x, t, T.cb, T.sym, ph.surface(T, t), T.B_ref, T.I0, T.cfg, T.L_ref,
                                       kdt=jnp.float32 if precision == "mixed" else None)

    def g_only(x, p):                                 # whole-grid constraints (with extra's rows appended below)
        return ph.cons(T, x, p, jnp.float32 if precision == "mixed" else None)

    if extra is not None:                             # appended rows (opt-in constraints or your own)
        rJ, J32, r_only, g_only, A_of, A32 = _with_extra(extra, T, s, rJ, J32, r_only, g_only, A_of, A32)

    mixed = lowJ

    def outer(x, lam, rho, mu, nu, omega, p, inner, rel=False, prog=0.0, q=None):
        """One outer AL iteration: inner LM loop to |grad Phi|_inf / (1 + |grad F|_inf) <= omega, or -
        rel=True, scale invariant - to |grad Phi|_inf <= omega |grad Phi(x_start)|_inf (at most `inner`
        accepted steps). prog > 0: progress stop instead - every 10 accepted steps, stop when the merit
        fell by less than prog * Phi over those steps (an ftol-type test, scale invariant)."""
        n = x.size
        I = jnp.eye(n, dtype=x.dtype)

        def phi_of(r, g):
            h = jnp.maximum(0.0, g + lam / rho)
            return r @ r + 0.5 * rho * (h @ h)

        r_of = r_only
        kdt_g = jnp.float32 if precision == "mixed" else None
        if cs_window and not T.lim.n_seg and precision != "fp32jac":
            from xcoil.curves import fourier as fr
            S_all = sp.eval_xyz(T.dist, p[0])
            gb0 = fr.coils_curves(x, T.cb, T.sym.C)[0].astype(jnp.float32)
            d2 = jnp.sum((gb0[:, :, None, :] - S_all.astype(jnp.float32)[None, None]) ** 2, -1)
            Sw = S_all[lax.top_k(-d2, cs_window)[1]]                      # (C, nq, K, 3)
            cons_of = lambda x, p, kd: pw.values_x(x, T.cb, T.sym, Sw, T.lim, ph.cap_at(T, p), kd)
        else:
            Sw = None
            cons_of = lambda x, p, kd: ph.cons(T, x, p, kd)
        if extra is not None:                         # extra's rows after the (windowed) base constraints
            base_cons = cons_of
            cons_of = lambda x, p, kd: jnp.concatenate([base_cons(x, p, kd), extra.g(x).astype(x.dtype)])
        if q is not None:                             # IRLS weights on the flux rows
            nf = q.shape[0]
            r_of = lambda x, p: r_only(x, p).at[:nf].multiply(q)

        # jax.named_scope labels below are metadata only (profiler attribution), no computation
        def derivs(x, r=None, g=None):
            if mixed:
                with jax.named_scope("flux_jac"):
                    J = J32(x, p)
                if r is None:
                    with jax.named_scope("flux_res"):
                        r = r_of(x, p)
                    with jax.named_scope("cons"):
                        g = cons_of(x, p, kdt_g)
            else:
                with jax.named_scope("flux_jac"):
                    r, J = rJ(x, p)
                    if q is not None:
                        r = r.at[:nf].multiply(q)
                with jax.named_scope("cons"):
                    g = cons_of(x, p, None)
            if q is not None:
                J = J.at[:nf].multiply(q[:, None])
            c = g + lam / rho
            h = jnp.maximum(0.0, c)
            with jax.named_scope("cons_jac"):
                A = A_of(x, p, h, c, Sw)
            with jax.named_scope("normal_eq"):
                gF = 2.0 * (J.T @ r)
                grad = gF + rho * (A.T @ h)                           # rows with h = 0 drop out
                Aa = A * ((c > -near) if near else (h > 0))[:, None]  # near: + near-active rows (majorizer)
                H = 2.0 * (J.T @ J) + rho * (Aa.T @ Aa)
            if pen_curv and T.cfg.arc_pen_thresh:
                with jax.named_scope("arc_curv"):
                    H = H + rs.arc_penalty_curvature(x, T.cb, T.sym.C, T.cfg, c_f)
            stat = jnp.max(jnp.abs(grad)) / (1.0 + jnp.max(jnp.abs(gF)))
            return r, g, grad, H, stat, J, A

        r, g, grad, H, stat, J, A = derivs(x)
        lk0 = links(x) if link_test else None             # topology of the outer iteration's start
        mu0 = 1e-3 * jnp.max(jnp.diag(H))
        mu = jnp.where(mu > 0, mu, mu0)
        Dw = jnp.asarray(_mode_weights(T, n), x.dtype) if damp == "fourier" else None
        # carry: x, r, g, grad, H, stat, phi, mu, nu, accepted, trials, n_jac, n_res, phi_ref, stalled
        f64, i32 = x.dtype, jnp.int32
        phi0 = phi_of(r, g)
        c0 = (x, r, g, grad, H, stat, phi0, mu.astype(f64), jnp.asarray(nu, f64),
              jnp.asarray(0, i32), jnp.asarray(0, i32), jnp.asarray(1, i32), jnp.asarray(0, i32),
              phi0, jnp.asarray(False))

        g0 = jnp.max(jnp.abs(grad))

        def cond(c):
            if prog > 0.0:
                go = ~c[14]
            else:
                go = (jnp.max(jnp.abs(c[3])) / g0 if rel else c[5]) > omega
            return go & (c[9] < inner) & (c[10] < 4 * inner) & (c[7] < 1e12)

        def body(c):
            x, r, g, grad, H, stat, phi, mu, nu, acc, tr, nj, nr, phi_ref, stalled = c
            if damp == "levenberg":
                D = I
            else:                                         # "fourier": Levenberg in mode-scaled variables
                D = jnp.diag(Dw)
            with jax.named_scope("lm_solve"):
                L = jnp.linalg.cholesky(H + mu * D)
                ok = jnp.all(jnp.isfinite(L))
                d = -jax.scipy.linalg.cho_solve((L, True), grad)
                pred = -(grad @ d + 0.5 * d @ (H @ d))
            x1 = x + d
            if link_test:                                 # coils may not pass through each other
                with jax.named_scope("link_test"):
                    ok = ok & jnp.all(links(x1) == lk0)
            with jax.named_scope("flux_res"):
                r1 = r_of(x1, p)
            with jax.named_scope("cons"):
                g1 = cons_of(x1, p, kdt_g)
            phi1 = phi_of(r1, g1)
            ratio = jnp.where(pred > 0, (phi - phi1) / pred, -1.0)
            accept = ok & (ratio > 1e-4)

            def on_accept(_):
                rr, gg, gr, HH, st = derivs(x1, r1, g1)[:5]
                mu1 = mu * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * ratio - 1.0) ** 3)
                ph1 = phi_of(rr, gg); a1 = acc + 1
                chk = (a1 % 10) == 0                                  # progress check every 10 steps
                stl = chk & ((phi_ref - ph1) < prog * ph1)
                pref = jnp.where(chk, ph1, phi_ref)
                return (x1, rr, gg, gr, HH, st, ph1, mu1, jnp.asarray(2.0, f64), a1, tr + 1, nj + 1, nr + 1,
                        pref, stl)

            def on_reject(_):
                return (x, r, g, grad, H, stat, phi, mu * jnp.where(ok, nu, 10.0), jnp.where(ok, 2.0 * nu, nu),
                        acc, tr + 1, nj, nr + 1, phi_ref, stalled)
            return lax.cond(accept, on_accept, on_reject, None)

        x, r, g, grad, H, stat, phi, mu, nu, acc, tr, nj, nr = lax.while_loop(cond, body, c0)[:13]
        mu = jnp.where(mu < 1e12, mu, -1.0)                       # reset on damping blow-up
        if Sw is not None:                                        # whole-grid constraints at the outer's end
            g = g_only(x, p)
        return x, g, stat, mu, nu, acc, tr, nj, nr, r @ r

    if jit:
        outer = partial(jax.jit, static_argnames=("inner", "rel", "prog"))(outer)
    return outer


class _CoarseView:
    """The same problem on coarser flux / distance grids (quadrature, DOFs and constraint rows
    unchanged, so the AL state carries over row for row). Tables are built from the fine
    problem's seed, so B_ref, I0 and L_ref - hence the scaling of every residual - are identical."""

    def __init__(self, prob, grid, grid_dist, nq=None, pen_w=None):
        from dataclasses import replace
        cfg = replace(prob.cfg, grid_dist=tuple(grid_dist), nq_opt=int(nq or prob.cfg.nq_opt))
        if pen_w is not None:                       # penalty weight for the first subproblem only
            cfg = replace(cfg, arc_pen_weight=float(pen_w))
        self.T = ph.build_tables(prob.b, cfg, prob.seed, grid=tuple(grid))
        self.c_f, self.eps, self.cnt = prob.c_f, prob.eps, prob.cnt
        self.m = int(ph.cons(self.T, jnp.asarray(prob.seed.x), (0.0, prob.eps)).size)
        self.F, self.g = prob.F, prob.g


def _coarse(prob, budget):
    key = (f"_coarse_{budget['coarse_grid']}_{budget.get('coarse_dist')}_{budget.get('coarse_nq')}"
           f"_{budget.get('pen_w_first')}")
    if not hasattr(prob, key):
        g = tuple(int(v) for v in str(budget["coarse_grid"]).split("x"))
        gd = tuple(int(v) for v in str(budget.get("coarse_dist", "")).split("x")) if budget.get("coarse_dist") \
            else prob.cfg.grid_dist
        setattr(prob, key, _CoarseView(prob, g, gd, budget.get("coarse_nq"), budget.get("pen_w_first")))
    v = getattr(prob, key); v.cnt = prob.cnt
    return v


# --- fully device-resident solve with on-device stopping (P5) --------------------------------

def _lancelot(g, lam, rho, omega, eta, om_min=1e-8, eta_min=1e-9, rho_fac=10.0, rho_max=RHO_MAX):
    """Nocedal & Wright Alg. 17.4 (inequality PHR), on device; rho x rho_fac on failure, capped at rho_max."""
    infeas = jnp.maximum(0.0, jnp.max(g))
    ok = infeas <= eta
    lam1 = jnp.where(ok, jnp.maximum(0.0, lam + rho * g), lam)
    rho1 = jnp.where(ok, rho, jnp.minimum(rho_fac * rho, rho_max))
    eta1 = jnp.where(ok, jnp.maximum(eta / rho ** 0.9, eta_min), jnp.maximum(0.1 / rho1 ** 0.1, eta_min))
    om1 = jnp.where(ok, jnp.maximum(omega / rho, om_min), jnp.maximum(1.0 / rho1, om_min))
    return lam1, rho1, om1, eta1, infeas


def _al_opts(budget):
    """Outer-loop and step options (defaults: device_lm.DEFAULTS).
      near_rows    near-active row window delta of _build (0: off)
      damp         "levenberg" (H + mu I) | "fourier" (H + mu diag((1 + m)^2), m the coefficient's Fourier mode)
      rho_rule     "lancelot" (N&W 17.4) | "conv": multipliers every outer; rho x rho_fac only when the
                   outer's inner loop converged (stat <= omega) and V = max|min(-g, lam/rho)| fell by less
                   than the factor rho_tau since the last outer (V > rho_tau V_prev; ALGENCAN /
                   Conn-Gould-Toint); omega x 0.1 after a converged outer | "guard": "conv", and rho x rho_fac
                   also after rho_stall consecutive unconverged outers whose V did not decrease (V >= V_prev:
                   the inner loop stalls at a rho too small to reduce the violation) | "algencan": rho x
                   rho_fac whenever V > rho_tau V_prev, converged or not
      rho_fac, rho_max, rho_tau   growth factor (10), cap (RHO_MAX), V-decrease factor of "conv" (0.25)
      rho_stall    stalled outers before "guard" raises rho (1)
      irls         eps > 0: each outer after the first reweights the flux residuals (irls_weights): one IRLS step
                   per outer towards the L1 field ratio sum w |B.n| / sum w |B| (0: off)
      dist_margin  distance margin beta_d of the coil-coil and coil-surface rows ((1 + beta_d) d_lim - softmin
                   <= 0) used by the solve; None: the problem's own. The soft-min runs over the distance grid's
                   points, the certificate measures the exact distance to the triangulated surface
      cs_window    K > 0: coil-surface soft-min over the K nearest distance-grid points inside an outer (_build)
      ftol         general stopping rule |F_k - F_{k-1}| <= ftol F_k (build_solver's ftol argument wins if given)"""
    d = dict(DEFAULTS, **budget)
    o = dict(near=float(d["near_rows"]), damp=str(d["damp"]), rule=str(d["rho_rule"]),
             rho_fac=float(d.get("rho_fac", 10.0)), rho_max=float(d.get("rho_max", RHO_MAX)),
             tau=float(d.get("rho_tau", 0.25)), stall=int(d.get("rho_stall", 1)), irls=float(d["irls"]),
             cs_window=int(d["cs_window"]),
             margin=None if d["dist_margin"] is None else float(d["dist_margin"]))
    assert o["rule"] in ("lancelot", "conv", "guard", "algencan") and _damp_ok(o["damp"]), o
    assert d.get("link_test", False) in (True, False, 0, 1), d.get("link_test")
    return o


def _info_of(c):
    return dict(outer=c[8], stopped=c[9], trials=c[10], jac=c[11], rho=c[2], first_steps=c[12])


def _al_key(o):
    return tuple(sorted((k, tuple(sorted(v.items())) if isinstance(v, dict) else v) for k, v in o.items()))


def _al_update(o, g, lam, rho, omega, eta, stat, V, om_min=1e-8, eta_min=1e-9):
    """Multiplier / penalty update after an outer: (lam, rho, omega, eta, V)."""
    if o["rule"] == "lancelot":
        lam1, rho1, om1, eta1, _ = _lancelot(g, lam, rho, omega, eta, om_min, eta_min, o["rho_fac"], o["rho_max"])
        return lam1, rho1, om1, eta1, V
    V1 = jnp.max(jnp.abs(jnp.minimum(-g, lam / rho)))
    conv = stat <= omega
    lam1 = jnp.maximum(0.0, lam + rho * g)
    up = V1 > o["tau"] * V
    if o["rule"] == "conv":
        up = conv & up
    elif o["rule"] == "guard":                      # eta (unused by these rules) counts the stalled outers
        n = jnp.where(~conv & (V1 >= V), jnp.floor(eta) + 1.0, 0.0)
        up = (conv & up) | (n >= o["stall"])
        eta = jnp.where(up, 0.0, n)
    rho1 = jnp.where(up, jnp.minimum(o["rho_fac"] * rho, o["rho_max"]), rho)
    om1 = jnp.where(conv, jnp.maximum(0.1 * omega, om_min), omega)
    return lam1, rho1, om1, eta, V1


# Defaults of the solve: near-active rows, Fourier-mode damping, the guard penalty
# rule (rho raised after 3 stalled outers), IRLS, a 1 % distance margin, ftol 1e-2 and the coil-surface window of 256
# points; with MAIN's inner 50 and first_prog 0.03.
DEFAULTS = dict(near_rows=1e-3, damp="fourier", rho_rule="guard", rho_stall=3, irls=0.1, dist_margin=0.01, ftol=1e-2,
                cs_window=256)

# the budget solve_all uses (on top of DEFAULTS)
MAIN = dict(precision="mixed", inner=50, inner_first=400, first_prog=0.03, pen_curv=1, pen_w_first=0.01,
            coarse_grid="16x32", coarse_dist="48x32", cc_rows=128)


def _with_margin(T, margin):
    """Tables T with the distance margin beta_d = margin (None: unchanged)."""
    if margin is None or margin == T.lim.beta_d:
        return T
    from dataclasses import replace
    return replace(T, lim=replace(T.lim, beta_d=float(margin)))


def _ftol(budget, ftol):
    return float(ftol if ftol is not None else dict(DEFAULTS, **budget)["ftol"])


def make_monitor(prob, kdt=jnp.float32):
    """On-device target monitor on the certification's grid and quadrature: (mean, max)
    |B.n|/|B| in percent, the same definition as the certificate (a coarser monitor grid
    over-estimated the QH max by ~2 %, the size of the target margin). kdt=float32: mixed-precision
    field (FP64 separations, FP32 kernel, FP64 accumulation; ~1e-7 relative), kdt=None: FP64."""
    from xcoil.field import mixed as mx
    from xcoil.verify import certify as cf
    V = prob.V
    W = jnp.sum(V.w)

    field = lambda x, P: mx.field_on(x, V.cbq, V.sym, P, V.I0, kdt)      # kdt=None: FP64

    def base(x):
        """(mean, max) on the certification grid."""
        B = field(x, V.X)
        ratio = jnp.abs(jnp.sum(B * V.n, 1)) / jnp.linalg.norm(B, axis=1)
        return 100.0 * jnp.sum(V.w * ratio) / W, 100.0 * jnp.max(ratio)

    def fine_max(x):
        """max on the fine grid (certify.MAX_GRID_FACTOR x finer), point-chunked."""
        B2 = cf.field_chunked(x, V, V.X2, field=lambda P: field(x, P))
        return 100.0 * jnp.max(jnp.abs(jnp.sum(B2 * V.n2, 1)) / jnp.linalg.norm(B2, axis=1))

    def mon(x):
        # mean on the certification grid, max over it and the fine grid (as the certificate)
        m, M = base(x)
        return m, jnp.maximum(M, fine_max(x))
    # lazy target test (build_solver): the fine grid is evaluated only when the mean and the
    # certification-grid max already meet the target; the decisions are those of mon, since the
    # certified max is never below the certification-grid max
    mon.base, mon.fine_max = base, fine_max
    mon.shared = dict(key=("cert", None if kdt is None else str(jnp.dtype(kdt))), data=V,
                      check=partial(_cert_check, kdt=kdt))
    return mon


def _cert_check(x, V, tgt, kdt=jnp.float32, mean_only=False):
    """make_monitor's lazy target test as a function of the problem data V (shared programs): mean and max on the
    certification grid meet tgt, then (unless mean_only) the max on the fine grid."""
    from xcoil.field import mixed as mx
    from xcoil.verify import certify as cf
    field = lambda P: mx.field_on(x, V.cbq, V.sym, P, V.I0, kdt)
    B = field(V.X)
    ratio = jnp.abs(jnp.sum(B * V.n, 1)) / jnp.linalg.norm(B, axis=1)
    m, M = 100.0 * jnp.sum(V.w * ratio) / jnp.sum(V.w), 100.0 * jnp.max(ratio)
    ok = (m <= tgt[0]) & (M <= tgt[1])
    if mean_only:
        return ok

    def fine(_):
        B2 = cf.field_chunked(x, V, V.X2, field=field)
        return 100.0 * jnp.max(jnp.abs(jnp.sum(B2 * V.n2, 1)) / jnp.linalg.norm(B2, axis=1)) <= tgt[1]
    return lax.cond(ok, fine, lambda _: jnp.asarray(False), None)


def true_violation_offsets(prob, T=None):
    """Per-row margin of the optimization constraints (beta on the curvature and MSC rows, beta_d
    on the distance rows, 0 on length): g - offset is the violation of the true limit, the
    quantity certification bounds by viol_tol. T: the tables the solve uses (default prob.T)."""
    T = prob.T if T is None else T
    lim = T.lim; C, nq = T.sym.C, T.cb.B0.shape[0]
    bl = pw.blocks_for(C, nq, lim)
    off = np.zeros(prob.m)
    off[bl["curv"]] = lim.beta; off[bl["msc"]] = lim.beta
    off[bl["cc"]] = lim.beta_d; off[bl["cs"]] = lim.beta_d
    return jnp.asarray(off)


def build_solver(prob, budget, target=None, monitor=None, max_outer=60, viol_tol=1e-3, ftol=None, shared=True,
                 extra=None):
    """Returns solve(x0) -> (x, info): the whole AL-LM solve as two jitted device programs
    (first outer - on the coarse grids if budget has coarse_grid - and a lax.while_loop over the
    remaining outers), no host synchronisation in between.
    budget["link_test"] rejects trial steps that change a coil-pair linking number.
    Stopping (on device), with v = max(g - margins) the violation of the true limits (the quantity
    certification bounds by viol_tol): target=(mean_pct, max_pct) with a monitor -> stop at the
    first outer whose monitor values meet the target with v <= viol_tol (time to target);
    otherwise (general rule) stop when v <= viol_tol and |F_k - F_{k-1}| <= ftol F_k.
    Always stops at max_outer.
    shared: the programs take the problem's tables as an input and are cached by structure
    (_shared_solver), so devices with the same coils, order, quadrature, grids and settings - any
    boundary, any limits - reuse one compiled program; in target mode too when the monitor provides
    its test as a function of the problem data (monitor.shared: make_monitor); shared=False compiles programs for
    this problem alone (the tables as constants).
    extra (Extra): constraint rows appended (_build); programs for this problem alone."""
    ftol = _ftol(budget, ftol)
    if shared and extra is None and (target is None or hasattr(monitor, "shared")):
        return _shared_solver(prob, budget, max_outer, viol_tol, ftol, target, monitor)
    jd = budget.get("precision", "fp64"); ccK = int(budget.get("cc_rows", 0))
    pc = bool(budget.get("pen_curv", False))
    inner, inner_first = int(budget["inner"]), int(budget.get("inner_first", budget["inner"]))
    lt = bool(budget.get("link_test", False))
    o = _al_opts(budget)
    Tf = _with_margin(prob.T, o["margin"])
    fine = _build(Tf, prob.c_f, jd, ccK, pc, lt, near=o["near"], damp=o["damp"], extra=extra,
                  cs_window=o["cs_window"])
    cview = _coarse(prob, budget) if budget.get("coarse_grid") else None
    first = _build(_with_margin(cview.T, o["margin"]), prob.c_f, jd, ccK, pc, lt, near=o["near"],
                   damp=o["damp"], extra=extra, cs_window=o["cs_window"]) if cview else fine
    me = 0 if extra is None else extra.m(prob.n)
    first_m = (cview.m if cview else prob.m) + me
    rho0 = float(budget.get("rho0", 10.0))
    omega_first = budget.get("omega_first")          # None: 1/rho0; 0: run all inner_first steps
    first_rel = budget.get("first_rel")              # e.g. 1e-2: first subproblem to |grad Phi| <= 1e-2 |grad Phi_0|
    first_prog = float(budget.get("first_prog", 0.0))  # e.g. 1e-3: first subproblem until 10 steps gain < 1e-3 Phi
    kdt = jnp.float32 if jd == "mixed" else None
    p = (jnp.asarray(1.0), jnp.asarray(prob.eps))
    tgt = None if target is None else (jnp.asarray(float(target[0])),          # max None: mean-only target
                                       jnp.asarray(jnp.inf if target[1] is None else float(target[1])))
    hp = jax.default_matmul_precision("highest")
    off = true_violation_offsets(prob, Tf)
    if me:
        off = jnp.concatenate([off, jnp.zeros(me)])

    def stop_test(x, g, F, Fprev):
        v = jnp.max(g - off)
        if tgt is not None:                          # monitor only once the true limits hold
            @jax.named_scope("target_test")
            def check(_):
                if not hasattr(monitor, "fine_max"):
                    m, M = monitor(x)
                    return (m <= tgt[0]) & (M <= tgt[1])
                m, M = monitor.base(x)
                ok = (m <= tgt[0]) & (M <= tgt[1])
                if target[1] is None:                # mean-only target: the max is not needed
                    return ok
                return lax.cond(ok, lambda _: monitor.fine_max(x) <= tgt[1], lambda _: jnp.asarray(False), None)
            return lax.cond(v <= viol_tol, check, lambda _: jnp.asarray(False), None)
        return (v <= viol_tol) & (jnp.abs(F - Fprev) <= ftol * F)

    @jax.jit
    def stage1(x0):
        mc = first_m
        rho = jnp.asarray(rho0, x0.dtype)
        omega, eta = 1.0 / rho, 0.1 / rho ** 0.1
        om1 = omega if omega_first is None else jnp.asarray(float(omega_first), x0.dtype)
        if first_rel is not None:
            om1 = jnp.asarray(float(first_rel), x0.dtype)
        x, g, stat, mu, nu, acc, tr, nj, nr, F = first(x0, jnp.zeros(mc, x0.dtype), rho, jnp.asarray(-1.0),
                                                      jnp.asarray(2.0), om1, p, inner=inner_first,
                                                      rel=first_rel is not None, prog=first_prog)
        # multiplier / penalty update with the full-resolution constraints: exact, since lam0 = 0
        g = ph.cons(Tf, x, p, kdt)
        if me:
            g = jnp.concatenate([g, extra.g(x).astype(g.dtype)])
        lam, rho, omega, eta, V = _al_update(o, g, jnp.zeros(prob.m + me, x0.dtype), rho, omega, eta, stat,
                                             jnp.asarray(jnp.inf, x0.dtype))
        done = stop_test(x, g, F, jnp.inf)
        return (x, lam, rho, mu, nu, omega, eta, F, jnp.asarray(1), done, tr, nj, acc, V)

    def body(c):
        x, lam, rho, mu, nu, omega, eta, Fp, k, done, trs, njs, acc1, V = c
        q = irls_weights(Tf, x, p[0], o["irls"]) if o["irls"] else None
        x, g, stat, mu, nu, acc, tr, nj, nr, F = fine(x, lam, rho, mu, nu, omega, p, inner=inner, q=q)
        lam, rho, omega, eta, V = _al_update(o, g, lam, rho, omega, eta, stat, V)
        return (x, lam, rho, mu, nu, omega, eta, F, k + 1, stop_test(x, g, F, Fp), trs + tr, njs + nj, acc1, V)

    @jax.jit
    def stage2(c):
        return jax.lax.while_loop(lambda c: (~c[9]) & (c[8] < max_outer), body, c)

    def solve(x0):
        with hp:
            c = stage2(stage1(jnp.asarray(x0, jnp.float64)))
        return c[0], _info_of(c)
    solve.stage1, solve.stage2, solve.precision = stage1, stage2, hp      # for profiling the two stages
    solve.body, solve.max_outer = body, max_outer
    return solve


# --- programs shared across devices of the same structure ----------------------------------------

_SHARED = {}


def _shared_outer(precision, cc_K, pen_curv, link_test, near=0.0, damp="levenberg", cs_window=0):
    """outer(x, lam, rho, mu, nu, omega, p, T, inner, rel, prog, q): _build's outer iteration with the
    tables T as an argument; one jitted function per option set (XLA compiles it once per array
    shapes and table structure)."""
    key = ("outer", precision, int(cc_K), bool(pen_curv), "chol", bool(link_test))
    if near or damp != "levenberg" or cs_window:
        key = key + (float(near), damp) + ((int(cs_window),) if cs_window else ())
    if key not in _SHARED:
        def outer(x, lam, rho, mu, nu, omega, p, T, inner, rel=False, prog=0.0, q=None):
            f = _build(T, 1.0 / T.cfg.f_ref, precision, cc_K, pen_curv, link_test, jit=False, near=near,
                       damp=damp, cs_window=cs_window)
            return f(x, lam, rho, mu, nu, omega, p, inner=inner, rel=rel, prog=prog, q=q)
        _SHARED[key] = jax.jit(outer, static_argnames=("inner", "rel", "prog"))
    return _SHARED[key]


def _shared_stages(budget, max_outer, viol_tol, ftol, mon=None):
    """(stage1(x0, D), stage2(c, D)) of build_solver, with the problem data D = (T fine, T first subproblem,
    p = (t, eps), true-violation offsets, monitor data, target) as an argument. mon = (key, check, mean_only): target
    mode, stop at the first outer where check(x, monitor data, target) holds with the limits met (monitor.shared);
    None: the general stopping rule."""
    jd = budget.get("precision", "fp64"); ccK = int(budget.get("cc_rows", 0))
    pc = bool(budget.get("pen_curv", False))
    inner, inner_first = int(budget["inner"]), int(budget.get("inner_first", budget["inner"]))
    lt = bool(budget.get("link_test", False))
    rho0 = float(budget.get("rho0", 10.0))
    omega_first, first_rel = budget.get("omega_first"), budget.get("first_rel")
    first_prog = float(budget.get("first_prog", 0.0))
    o = _al_opts(budget)
    key = ("stages", jd, ccK, pc, inner, inner_first, "chol", lt, rho0, omega_first, first_rel, first_prog,
           int(max_outer), float(viol_tol), float(ftol), _al_key(o), None if mon is None else (mon[0], mon[2]), None)
    if key in _SHARED:
        return _SHARED[key]
    outer = _shared_outer(jd, ccK, pc, lt, o["near"], o["damp"], o["cs_window"])
    kdt = jnp.float32 if jd == "mixed" else None

    def stop_test(g, off, F, Fprev, x, MD, tgt):
        if mon is None:
            return (jnp.max(g - off) <= viol_tol) & (jnp.abs(F - Fprev) <= ftol * F)
        check = mon[1]                                # monitor only once the true limits hold
        return lax.cond(jnp.max(g - off) <= viol_tol, lambda _: check(x, MD, tgt), lambda _: jnp.asarray(False), None)

    @jax.jit
    def stage1(x0, D):
        Tf, Tc, p, off, MD, tgt = D
        mc = jax.eval_shape(lambda: ph.cons(Tc, x0, p)).shape[0]
        m = off.shape[0]
        rho = jnp.asarray(rho0, x0.dtype)
        omega, eta = 1.0 / rho, 0.1 / rho ** 0.1
        om1 = omega if omega_first is None else jnp.asarray(float(omega_first), x0.dtype)
        if first_rel is not None:
            om1 = jnp.asarray(float(first_rel), x0.dtype)
        x, g, stat, mu, nu, acc, tr, nj, nr, F = outer(x0, jnp.zeros(mc, x0.dtype), rho, jnp.asarray(-1.0),
                                                      jnp.asarray(2.0), om1, p, Tc, inner=inner_first,
                                                      rel=first_rel is not None, prog=first_prog)
        g = ph.cons(Tf, x, p, kdt)                    # multipliers from the full-resolution constraints
        lam, rho, omega, eta, V = _al_update(o, g, jnp.zeros(m, x0.dtype), rho, omega, eta, stat,
                                             jnp.asarray(jnp.inf, x0.dtype))
        done = stop_test(g, off, F, jnp.inf, x, MD, tgt)
        return (x, lam, rho, mu, nu, omega, eta, F, jnp.asarray(1), done, tr, nj, acc, V)

    @jax.jit
    def stage2(c, D):
        Tf, _, p, off, MD, tgt = D

        def cond(c):
            return (~c[9]) & (c[8] < max_outer)

        def body(c):
            x, lam, rho, mu, nu, omega, eta, Fp, k, done, trs, njs, acc1, V = c
            q = irls_weights(Tf, x, p[0], o["irls"]) if o["irls"] else None
            x, g, stat, mu, nu, acc, tr, nj, nr, F = outer(x, lam, rho, mu, nu, omega, p, Tf, inner=inner, q=q)
            lam, rho, omega, eta, V = _al_update(o, g, lam, rho, omega, eta, stat, V)
            return (x, lam, rho, mu, nu, omega, eta, F, k + 1, stop_test(g, off, F, Fp, x, MD, tgt), trs + tr, njs + nj,
                    acc1, V)
        return jax.lax.while_loop(cond, body, c)

    _SHARED[key] = (stage1, stage2)
    return _SHARED[key]


def _shared_solver(prob, budget, max_outer=60, viol_tol=1e-3, ftol=1e-2, target=None, monitor=None):
    """build_solver's solve(x0) for this problem on the shared programs (see build_solver)."""
    mon, MD, tgt = None, None, None
    if target is not None:
        sh = monitor.shared
        mean_only = target[1] is None
        check = partial(_cert_check, kdt=sh["check"].keywords["kdt"], mean_only=mean_only)
        mon, MD = (sh["key"], check, mean_only), sh["data"]
        tgt = (jnp.asarray(float(target[0])), jnp.asarray(jnp.inf if target[1] is None else float(target[1])))
    s1, s2 = _shared_stages(budget, max_outer, viol_tol, ftol, mon)
    cview = _coarse(prob, budget) if budget.get("coarse_grid") else None
    m = _al_opts(budget)["margin"]
    Tf = _with_margin(prob.T, m)
    off = true_violation_offsets(prob, Tf)
    D = (Tf, _with_margin(cview.T, m) if cview else Tf, (jnp.asarray(1.0), jnp.asarray(prob.eps)), off, MD, tgt)
    hp = jax.default_matmul_precision("highest")

    def stage1(x0):
        return s1(jnp.asarray(x0, jnp.float64), D)

    def stage2(c):
        return s2(c, D)

    def solve(x0):
        with hp:
            c = s2(s1(jnp.asarray(x0, jnp.float64), D), D)
        return c[0], _info_of(c)
    solve.stage1, solve.stage2, solve.precision = stage1, stage2, hp
    solve.data = D
    return solve


def build_batched_solver(prob, budget, target=None, monitor=None, max_outer=60, viol_tol=1e-3, ftol=None, extra=None):
    """Multi-start: solve_batch(X0) -> (X, info) for B starts X0 (B, n) of the same problem in one
    solve - build_solver's two device programs vmapped over the starts (one compiled program per B).
    Every member follows its own AL-LM iteration and stopping rule; the batch ends when every member
    has stopped (or reached max_outer, the per-member cap), so its time is that of the slowest member,
    and finished members are carried along unchanged. info: build_solver's entries as (B,) arrays."""
    one = build_solver(prob, budget, target, monitor, max_outer, viol_tol, ftol, shared=extra is None, extra=extra)
    s1, s2 = jax.jit(jax.vmap(one.stage1)), jax.jit(jax.vmap(one.stage2))

    def solve_batch(X0):
        with one.precision:
            c = s2(s1(jnp.asarray(X0, jnp.float64)))
        return c[0], dict(outer=c[8], stopped=c[9], trials=c[10], jac=c[11], rho=c[2], first_steps=c[12])
    solve_batch.stage1, solve_batch.stage2, solve_batch.single = s1, s2, one
    solve_batch.max_outer = max_outer
    return solve_batch


def distance_shortfall(prob, cert):
    """How far the certified coil-surface and coil-coil distances fall below their limits (relative; <= 0: none)."""
    lim = prob.lim
    return max(lim.dcs / float(cert["d_cs"]) - 1.0, lim.dcc / float(cert["d_cc"]) - 1.0)


def repair_distances(prob, x, cert, budget, factor=1.5, extra=2e-3, max_outer=60):
    """Certificate feedback for the distance limits: a design whose certificate fails only by a distance shortfall s
    (the solve's soft-min runs over the distance grid's points, the certificate measures the exact distance to the
    triangulated surface) is re-solved from x with the distance margin factor s + extra, without the coarse first
    outer. Returns (x, cert, margin): the re-solved design if it is certified or has a smaller mean, else the input;
    margin None when nothing was done."""
    s = distance_shortfall(prob, cert)
    if cert["certified"] or cert.get("linked") or s <= 0:
        return x, cert, None
    m = round(factor * s + extra, 4)
    b = {k: v for k, v in budget.items() if k not in ("coarse_grid", "coarse_dist")}
    b.update(inner_first=int(b.get("inner", 50)), first_prog=0.0, rho0=1e3, dist_margin=m)
    x2, _ = build_solver(prob, b, max_outer=max_outer)(np.asarray(x))
    x2 = np.asarray(x2)
    c2 = prob.certify(x2)
    if c2["certified"] or c2["mean_bn_pct"] < cert["mean_bn_pct"]:
        return x2, c2, m
    return x, cert, m
