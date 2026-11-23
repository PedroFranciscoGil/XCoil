"""The coil problem (CoilProblem): boundary, settings, tables, objective, constraints and certificate.

P(t, eps):  min F(x, t) = c_f |r(x, t)|^2      s.t.  g(x, t) <= 0
r = (flux residual on Sigma(t), regularizers), g = the pointwise constraint set of homotopy/problem_h (length cap eps,
curvature, MSC, coil-coil, coil-surface). The fixed-boundary solve uses t = 1 (the target boundary).
Callbacks are jitted JAX (FP64) and counted. user_problem.make_problem builds one.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import nnls

import xcoil  # noqa: F401
from xcoil.homotopy import boundary as bd, problem_h as ph, seeds as seeds_mod
from xcoil.curves import currents as cur
from xcoil.homotopy.config import for_device
from xcoil.verify import certify as cf


_CERT = jax.jit(cf.certify, static_argnames=("viol_tol",))


def _arcvar(y, cb, M, C):
    from xcoil.field import residuals as rs
    return jax.vmap(lambda c: rs._coil_arcvar(c, cb, M))(cur.geo(y, cb, C).reshape(C, -1))


_ARCVAR = jax.jit(_arcvar, static_argnames=("C",))


@dataclass
class Counters:
    f: int = 0; grad: int = 0; jac_r: int = 0; cons: int = 0; jac_c: int = 0; hess: int = 0

    def as_dict(self):
        return dict(self.__dict__)


class CoilProblem:
    def __init__(self, device="lpqa", eps=18.2, grid=None, **overrides):
        cfg = for_device(device, **overrides)
        self.cfg, self.eps = cfg, float(eps)
        self.b = bd.load(cfg.surface)
        lim = ph.limits_from(cfg)
        order = cfg.order if cfg.basis == "fourier" else cfg.n_ctrl
        self.seed = seeds_mod.build(self.b, cfg.n_coils, lim, cfg.basis, order, cfg.nq_opt, cfg.grid_dist)
        self.T = ph.build_tables(self.b, cfg, self.seed, grid=grid or cfg.grid_fine)
        self.V = cf.build(self.b, cfg, self.T)
        T = self.T
        self.c_f = 1.0 / T.cfg.f_ref
        self.n = int(self.seed.x.size) + cfg.n_coils - 1     # coefficients + free currents u (curves/currents)
        self.m = int(ph.cons(T, jnp.asarray(self.seed.x), (0.0, self.eps)).size)
        self._r = jax.jit(lambda x, t: ph.resid(T, x, (t, self.eps)))
        self._rJ = jax.jit(lambda x, t: ph.resid_jac(T, x, (t, self.eps)))
        self._g = jax.jit(lambda x, t: ph.cons(T, x, (t, self.eps)))
        self._gA = jax.jit(lambda x, t: ph.cons_jac(T, x, (t, self.eps)))
        self._curv = jax.jit(lambda x, t, z, r: ph.exact_blockdiag(self.T, x, (t, self.eps), z, r, clip=False))
        self._curv_clip = jax.jit(lambda x, t, z, r: ph.exact_blockdiag(self.T, x, (t, self.eps), z, r, clip=True))
        self._curv_cons = jax.jit(lambda x, t, z: ph.lagrangian_blockdiag(self.T, x, (t, self.eps), z))
        self.cnt = Counters()
        self.lim = ph.limits_from(cfg)

    # --- callbacks (numpy in / out, counted) -----------------------------------------------
    def F(self, x, t):
        self.cnt.f += 1
        r = np.asarray(self._r(jnp.asarray(x), t)); return self.c_f * float(r @ r)

    def F_grad(self, x, t):
        self.cnt.grad += 1
        r, J = self._rJ(jnp.asarray(x), t); r, J = np.asarray(r), np.asarray(J)
        return self.c_f * float(r @ r), 2 * self.c_f * J.T @ r

    def resid_jac(self, x, t):
        self.cnt.jac_r += 1
        r, J = self._rJ(jnp.asarray(x), t)
        return np.sqrt(self.c_f) * np.asarray(r), np.sqrt(self.c_f) * np.asarray(J)

    def g(self, x, t):
        self.cnt.cons += 1
        return np.asarray(self._g(jnp.asarray(x), t))

    def g_jac(self, x, t):
        self.cnt.jac_c += 1
        g, A = self._gA(jnp.asarray(x), t); return np.asarray(g), np.asarray(A)

    def curvature(self, x, t, z):
        """Exact per-coil second-order terms (flux S + z-weighted constraint Hessians), unclipped."""
        self.cnt.hess += 1
        r = self._r(jnp.asarray(x), t)
        return np.asarray(self._curv(jnp.asarray(x), t, jnp.asarray(z), r)) / 1.0

    def lagr_hess(self, x, t, z, r, kind):
        """Second-order terms missing from the Gauss-Newton AL model, z = AL multiplier estimate:
        'cons'  per-coil PSD Hessian of z^T g (length, curvature, MSC rows);
        'exact' per-coil exact curvature (flux S term + all z-weighted rows incl. distances),
                each coil block projected to PSD."""
        self.cnt.hess += 1
        if kind == "cons":
            return np.asarray(self._curv_cons(jnp.asarray(x), t, jnp.asarray(z)))
        return np.asarray(self._curv_clip(jnp.asarray(x), t, jnp.asarray(z), jnp.asarray(r) / np.sqrt(self.c_f)))

    # --- solver-agnostic quality measures -------------------------------------------------
    def kkt(self, x, t, act_tol=1e-3):
        """Stationarity with NNLS multipliers on near-active rows (solver-agnostic), max(0, max g)."""
        _, gF = self.F_grad(x, t); g, A = self.g_jac(x, t)
        act = np.where(g > -act_tol)[0]
        if act.size:
            z, _ = nnls(A[act].T, -gF, maxiter=20 * act.size)
            rd = gF + A[act].T @ z
        else:
            rd = gF
        return dict(stat=float(np.max(np.abs(rd)) / (1 + np.max(np.abs(gF)))), infeas=max(0.0, float(g.max())),
                    n_active=int(act.size))

    def certify(self, x):
        # one compiled certificate for all problems of the same structure (tables and limits are inputs)
        c = jax.tree_util.tree_map(np.asarray, _CERT(jnp.asarray(x, jnp.float64)[None], jnp.asarray([self.eps]),
                                                     self.V, self.lim, viol_tol=float(self.cfg.viol_tol)))
        return dict(f_B_hi=float(c["f_B_hi"][0]), mean_bn_pct=100 * float(c["mean_bn"][0]),
                    max_bn_pct=100 * float(c["max_bn"][0]), L=float(c["L"][0]), max_kappa=float(c["max_kappa"][0]),
                    msc_max=float(np.max(c["msc"][0])), d_cc=float(c["d_cc"][0]), d_cs=float(c["d_cs"][0]),
                    v=float(c["v"][0]), certified=bool(c["certified"][0]), linked=bool(c["linked"][0]),
                    arc_var_max=float(c["arc_var_max"][0]), L_max_coil=float(c["L_max_coil"][0]),
                    I_rel_min=float(np.min(c["I_rel"][0])), I_rel_max=float(np.max(c["I_rel"][0])),
                    **self.arc_penalty_report(x))

    def arc_penalty_report(self, x):
        """Arclength variation in the penalty's own definition (simsopt ArclengthVariation with the
        configured nintervals, on the optimization quadrature): max over coils [m^2] and the
        penalty term w sum_c max(0, V_c - T)^2 (normalized-flux units, as in the objective)."""
        cfg = self.cfg
        if not cfg.arc_pen_thresh:
            return {}
        from xcoil.field import residuals as rs
        nq = self.T.cb.B0.shape[0]
        M = rs.arc_intervals(nq, rs._arc_nint(cfg, nq))
        v = np.asarray(_ARCVAR(jnp.asarray(x, jnp.float64), self.T.cb, M, self.T.sym.C))
        pen = cfg.arc_pen_weight * float(np.sum(np.maximum(0.0, v - cfg.arc_pen_thresh) ** 2))
        return dict(arc_pen_var_max=float(v.max()), arc_pen=pen)

    def warmup(self):
        x = cur.with_currents(self.seed.x, self.T.cb, self.cfg.n_coils)
        self.F_grad(x, 0.5); self.g_jac(x, 0.5); self.F(x, 0.5); self.g(x, 0.5); self.resid_jac(x, 0.5)
        self.curvature(x, 0.5, np.zeros(self.m)); self.kkt(x, 0.5); self.certify(x)
        self.cnt = Counters()
