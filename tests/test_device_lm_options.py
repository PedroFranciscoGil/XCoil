"""device_lm options (near_rows, damp, rho_rule, irls, dist_margin, cs_window, ftol) and their defaults; target mode
on shared and per-problem programs; the batched multi-start solve."""
from __future__ import annotations


import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401
from xcoil.fixed import device_lm as dl
import problems as pd_
from xcoil.fixed.device_lm import MAIN

BUDGET = dict(MAIN, link_test=True)
MAX_OUTER = 4


def _problem():
    prob = pd_.problem_for("qa_auglag")
    x0 = np.asarray([s for s in pd_.starts_for(prob) if s.label == "L100"][0].x)
    if x0.size < prob.n:                                  # free currents
        x0 = np.concatenate([x0, np.zeros(prob.n - x0.size)])
    return prob, x0


@pytest.fixture(scope="module")
def setup():
    return _problem()


def run_defaults(prob, x0, budget=BUDGET):
    """(x, outer, trials, jac, rho) of a short solve in general and in target mode."""
    out = {}
    target = (2.0 * pd_.QA_TARGET_MEAN, None)
    for mode, solve in (("gen", dl.build_solver(prob, budget, max_outer=MAX_OUTER)),
                        ("tgt", dl.build_solver(prob, budget, target=target, monitor=dl.make_monitor(prob),
                                                max_outer=MAX_OUTER))):
        x, info = solve(x0)
        out[f"{mode}.x"] = np.asarray(x)
        for k in ("outer", "trials", "jac", "rho"):
            out[f"{mode}.{k}"] = np.asarray(info[k])
    return out


def test_tables_jax_match_numpy(setup, monkeypatch):
    """The default JAX boundary tables agree with the numpy ones at rounding level (same problem, same start)."""
    a, xa = setup
    fa = run_defaults(a, xa, BUDGET)
    monkeypatch.setenv("XCOIL_TABLES", "numpy")
    b, xb = _problem()
    assert np.allclose(xa, xb, rtol=0, atol=1e-8 * np.abs(xb).max())
    fb = run_defaults(b, xb, BUDGET)
    assert abs(np.linalg.norm(fa["gen.x"] - fb["gen.x"])) <= 2e-2 * np.linalg.norm(fb["gen.x"])


def test_explicit_defaults_identical(setup):
    prob, x0 = setup
    a, b = run_defaults(prob, x0, BUDGET), run_defaults(prob, x0, dict(BUDGET, **dl.DEFAULTS))
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_defaults_are_the_new_setting():
    o = dl._al_opts(BUDGET)
    assert (o["near"], o["damp"], o["rule"], o["irls"], o["margin"], o["cs_window"]) == \
        (1e-3, "fourier", "guard", 0.1, 0.01, 256)
    assert o["stall"] == 3
    assert (MAIN["inner"], MAIN["first_prog"]) == (50, 0.03)


def test_guard_raises_rho_after_stalled_outers():
    """guard: rho grows after rho_stall consecutive unconverged outers whose V did not decrease, not before."""
    o = dl._al_opts(BUDGET)
    g, lam = jnp.asarray([1.0]), jnp.zeros(1)
    rho, omega, eta, V = jnp.asarray(10.0), jnp.asarray(1e-3), jnp.asarray(0.08), jnp.asarray(0.5)
    rhos = []
    for _ in range(4):                       # stat > omega: unconverged; V stays at 1 >= previous
        lam, rho, omega, eta, V = dl._al_update(o, g, jnp.zeros(1), rho, omega, eta, jnp.asarray(1.0), V)
        rhos.append(float(rho))
    assert rhos == [10.0, 10.0, 100.0, 100.0]   # 1st outer: V rose from 0.5 (stall 1), then 2, 3 -> raise, reset


@pytest.mark.parametrize("opts", [dict(), dict(near_rows=0.0), dict(damp="levenberg"), dict(rho_rule="conv"),
                                  dict(rho_rule="lancelot"), dict(rho_rule="algencan"), dict(irls=0.0),
                                  dict(dist_margin=None), dict(cs_window=0)])
def test_options_run(setup, opts):
    prob, x0 = setup
    x, info = dl.build_solver(prob, dict(BUDGET, **opts), max_outer=MAX_OUTER)(x0)
    x = np.asarray(x)
    assert np.all(np.isfinite(x)) and 1 <= int(info["outer"]) <= MAX_OUTER
    assert prob.F(x, 1.0) < prob.F(x0, 1.0)


def test_unknown_option_values_rejected(setup):
    prob, _ = setup
    with pytest.raises(AssertionError):
        dl.build_solver(prob, dict(BUDGET, damp="other"))
    with pytest.raises(AssertionError):
        dl.build_solver(prob, dict(BUDGET, rho_rule="other"))


def test_mode_weights(setup):
    prob, _ = setup
    T = prob.T
    d = dl._mode_weights(T, prob.n)
    K, nc = T.cb.K, T.sym.C * 3 * T.cb.K
    assert d.shape == (prob.n,) and np.all(d[nc:] == 1.0)
    assert d[0] == 1.0 and d[1] == d[2] == 4.0
    m = (np.arange(nc) % K + 1) // 2
    assert np.array_equal(d[:nc], (1.0 + m) ** 2)


def test_irls_weights_keep_the_objective_size(setup):
    prob, x0 = setup
    T = prob.T
    q = dl.irls_weights(T, jnp.asarray(x0), 1.0, 0.1)
    from xcoil.homotopy import problem_h as ph
    from xcoil.field import residuals as rs
    X, n, w = ph.surface(T, 1.0)
    r = rs.magnetic_residual(jnp.asarray(x0), T.cb, T.sym, X, n, w, T.B_ref, T.I0)
    assert np.isclose(float(jnp.sum((q * r) ** 2)), float(jnp.sum(r * r)), rtol=1e-10)
    assert np.all(np.asarray(q) > 0)


def test_coil_surface_jacobian_rows(setup):
    from xcoil.constraints import pointwise as pw
    from xcoil.curves import fourier as fr
    from xcoil.homotopy import surface_path as sp
    prob, x0 = setup
    T = prob.T
    g = fr.coils_curves(jnp.asarray(x0), T.cb, T.sym.C)[0]
    S = sp.eval_xyz(T.dist, 1.0)
    full = np.asarray(pw.coil_surface_jacobian(g, T.cb, S, T.lim.alpha_cs, jnp.float32))
    nq = g.shape[1]
    ci = jnp.asarray([0, 1, T.sym.C - 1, 0]); ki = jnp.asarray([3, nq // 2, nq - 1, 0])
    rows = np.asarray(pw.coil_surface_jacobian_rows(g, T.cb, S, T.lim.alpha_cs, ci, ki, jnp.float32))
    assert np.allclose(rows, full[np.asarray(ci) * nq + np.asarray(ki)], rtol=1e-6, atol=1e-10)


def test_cs_window_covering_the_grid_is_exact(setup):
    """A window as large as the distance grid gives the whole-grid coil-surface soft-min (same points, reordered)."""
    from xcoil.constraints import pointwise as pw
    from xcoil.curves import fourier as fr
    from xcoil.homotopy import surface_path as sp
    prob, x0 = setup
    T = prob.T
    S = sp.eval_xyz(T.dist, 1.0)
    g = fr.coils_curves(jnp.asarray(x0), T.cb, T.sym.C)[0]
    d2 = jnp.sum((g[:, :, None, :] - S[None, None]) ** 2, -1)
    W = S[jax.lax.top_k(-d2, S.shape[0])[1]]
    a = pw.coil_surface_softmin(g, S, T.lim.alpha_cs)
    b = pw.coil_surface_softmin(g, W, T.lim.alpha_cs)
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=0)


def test_shared_target_mode(setup):
    """Target mode on the shared programs (default) and on per-problem programs (shared=False): the same stop
    decision and the same design to rounding; a second problem of the same structure reuses the compiled stages."""
    prob, x0 = setup
    target = (2.0 * pd_.QA_TARGET_MEAN, None)
    mon = dl.make_monitor(prob)
    xs, info_s = dl.build_solver(prob, BUDGET, target=target, monitor=mon, max_outer=MAX_OUTER)(x0)
    xp, info_p = dl.build_solver(prob, BUDGET, target=target, monitor=mon, max_outer=MAX_OUTER, shared=False)(x0)
    assert int(info_s["outer"]) == int(info_p["outer"]) and bool(info_s["stopped"]) == bool(info_p["stopped"])
    # the two programs differ at rounding level (constants vs inputs), which the LM iteration amplifies: compare the
    # designs' field errors, not their coefficients
    Fs, Fp = float(prob.F(np.asarray(xs), 1.0)), float(prob.F(np.asarray(xp), 1.0))
    assert abs(Fs - Fp) <= 0.05 * max(Fs, Fp)
    n = len(dl._SHARED)
    prob2 = pd_.problem_for("qa_auglag")                                  # a new problem object, same structure
    dl.build_solver(prob2, BUDGET, target=target, monitor=dl.make_monitor(prob2), max_outer=MAX_OUTER)(x0)
    assert len(dl._SHARED) == n



def test_batched_multistart(setup):
    """build_batched_solver: B starts in one program; every member follows its own solve. vmapped reductions round
    differently, so after 2 outers the field error agrees to ~1e-3, not bit for bit."""
    prob, x0 = setup
    X0 = np.stack([x0, x0 * (1.0 + 1e-3)])
    X, info = dl.build_batched_solver(prob, BUDGET, max_outer=2)(X0)
    x1, _ = dl.build_solver(prob, BUDGET, max_outer=2)(X0[0])
    assert np.asarray(X).shape == X0.shape and np.asarray(info["outer"]).shape == (2,)
    F1, Fb = float(prob.F(np.asarray(x1), 1.0)), float(prob.F(np.asarray(X)[0], 1.0))
    assert abs(F1 - Fb) <= 5e-3 * F1
