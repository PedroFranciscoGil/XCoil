"""Lp curvature (xcoil.constraints.lp_curvature): the functional on a circle, the rows' Jacobian against central
differences, and (slow) a limit below the unconstrained design's value met and certified."""
import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401  (x64)
from xcoil.constraints import lp_curvature as lpc
from xcoil.curves import fourier as fr


def test_circle():
    """A circle of radius R: kappa = 1/R, J = (1/p) (1/R - k0)^p 2 pi R."""
    R, p, k0 = 0.5, 2.0, 1.0
    cb = fr.make_basis("fourier", 256, 1)
    x = np.zeros(9); x[2] = R; x[4] = R
    J = float(lpc.lp_of_x(jnp.asarray(x), cb, 1, p, k0)[0])
    assert abs(J - (1 / p) * (1 / R - k0) ** p * 2 * math.pi * R) < 1e-10


def test_rows_jacobian():
    from problems import problem_for, starts_for
    prob = problem_for("qa_auglag")
    x = jnp.asarray(starts_for(prob)[0].x, float)
    ex = lpc.lp_extra(prob, 0.5, 2.0, 2.0)
    J = np.asarray(ex.jg(x))
    v = np.random.default_rng(0).normal(size=x.size); v /= np.linalg.norm(v)
    h = 1e-6
    fd = (np.asarray(ex.g(x + h * v)) - np.asarray(ex.g(x - h * v))) / (2 * h)
    assert np.max(np.abs(J @ v - fd)) <= 1e-6 * max(1.0, np.max(np.abs(fd)))


@pytest.mark.slow
def test_lp_limit_in_the_solver():
    """LP-QA: J_max 20 % below the unconstrained design's largest J_c (p 2, threshold 2 /m) is met and certified."""
    from xcoil.fixed import user_problem as up
    b, _ = up.load_boundary("landreman_paul_qa")
    prob, _ = up.make_problem(b, "lpqa_lp", ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0,
                              msc=4.0, dcc=0.1, dcs=0.15, nq=64)
    start, _ = up.initial_coils(prob, init="circle", size="half-R0")
    quiet = lambda *a, **k: None
    r0 = up.best(up.solve_all(prob, [start], log=quiet)[0])
    J0 = max(lpc.certify_lp(r0["x"], prob, 1.0, 2.0, 2.0)["lp_curvature"])
    lim = 0.8 * J0
    r = up.best(up.solve_all(prob, [up.warm_start(start, r0["x"])], max_outer=30, log=quiet,
                             lp_curvature=dict(J_max=lim, p=2.0, threshold=2.0))[0])
    c = r["cert"]["lp_curvature"]
    assert c["ok"] and r["cert"]["certified"], c
