"""Toroidal ordering rows (xcoil.constraints.ordering): the fixed reference axis, feasible and crossing coil sets,
the rows' Jacobian, and independence of the boundary once the axis is fixed."""
import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401  (x64)
from xcoil.constraints import ordering as od
from xcoil.curves import fourier as fr
from xcoil.homotopy import boundary as bd


@pytest.mark.parametrize("name", ["landreman_paul_qa", "w7x", "qh_reactor"])
def test_axis_from_modes_equals_poloidal_mean(name):
    """The m = 0 modes are the theta-mean of R and Z at each phi (axis_from_points on a full-torus grid)."""
    b = bd.load(name)
    nfp, nphi, ntheta = int(b.nfp), 128, 64
    ph = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    th = 2 * np.pi * np.arange(ntheta) / ntheta
    S = b.eval_xyz(th[None, :], ph[:, None]).reshape(-1, 3)
    nmax = max(1, min(8, nphi // (2 * nfp) - 1))
    a1, b1 = od.axis_from_boundary(b, nmax)
    a2, b2 = (np.asarray(v) for v in od.axis_from_points(jnp.asarray(S), nphi, ntheta, nfp, nmax))
    assert np.allclose(a1, a2, atol=1e-12) and np.allclose(b1, b2, atol=1e-12)


@pytest.fixture(scope="module")
def lpqa():
    from problems import problem_for, starts_for
    prob = problem_for("qa_auglag")
    x = np.asarray(starts_for(prob)[0].x, float)
    return prob, x


def _rows(prob, x, axis=None, gap_deg=1.0):
    return np.asarray(od.ordering_extra(prob, axis, gap_deg=gap_deg).g(jnp.asarray(x)))


def test_ordered_start_feasible_and_swap_violates(lpqa):
    """Evenly spaced starting coils satisfy the rows; swapping coils 0 and 1 makes their pair cross."""
    prob, x = lpqa
    assert np.max(_rows(prob, x)) < 0
    C, K = prob.T.sym.C, prob.T.cb.K
    xs = x.copy()
    c0, c1 = slice(0, 3 * K), slice(3 * K, 6 * K)
    xs[c0], xs[c1] = x[c1], x[c0]
    r = _rows(prob, xs).reshape(C + 1, -1)
    assert np.max(r[0]) > 0                              # pair 0|1 crosses
    rep = od.certify_ordering(xs, prob)
    assert not rep["ok"] and rep["gap_window_deg"][0] < 0


def test_rows_jacobian(lpqa):
    prob, x = lpqa
    ex = od.ordering_extra(prob)
    J = np.asarray(ex.jg(jnp.asarray(x)))
    v = np.random.default_rng(0).normal(size=x.size); v /= np.linalg.norm(v)
    h = 1e-6
    fd = (np.asarray(ex.g(jnp.asarray(x + h * v))) - np.asarray(ex.g(jnp.asarray(x - h * v)))) / (2 * h)
    assert np.max(np.abs(J @ v - fd)) <= 1e-6 * max(1.0, np.max(np.abs(fd)))


def test_fixed_axis_rows_do_not_depend_on_the_boundary(lpqa):
    """With an explicit axis, the rows of a problem on another boundary are the same (they depend on the coils only);
    with the default axis they follow the boundary's own axis."""
    prob, x = lpqa
    axis = od.axis_from_boundary(prob.b)
    moved = bd.from_modes(prob.b.xm, prob.b.xn, prob.b.rc * 1.0, prob.b.zs * 1.0, prob.b.nfp)
    sel = (moved.xm == 0) & (moved.xn == 1)
    moved.rc[sel] += 0.05                                # shift the axis
    from types import SimpleNamespace
    other = SimpleNamespace(T=prob.T, b=moved)
    assert np.array_equal(_rows(prob, x, axis), _rows(other, x, axis))
    assert not np.allclose(_rows(prob, x), _rows(other, x))


def test_ordering_in_the_solver():
    """LP-QA (simsopt auglag_qa problem) solved with the ordering rows: certified, no pair crossing, every penalized
    gap at least the 1 degree minimum, about the boundary's fixed axis."""
    from xcoil.fixed import user_problem as up
    b, _ = up.load_boundary("landreman_paul_qa")
    prob, _ = up.make_problem(b, "lpqa_order", ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0,
                              msc=4.0, dcc=0.1, dcs=0.15, nq=64)
    start, _ = up.initial_coils(prob, init="circle", size="half-R0")
    r = up.best(up.solve_all(prob, [start], log=lambda *a, **k: None, ordering=dict(gap_deg=1.0))[0])
    o = r["cert"]["ordering"]
    assert o["ok"] and min(o["gap_window_deg"]) > 0 and min(o["gap_penalized_deg"]) >= 1.0 - 1e-6, o
    assert r["cert"]["certified"]
