"""Coil forces (xcoil.constraints.forces): the regularized self-force of a circular coil against the closed form,
Jacobian of the force rows vs finite differences, and the max-force constraint inside the solver."""
import math

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import xcoil  # noqa: F401  (x64)
from xcoil.constraints import forces as fo
from xcoil.curves import fourier as fr
from xcoil.curves import symmetry as sym_mod

MU0 = 4e-7 * math.pi


def _circle(R, nq=256):
    cb = fr.make_basis("fourier", nq, 1)                 # K = 3: (1, sin, cos)
    x = np.zeros(9)
    x[2] = R                                             # x = R cos(2 pi s)
    x[4] = R                                             # y = R sin(2 pi s)
    return jnp.asarray(x), cb


@pytest.mark.parametrize("R,a", [(1.0, 0.01), (2.0, 0.1)])
def test_hoop_force_of_a_ring(R, a):
    """One circular coil (no other coils): |dF/dl| = mu0 I^2 / (4 pi R) (ln(8 R / a) - 3/4) for a round conductor with
    uniform current (the regularization delta = a^2 / sqrt(e) reproduces it)."""
    x, cb = _circle(R)
    sym = sym_mod.build(1, 1, False)
    I = jnp.asarray([1.0e6])
    F = np.asarray(fo.coil_forces(x, I, cb, sym, fo.regularization_circ(a)))
    exact = MU0 * 1e12 / (4 * math.pi * R) * (math.log(8 * R / a) - 0.75)
    assert np.allclose(F, exact, rtol=1e-6)


def test_two_rings_attract_more():
    """Two coaxial rings with equal currents (nfp 2, stellarator symmetric, one base coil: a pair of images): the
    mutual field adds to the hoop force of each, which stays uniform along the coil."""
    x, cb = _circle(1.0)
    x = x.at[8].set(0.0)
    x = x.at[6].add(0.3)                                 # z offset 0.3: images at z = +-0.3
    sym = sym_mod.build(1, 1, True)
    I = jnp.asarray([1.0e6])
    F = np.asarray(fo.coil_forces(x, I, cb, sym, fo.regularization_circ(0.01)))
    alone = MU0 * 1e12 / (4 * math.pi) * (math.log(800) - 0.75)
    assert np.ptp(F) < 1e-6 * F.mean() and abs(F.mean() - alone) > 1e-3 * alone


def test_force_rows_jacobian():
    """jax.jacrev of the rows (coefficients and free currents) vs central differences."""
    from problems import problem_for, starts_for
    prob = problem_for("qa_auglag")
    T = prob.T
    from xcoil.curves import currents as cur
    x = np.array(cur.with_currents(starts_for(prob)[0].x, T.cb, T.sym.C), float)
    x[cur.n_geo(T.cb, T.sym.C):] = np.linspace(-0.05, 0.05, T.sym.C - 1)    # unequal currents
    x = jnp.asarray(x)
    cb = fo.basis_like(T.cb, 32)
    f = lambda y: fo.force_rows(y, cb, T.sym, T.I0, 1.0e5, fo.regularization_circ(0.02))
    J = np.asarray(jax.jacrev(f)(x))
    rng = np.random.default_rng(0)
    v = rng.normal(size=x.shape[0]); v /= np.linalg.norm(v)
    h = 1e-6
    fd = (np.asarray(f(x + h * v)) - np.asarray(f(x - h * v))) / (2 * h)
    assert np.max(np.abs(J @ v - fd)) <= 1e-6 * np.max(np.abs(fd))


@pytest.mark.slow
def test_max_force_constraint_in_the_solver():
    """LP-QA (simsopt auglag_qa problem, 4 coils of order 4, 64 points per coil): from the unconstrained design, a force
    limit 10 % below its maximum is met and certified on 768 points (with a 0.5 % margin for the sampling gap between
    the 64 solver points and the certificate). About 3 minutes on a laptop CPU."""
    import dataclasses
    from xcoil.fixed import user_problem as up
    b, _ = up.load_boundary("landreman_paul_qa")
    prob, _ = up.make_problem(b, "lpqa_force", ncoils=4, order=4, length=4.0, per_coil_length=True, kappa=3.0,
                              msc=4.0, dcc=0.1, dcs=0.15, nq=64)
    start, _ = up.initial_coils(prob, init="circle", size="half-R0")
    quiet = lambda *a, **k: None
    r0 = up.best(up.solve_all(prob, [start], log=quiet)[0])
    F0 = max(fo.certify_force(r0["x"], prob, 1.0, 0.02)["max_force"])
    lim = 0.9 * F0
    warm = dataclasses.replace(start, x=np.asarray(r0["x"]), label="warm")
    r = up.best(up.solve_all(prob, [warm], log=quiet, max_outer=30,
                             max_force=dict(F_lim=lim, radius=0.02, margin=0.005))[0])
    f = r["cert"]["force"]
    assert f["ok"] and max(f["max_force"]) <= lim, f
