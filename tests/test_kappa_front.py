"""Curvature-limit fronts: with front_param="kappa" the constraint caps are [front_L, kappa/kappa_lim];
at the nominal limit the constraints equal those of the length front at front_L, and the curvature
rows scale exactly with the limit."""
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from problems import problem_for, starts_for
from xcoil.homotopy import problem_h as ph
from xcoil.constraints import pointwise as pw


def test_kappa_front_constraints():
    pl = problem_for("qa_auglag", nq_opt=64)
    pk = problem_for("qa_auglag", nq_opt=64, front_param="kappa", front_L=pl.eps)
    x = jnp.asarray(starts_for(pl)[0].x)
    kn = pl.lim.kappa_lim
    gl, Al = ph.cons_jac(pl.T, x, (1.0, pl.eps))
    gk, Ak = ph.cons_jac(pk.T, x, (1.0, kn))
    assert np.allclose(gl, gk) and np.allclose(Al, Ak)
    # doubling the curvature limit: curvature rows (kap - (1-b) 2k)/(2k) = (g + 1 - b)/2 - (1 - b)
    g2, A2 = ph.cons_jac(pk.T, x, (1.0, 2 * kn))
    bl = pw.blocks_for(pk.cfg.n_coils, pk.T.cb.B0.shape[0], pk.lim)["curv"]
    b = pk.lim.beta
    assert np.allclose(g2[bl], (gl[bl] + 1 - b) / 2 - (1 - b))
    assert np.allclose(A2[bl], Al[bl] / 2)
    other = np.ones(gl.shape[0], bool); other[bl] = False
    assert np.allclose(g2[other], gl[other])
    # the derivative of the constraints in the front parameter (used by the tangent) is nonzero
    dg = jax.jvp(lambda e: ph.cons(pk.T, x, (1.0, e)), (jnp.asarray(kn),), (jnp.asarray(1.0),))[1]
    assert np.all(dg[other] == 0) and np.any(dg[bl] != 0)
