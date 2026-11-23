"""Small test problems built from the named devices (xcoil.homotopy.devices).

  qa_auglag  Landreman-Paul QA with the limits of simsopt's auglag_qa.py (4 coils, order 4, L_c <= 4 m per coil,
             d_cc 0.1, d_cs 0.15, kappa 3, MSC 4)
  w7x        W7-X, 4 coils, total length 38 m, d_cc 0.30, d_cs 0.35, kappa 2.5, MSC 1.5
Both with the arclength-variation penalty of auglag_qa.py and a coil-surface distance grid fine enough
toroidally for the distance limit.
"""
from __future__ import annotations

from xcoil.fixed.user_problem import ARC, start_cap

SETUPS = {
    "qa_auglag": dict(eps=4.0, overrides=dict(grid_dist=(256, 64), **ARC)),
    "w7x": dict(eps=38.0, overrides=dict(kappa_lim=2.5, msc_lim=1.5, dcc=0.3, dcs=0.35, grid_dist=(384, 64), **ARC)),
}
# a field-error target (mean |B.n|/|B| [%]) that the qa_auglag problem reaches from its 3D starts
QA_TARGET_MEAN = 0.5855 * 1.01


def problem_for(dev, **kw):
    from xcoil.explore.problem import CoilProblem
    d = SETUPS[dev]
    return CoilProblem(dev, d["eps"], **dict(d["overrides"], **kw))


def starts_for(prob, families=("offset_3d",)):
    from xcoil.fixed.starts import generate
    return generate(prob.b, prob.cfg.n_coils, prob.cfg.order, start_cap(prob), prob.lim.dcs, families=families)
