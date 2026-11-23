"""Named coil problems: a bundled boundary with its coil count, limits and resolution.

config.for_device(key) builds the problem settings (HConfig) of an entry; user_problem.make_problem
registers every user problem here as "user_<key>". Limits are (kappa_lim [1/m], msc_lim [1/m^2],
d_cc [m], d_cs [m]); eps_lo / eps_hi bound the total base-coil length [m] (per coil when
per_coil_length is True; None: set from the starting coils).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    name: str               # boundary name (bundled .npz under surface/data, or user/<key>)
    n_coils: int
    stellsym: bool
    kappa_lim: float
    msc_lim: float
    dcc: float
    dcs: float
    eps_lo: float | None
    eps_hi: float | None
    order: int              # Fourier order M of every coil
    grid_dist: tuple
    reference: str = ""     # source of the boundary and limits
    nq_opt: int = 128       # quadrature points per coil
    w_msc_final: float = 0.0
    per_coil_length: bool = False  # eps is a per-coil length cap (simsopt auglag_qa convention)


DEVICES = {
    "lpqa": Device("landreman_paul_qa", 4, True, 5.0, 5.0, 0.10, 0.15, 13.5, 19.5, 8, (64, 64),
                   reference="Landreman-Paul precise QA (Phys. Rev. Lett. 128, 035001, 2022)"),
    "w7x": Device("w7x", 4, True, 2.5, 1.5, 0.30, 0.35, 38.0, 38.0, 12, (96, 64),
                  reference="W7-X standard configuration", nq_opt=256, w_msc_final=2e-4),
    "qa_auglag": Device("landreman_paul_qa", 4, True, 3.0, 4.0, 0.10, 0.15, 4.0, 4.0, 4, (64, 64),
                        reference="simsopt examples/3_Advanced auglag_qa.py: 4 coils, order 4, L_c <= 4 m per coil, "
                                  "d_cc 0.1, d_cs 0.15, kappa 3, MSC 4",
                        nq_opt=128, per_coil_length=True),
    "qh_auglag": Device("qh_reactor", 4, True, 1.0, 0.1, 0.8, 1.0, 160.0, 160.0, 7, (96, 64),
                        reference="simsopt examples/3_Advanced auglag_qh_reactorscale.py: 4 coils, order 7, "
                                  "total L <= 160 m, d_cc 0.8, d_cs 1.0, kappa 1, MSC 0.1",
                        nq_opt=384),
    "qh_reactor": Device("qh_reactor", 4, True, 1.0, 0.1, 0.8, 1.0, None, 160.0, 8, (96, 64),
                         reference="Landreman-Paul precise QH at reactor scale (simsopt auglag_qh_reactorscale)"),
    "ellipse": Device("rotating_ellipse", 4, True, 5.0, 5.0, 0.15, 0.15, 15.0, 20.0, 8, (64, 64),
                      reference="rotating ellipse (simsopt examples)"),
    "cfqs": Device("cfqs_2b40", 4, True, 5.0, 5.0, 0.10, 0.15, None, None, 8, (64, 64),
                   reference="CFQS (input.cfqs_2b40), nfp 2 QA"),
}
