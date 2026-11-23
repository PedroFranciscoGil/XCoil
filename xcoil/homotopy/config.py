"""HConfig: the frozen settings of a coil problem (boundary, coils, limits, representation, grids, objective weights,
verification), hashed into results. Every field is described in docs/reference.md (xcoil/options.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, replace, fields

import jax


@dataclass(frozen=True)
class HConfig:
    # problem
    surface: str = "landreman_paul_qa"; n_coils: int = 4; stellsym: bool = True
    kappa_lim: float = 5.0; msc_lim: float = 5.0; dcc: float = 0.1; dcs: float = 0.15
    eps_lo: float = 13.5; eps_hi: float = 19.5
    # representation
    basis: str = "fourier"; order: int = 8; n_ctrl: int = 0; nq_opt: int = 128
    grid_fine: tuple = (32, 64); grid_dist: tuple = (64, 64)
    # surface path and length cap (t = 1 is the target boundary)
    path: str = "linear"; t_h: float = 0.3; disp_guard: float = 0.4
    # objective, model, margins
    f_ref: float = 1e-4
    lagrangian_hessian: str = "exact_blockdiag"; beta_curv: float = 0.03; beta_d: float = 0.0
    # front parameter p[1] of the constraints: "length" (the length cap) or "kappa" (the curvature
    # limit, with the length cap fixed at front_L); see problem_h.cap_at
    front_param: str = "length"; front_L: float = 0.0
    w_arc: float = 1e-3; w_L0: float = 1e-2; w_msc0: float = 0.0; w_msc_final: float = 0.0
    msc_final_ramp: str = "t2"      # w_msc_final ramped in as t^2; "none" = constant
    softmin_alpha: float = 100.0
    dist_segments: int = 0            # distance rows per coil (0 = pointwise)
    # verification
    verify_grid: tuple = (64, 128); nq_verify: int = 768   # resolves the maximum curvature
    viol_tol: float = 1e-3
    precision: str = "fp64"; seed: int = 0
    # length convention, arclength terms
    per_coil_length: bool = False   # length cap per base coil (simsopt auglag_qa) instead of total
    arc_var_lim: float = 0.0        # > 0: per-coil arclength-variation constraint [m^2] (simsopt def.)
    arc_pen_thresh: float = 0.0     # > 0: arclength-variation PENALTY w sum_c max(0, Var_c - T)^2 in the
    arc_pen_weight: float = 1.0     #      objective (normalized-flux units), T [m^2]
    arc_pen_intervals: int = 0      # simsopt nintervals: 0 = "full" (every quadrature point),
                                    # -1 = "partial" (2 * order intervals), n > 0 = n intervals
    beta_L: float = 0.05            # sec. 10.2: eps(0) = L_seed (1 + beta_L)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()
                              ).hexdigest()[:16]

    def with_(self, **kw) -> "HConfig":
        return replace(self, **kw)


# As a pytree (inside compiled programs): the device's limit values are data; the boundary name and
# the length-scan range (build-time only) are blanked; every other field is structure. Devices with
# the same structure (coils, order, quadrature, grids, settings) then share compiled programs.
_CFG_DATA = ("kappa_lim", "msc_lim", "dcc", "dcs")
_CFG_HOST = dict(surface="", eps_lo=0.0, eps_hi=0.0)


def _cfg_flatten(c):
    aux = tuple((f.name, _CFG_HOST.get(f.name, getattr(c, f.name))) for f in fields(c) if f.name not in _CFG_DATA)
    return tuple(getattr(c, k) for k in _CFG_DATA), aux


jax.tree_util.register_pytree_node(
    HConfig, _cfg_flatten, lambda aux, ch: HConfig(**dict(aux), **dict(zip(_CFG_DATA, ch))))


def for_device(key: str, **overrides) -> HConfig:
    """HConfig from the device table (sec. 16.1) plus overrides."""
    from .devices import DEVICES
    d = DEVICES[key]
    cfg = HConfig(surface=d.name, n_coils=d.n_coils, stellsym=d.stellsym, kappa_lim=d.kappa_lim,
                  msc_lim=d.msc_lim, dcc=d.dcc, dcs=d.dcs,
                  eps_lo=d.eps_lo if d.eps_lo is not None else float("nan"),
                  eps_hi=d.eps_hi if d.eps_hi is not None else float("nan"),
                  order=d.order, grid_dist=d.grid_dist, nq_opt=d.nq_opt, w_msc_final=d.w_msc_final,
                  per_coil_length=d.per_coil_length)
    return replace(cfg, **overrides)
