"""Size classes: round a problem's resolution up to a few fixed values, so that devices with the same
structure (field periods, coils, Fourier order) share compiled programs (device_lm shared solver,
quick check, certificate - the tables are program inputs).

  nq       points per coil: the smallest class >= the requested value (finer, never coarser)
  nphi     toroidal points of the coil-surface distance grid: the smallest base class >= the request,
           then up to a multiple of 2 nfp
Rounding up only refines the discretization, so a class result is at least as accurate as the
unrounded one; the cost is the extra points (at most one class step: <= 50 %, mostly 15-30 %).

nq = 1024 is not a class: on some GPUs XLA spends tens of minutes compiling some of the solver's
reduction fusions at exactly 1024 points per coil (960 and 1056 compile in about a minute), so 1024
rounds up to 1152.

    key = structure_key(nfp, C, order, nq, grid_dist, ...)   # devices with equal keys share programs
"""
from __future__ import annotations

import math

NQ_CLASSES = (128, 256, 384, 512, 640, 768, 896, 1152, 1280, 1536, 1792, 2048)
NPHI_BASE = (64, 96, 128, 192, 256, 384, 512, 768, 1024)


def nq_class(nq):
    for c in NQ_CLASSES:
        if c >= nq:
            return c
    return NQ_CLASSES[-1]


def nphi_class(nphi, nfp):
    base = next((c for c in NPHI_BASE if c >= nphi), NPHI_BASE[-1])
    return 2 * nfp * math.ceil(base / (2 * nfp))


def grid_class(grid, nfp):
    """(nphi, ntheta) distance grid -> its class (ntheta unchanged)."""
    return (nphi_class(int(grid[0]), nfp), int(grid[1]))


def structure_key(nfp, n_coils, order, nq, grid_dist, per_coil_length=False, stellsym=True):
    """What decides whether two problems can share compiled programs (with equal settings)."""
    return (int(nfp), int(n_coils), int(order), int(nq), tuple(int(v) for v in grid_dist), bool(per_coil_length),
            bool(stellsym))
