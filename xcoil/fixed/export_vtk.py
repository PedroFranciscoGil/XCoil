"""Per-result VTK export (legacy .vtk): full-torus target surface coloured by B.n/|B| (plus |B|,
B.n) of the optimized coils, and all physical coils (cell data: base-coil index, current [A] with
the symmetry sign, current relative to I0).
    export(prob, x, prefix)  ->  prefix_surface.vtk, prefix_coils.vtk
"""
from __future__ import annotations

import os

import numpy as np
import jax
import jax.numpy as jnp

from xcoil.homotopy import surface_path as sp
from xcoil.curves import fourier as fr, symmetry as sym_mod, currents as cur
from xcoil.field import residuals as rs
from xcoil.io.vtk_xml import write_vtk_surface, write_vtk_coils, write_vts, write_vtu_lines


class Exporter:
    def __init__(self, prob, npp=64, nth=128, nq=256):
        b, T = prob.b, prob.T
        tab = sp.build(b, npp * b.nfp, nth, "full", "linear")
        X, n, _ = sp.evaluate(tab, 1.0)
        self.shape = (npp * b.nfp, nth); self.X, self.n = np.asarray(X), np.asarray(n)
        cbd = fr.make_basis(prob.cfg.basis, nq, prob.cfg.order)
        self.field = jax.jit(lambda x: rs.field_on(x, cbd, T.sym, X, T.I0))
        self.coils = jax.jit(lambda x: sym_mod.expand(fr.coils_curves(x, cbd, T.sym.C)[0], T.sym))
        self.base = np.asarray(T.sym.coil, float)
        self.cur = jax.jit(lambda x: cur.phys_currents(x, cbd, T.sym, T.I0))
        self.I0 = float(T.I0)

    def __call__(self, x, prefix, fmt="vtk"):
        """fmt "vtk": legacy prefix_surface.vtk + prefix_coils.vtk; "xml": prefix_surface.vts
        (StructuredGrid) + prefix_coils.vtu (UnstructuredGrid of closed polylines)."""
        x = jnp.asarray(np.asarray(x, float))
        B = np.asarray(self.field(x)); Bn = np.sum(B * self.n, 1); Bm = np.linalg.norm(B, axis=1)
        sh = self.shape
        os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
        surf = {"Bn_over_B": (Bn / Bm).reshape(sh), "modB": Bm.reshape(sh), "Bn": Bn.reshape(sh)}
        I = np.asarray(self.cur(x)); C = np.asarray(self.coils(x))
        cells = {"base_coil": self.base, "current": I, "current_rel": np.abs(I) / abs(self.I0)}
        if fmt == "xml":
            write_vts(prefix + "_surface.vts", self.X.reshape(sh + (3,)), surf)
            write_vtu_lines(prefix + "_coils.vtu", C, cells,
                            {"current": np.repeat(I[:, None], C.shape[1], 1), "idx": np.repeat(self.base[:, None], C.shape[1], 1)})
            return prefix + "_surface.vts", prefix + "_coils.vtu"
        write_vtk_surface(prefix + "_surface.vtk", self.X.reshape(sh + (3,)), surf)
        write_vtk_coils(prefix + "_coils.vtk", C, cells)
        return prefix + "_surface.vtk", prefix + "_coils.vtk"
