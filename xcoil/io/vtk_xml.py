"""Minimal XML VTK writers (numpy only): StructuredGrid surfaces (.vts), PolyData polylines (.vtp),
UnstructuredGrid polylines (.vtu, as simsopt's coils_to_vtk) and a .pvd collection (one entry per
frame and part); legacy ASCII .vtk writers below."""
from __future__ import annotations

import os

import numpy as np


def _arr(name, a, ncomp=1):
    a = np.asarray(a, float).reshape(-1)
    body = " ".join(f"{v:.9g}" for v in a)
    return (f'<DataArray type="Float64" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">'
            f"{body}</DataArray>")


def write_vts(path, P, point_data=None):
    """Closed structured surface: P (nphi, ntheta, 3) on a periodic grid; the first row / column
    are appended so ParaView draws a closed torus. point_data: {name: (nphi, ntheta) array}."""
    P = np.asarray(P, float)
    Pc = np.concatenate([P, P[:1]], 0); Pc = np.concatenate([Pc, Pc[:, :1]], 1)
    nu, nv = Pc.shape[0], Pc.shape[1]
    pts = np.transpose(Pc, (1, 0, 2)).reshape(-1, 3)          # VTK order: i (phi) fastest
    pd = ""
    for name, v in (point_data or {}).items():
        v = np.asarray(v, float)
        vc = np.concatenate([v, v[:1]], 0); vc = np.concatenate([vc, vc[:, :1]], 1)
        pd += _arr(name, np.transpose(vc, (1, 0)).reshape(-1))
    ext = f"0 {nu - 1} 0 {nv - 1} 0 0"
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">'
                f'<StructuredGrid WholeExtent="{ext}"><Piece Extent="{ext}"><PointData>{pd}</PointData>'
                f'<Points>{_arr("Points", pts, 3)}</Points></Piece></StructuredGrid></VTKFile>\n')


def write_vtp_lines(path, curves, cell_data=None):
    """Closed polylines: curves (ncurve, n, 3). cell_data: {name: (ncurve,) array}."""
    curves = np.asarray(curves, float)
    nc, n, _ = curves.shape
    pts = curves.reshape(-1, 3)
    conn = " ".join(" ".join(str(c * n + k) for k in list(range(n)) + [0]) for c in range(nc))
    offs = " ".join(str((c + 1) * (n + 1)) for c in range(nc))
    cd = "".join(_arr(k, v) for k, v in (cell_data or {}).items())
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">'
                f'<PolyData><Piece NumberOfPoints="{nc * n}" NumberOfLines="{nc}"><CellData>{cd}</CellData>'
                f'<Points>{_arr("Points", pts, 3)}</Points>'
                f'<Lines><DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>'
                f'<DataArray type="Int64" Name="offsets" format="ascii">{offs}</DataArray></Lines>'
                f'</Piece></PolyData></VTKFile>\n')


def write_vtu_lines(path, curves, cell_data=None, point_data=None):
    """Closed polylines as an UnstructuredGrid (.vtu; one VTK_POLY_LINE cell per curve, closed by
    repeating its first point). curves (ncurve, n, 3); cell_data {name: (ncurve,)}; point_data
    {name: (ncurve, n)}."""
    curves = np.asarray(curves, float)
    nc, n, _ = curves.shape
    pts = curves.reshape(-1, 3)
    conn = " ".join(" ".join(str(c * n + k) for k in list(range(n)) + [0]) for c in range(nc))
    offs = " ".join(str((c + 1) * (n + 1)) for c in range(nc))
    types = " ".join(["4"] * nc)                                  # VTK_POLY_LINE
    cd = "".join(_arr(k, v) for k, v in (cell_data or {}).items())
    pd = "".join(_arr(k, np.asarray(v, float).reshape(-1)) for k, v in (point_data or {}).items())
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">'
                f'<UnstructuredGrid><Piece NumberOfPoints="{nc * n}" NumberOfCells="{nc}">'
                f'<PointData>{pd}</PointData><CellData>{cd}</CellData>'
                f'<Points>{_arr("Points", pts, 3)}</Points>'
                f'<Cells><DataArray type="Int64" Name="connectivity" format="ascii">{conn}</DataArray>'
                f'<DataArray type="Int64" Name="offsets" format="ascii">{offs}</DataArray>'
                f'<DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray></Cells>'
                f'</Piece></UnstructuredGrid></VTKFile>\n')


def write_pvd(path, entries):
    """entries: list of (timestep, part, relative file path)."""
    rows = "\n".join(f'    <DataSet timestep="{ts:.9g}" group="" part="{part}" file="{fn}"/>' for ts, part, fn in entries)
    with open(path, "w") as f:
        f.write(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
                f"  <Collection>\n{rows}\n  </Collection>\n</VTKFile>\n")


class Recorder:
    """Collects frames (surface + coils) and writes the .pvd at the end."""

    def __init__(self, outdir, name="run"):
        self.dir = outdir; self.name = name; self.entries = []; self.frame = 0
        os.makedirs(outdir, exist_ok=True)

    def add(self, surface_P, surface_data, coils, coil_data=None):
        k = self.frame
        fs, fc = f"{self.name}_surface_{k:04d}.vts", f"{self.name}_coils_{k:04d}.vtp"
        write_vts(os.path.join(self.dir, fs), surface_P, surface_data)
        write_vtp_lines(os.path.join(self.dir, fc), coils, coil_data)
        self.entries += [(k, 0, fs), (k, 1, fc)]
        self.frame += 1
        write_pvd(os.path.join(self.dir, f"{self.name}.pvd"), self.entries)


# --- legacy .vtk (ASCII) ---------------------------------------------------------------------

def write_vtk_surface(path, P, point_data=None, title="xcoil surface"):
    """Legacy VTK STRUCTURED_GRID of a closed periodic surface P (nphi, ntheta, 3) (the first row /
    column are appended so the torus closes); point_data {name: (nphi, ntheta)} scalars."""
    P = np.asarray(P, float)
    Pc = np.concatenate([P, P[:1]], 0); Pc = np.concatenate([Pc, Pc[:, :1]], 1)
    nu, nv = Pc.shape[0], Pc.shape[1]
    pts = np.transpose(Pc, (1, 0, 2)).reshape(-1, 3)
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET STRUCTURED_GRID\n"
                f"DIMENSIONS {nu} {nv} 1\nPOINTS {nu * nv} double\n")
        np.savetxt(f, pts, fmt="%.9g")
        if point_data:                        # FIELD block: every reader loads all arrays
            f.write(f"POINT_DATA {nu * nv}\nFIELD FieldData {len(point_data)}\n")
            for name, v in point_data.items():
                v = np.asarray(v, float)
                vc = np.concatenate([v, v[:1]], 0); vc = np.concatenate([vc, vc[:, :1]], 1)
                f.write(f"{name} 1 {nu * nv} double\n")
                np.savetxt(f, np.transpose(vc, (1, 0)).reshape(-1), fmt="%.9g")


def write_vtk_coils(path, curves, cell_data=None, title="xcoil coils"):
    """Legacy VTK POLYDATA of closed polylines curves (ncurve, n, 3); cell_data {name: (ncurve,)}."""
    curves = np.asarray(curves, float)
    nc, n, _ = curves.shape
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET POLYDATA\nPOINTS {nc * n} double\n")
        np.savetxt(f, curves.reshape(-1, 3), fmt="%.9g")
        f.write(f"LINES {nc} {nc * (n + 2)}\n")
        for c in range(nc):
            f.write(" ".join(str(v) for v in [n + 1] + [c * n + k for k in range(n)] + [c * n]) + "\n")
        if cell_data:
            f.write(f"CELL_DATA {nc}\nFIELD FieldData {len(cell_data)}\n")
            for name, v in cell_data.items():
                f.write(f"{name} 1 {nc} double\n")
                np.savetxt(f, np.asarray(v, float).reshape(-1), fmt="%.9g")
