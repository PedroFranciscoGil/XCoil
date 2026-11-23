# Target boundaries

XCoil designs coils for one plasma boundary: a stellarator-symmetric closed surface, held as a `FourierBoundary`

    R(θ, φ) = Σ rc(m, n) cos(mθ − n·nfp·φ),     Z(θ, φ) = Σ zs(m, n) sin(mθ − n·nfp·φ)

in the cylindrical angle φ, with n in units of nfp (the VMEC / simsopt convention). Every way to give one:

| you have | how | notes |
|---|---|---|
| a bundled device | `up.load_boundary("landreman_paul_qa")`, CLI `--surface w7x` | `landreman_paul_qa` (nfp 2, QA), `w7x` (nfp 5), `qh_reactor` (nfp 4, reactor scale), `cfqs_2b40` (nfp 2), `rotating_ellipse`; `xcoil-check` lists them |
| a VMEC / VMEX input file | `up.load_boundary("input.my_device")` | reads `NFP`, `RBC(n,m)`, `ZBS(n,m)`; everything else is ignored |
| a VMEC / VMEX `wout` file | `up.load_boundary("wout_my_device.nc")` | the last closed flux surface; netCDF3 with scipy, netCDF4 with the `netcdf` extra; `up.read_wout(path, s_index)` for another surface |
| Fourier arrays | `bd.from_modes(xm, xn, rc, zs, nfp)` | `from xcoil.homotopy import boundary as bd`; n in units of nfp |
| a simsopt surface object | `bd.from_simsopt(surface)` (a `SurfaceRZFourier`), or `up.refit_rz(surface)` (any simsopt surface, refitted) | needs simsopt |
| a simsopt JSON file (e.g. a QUASR device) | `up.load_boundary("serial0755658.json")` | needs simsopt; surfaces not given as R, Z series in φ (QUASR's `SurfaceXYZTensorFourier`) are refitted; `info["fit_err_m"]` is the fit error |
| a DESC equilibrium | save it as a wout (`desc.vmec.VMECIO.save(eq, "wout_desc.nc")`) or as a VMEC input, then load that | XCoil does not import DESC |
| R, Z on a (θ, φ) grid | `bd.fit_grid(R, Z, nfp)` | exact 2D DFT; R, Z of shape (ntheta, nphi) over one field period |
| an XCoil `.npz` boundary | `up.load_boundary("my_boundary.npz")` | keys xm, xn, rc, zs, nfp, stellsym |

`up.write_vmec_input(b, "input.my_device")` writes any boundary as a minimal VMEC input file, readable by
`load_boundary`, VMEC, VMEX and simsopt. `b.eval_rz(theta, phi)` and `b.eval_xyz(theta, phi)` evaluate a boundary;
`b.R00` is its major radius.

Only stellarator-symmetric boundaries are supported (non-symmetric terms are carried but the coils are always
stellarator symmetric).

## Size and limits

Limits are in metres and scale with the device: copying the LP QA limits (R0 ≈ 1 m) to a reactor-scale boundary
(R0 ≈ 10 m) gives meaningless coils. Scale lengths and distances with the major radius, curvature with its inverse
and the MSC with its inverse square, as tutorial 5 does for the QH reactor:

```python
b, _ = up.load_boundary("qh_reactor")
S = b.R00
prob, _ = up.make_problem(b, "qh", ncoils=4, order=4, length=4.0 * S, per_coil_length=True,
                          kappa=3.0 / S, msc=4.0 / S**2, dcc=0.1 * S, dcs=0.15 * S, nq=64)
```

## Where problems are stored

`make_problem(b, key, ...)` registers the boundary under the name `user/<key>`, stored as `<key>.npz` in
`~/.cache/xcoil/boundaries` (set `XCOIL_USER_DIR` to change it). Use a new key for every different boundary.
