# Tutorial 3: your own boundary

Every way to give a target boundary is in [boundaries.md](../boundaries.md). This tutorial covers the common ones and
designs coils for a boundary built from Fourier coefficients. Script:
[`examples/tutorial_03_own_boundary.py`](../../examples/tutorial_03_own_boundary.py).

```bash
python examples/tutorial_03_own_boundary.py
```

## From a file

```python
b1, info1 = up.load_boundary("examples/input.LandremanPaul2021_QA")      # VMEC / VMEX input: NFP, RBC, ZBS
b, info = up.load_boundary("wout_my_device.nc")                           # VMEC wout: the last closed flux surface
```

## From Fourier coefficients

R = Σ rc cos(mθ − n·nfp·φ), Z = Σ zs sin(mθ − n·nfp·φ), with n in units of nfp:

```python
import numpy as np
from xcoil.homotopy import boundary as bd

xm = np.array([0, 0, 1, 1, 1]); xn = np.array([0, 1, 0, 1, -1])
rc = np.array([1.0, 0.05, 0.15, 0.02, 0.0]); zs = np.array([0.0, -0.05, 0.15, 0.02, 0.0])
b2 = bd.from_modes(xm, xn, rc, zs, nfp=3, stellsym=True, name="my_rotating_ellipse")
up.write_vmec_input(b2, "input.my_rotating_ellipse")      # a plain VMEC input file, for VMEC / simsopt / XCoil
```

## From simsopt or DESC

- simsopt `SurfaceRZFourier`: `bd.from_simsopt(surface)`; any other simsopt surface: `up.refit_rz(surface)`. A
  simsopt JSON file (e.g. a QUASR device): `up.load_boundary("serial0755658.json")`.
- DESC: save the equilibrium as a wout (`desc.vmec.VMECIO.save(eq, "wout_desc.nc")`), then load the wout.

## Coils for it

The limits are in metres, so set them for the size of your device: this boundary has a minor radius of about 0.15 m.

```python
prob, st = up.make_problem(b3, "tutorial3", ncoils=3, order=4, length=1.6, per_coil_length=True, kappa=10.0,
                           msc=30.0, dcc=0.05, dcs=0.08, nq=64)
start, chk = up.initial_coils(prob, init="3d", size="clear")
r = up.best(up.solve_all(prob, [start])[0])
```

## Output

```
(a) VMEC input file nfp 2 61 modes
(b) R00 = 1.0 nfp 3
(c) VMEC input file max |dR| 0.0
  3d_clear             40.11 s  outer 16  mean 11.8896 %  L 4.800  kappa 4.045  MSC 15.470  d_cc 0.2547  d_cs 0.0822  v 3.3e-05
  certificate 3d_clear           CERTIFIED  mean 11.8896 %  max 31.114 %  (34.6 s)
certified True, mean |B.n|/|B| 11.890 %
```

The rotating ellipse written to a VMEC input file reads back exactly ((c), |dR| = 0). The coils are certified, but
the field error is 12 %: a rotating ellipse is not an optimized stellarator. Three short coils of order 4 cannot make
it well. More coils, a higher order and a longer budget lower the error; the boundary itself is the main limit.
