"""XCoil: fast, differentiable, certified stellarator coil optimization in JAX.

Importing xcoil enables JAX's 64-bit mode (xcoil.config). Start with xcoil.fixed.user_problem;
see the README and docs/.
"""

from . import config  # noqa: F401  (enables x64 on import)
