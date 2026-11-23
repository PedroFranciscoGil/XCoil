"""Precision strategy.

FP64 is the numerical reference and the default. FP32 and mixed are opt-in performance modes
evaluated only after FP64 end-to-end behaviour is stable. Importing this module enables x64 in
JAX so every computation runs in true double precision by default.
"""
from __future__ import annotations

import os
import jax

# precision modes: "fp64" (reference/default), "fp32", "mixed".
# Controlled by env var XCOIL_PRECISION so GPU timing can select FP32 (consumer-GPU FP64 is
# ~1/32 throughput; FP32/mixed are the performance modes).
PRECISION = os.environ.get("XCOIL_PRECISION", "fp64")

if PRECISION == "fp64":
    jax.config.update("jax_enable_x64", True)

# Persistent compilation cache: compiled programs are stored on disk and reused by later processes with the same
# programs, JAX/XLA version, flags and GPU, including programs that compile in 0.1-1 s (JAX caches only those above
# 1 s by default). XCOIL_COMPILE_CACHE: the directory, or "off"; unset, a directory already given to JAX
# (JAX_COMPILATION_CACHE_DIR) is kept, else ~/.cache/xcoil/jax.
COMPILE_CACHE = os.environ.get("XCOIL_COMPILE_CACHE") or jax.config.jax_compilation_cache_dir or \
    os.path.join(os.path.expanduser("~"), ".cache", "xcoil", "jax")

if COMPILE_CACHE.lower() in ("off", "0"):
    jax.config.update("jax_enable_compilation_cache", False)
else:
    jax.config.update("jax_compilation_cache_dir", COMPILE_CACHE)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.1)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

_REAL = {"fp64": "float64", "fp32": "float32", "mixed": "float32"}


def real_dtype(mode: str | None = None) -> str:
    """Dtype for geometry/Biot-Savart hot-path arrays under the given precision mode."""
    return _REAL[mode or PRECISION]
