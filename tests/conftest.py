"""pytest setup: x64 JAX; cross-checks against simsopt are skipped when simsopt is not installed
(pip install "xcoil[simsopt]" to run them)."""
import importlib.util

import pytest

import xcoil  # noqa: F401  (enables x64)

HAVE_SIMSOPT = importlib.util.find_spec("simsopt") is not None
SIMSOPT_TESTS = ("simsopt", "test_linking")


def pytest_collection_modifyitems(config, items):
    if HAVE_SIMSOPT:
        return
    skip = pytest.mark.skip(reason="needs simsopt (optional extra)")
    for it in items:
        if any(k in it.name for k in SIMSOPT_TESTS):
            it.add_marker(skip)
