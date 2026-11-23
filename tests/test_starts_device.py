"""Start checks on the default JAX device (starts._linked, _min_coil_coil) agree with the numpy loops
(XCOIL_TABLES=numpy), on the qa_auglag starts and on linked / crossing pairs."""
from __future__ import annotations

import numpy as np
import pytest

import xcoil  # noqa: F401
import problems as pd_
from xcoil.fixed import starts as st


@pytest.fixture(scope="module")
def pts():
    prob = pd_.problem_for("qa_auglag")
    s = pd_.starts_for(prob, families=("offset_3d", "offset_planar"))
    return prob, [np.asarray(x.points) for x in s]


def _both(monkeypatch, f):
    monkeypatch.setenv("XCOIL_TABLES", "jax"); a = f()
    monkeypatch.setenv("XCOIL_TABLES", "numpy"); b = f()
    return a, b


def test_min_coil_coil_matches(pts, monkeypatch):
    prob, P = pts
    nfp, ss = prob.b.nfp, True
    for p in P:
        a, b = _both(monkeypatch, lambda: st._min_coil_coil(p, nfp, ss))
        assert abs(a - b) <= 1e-12 * max(1.0, b)


def test_linked_matches(pts, monkeypatch):
    prob, P = pts
    nfp, ss = prob.b.nfp, True
    for p in P:
        for scale in (1.0, 3.0):                                  # unlinked starts, and grown until they link
            q = p.copy(); c = q.mean(1, keepdims=True); q = c + scale * (q - c)
            a, b = _both(monkeypatch, lambda: st._linked(q, nfp, ss, n=256))
            assert a == b
