"""The tutorial scripts (examples/tutorial_*.py) run end to end and print what docs/tutorials says they print.
Tutorials 1, 3 and 6 run in the CI (about a minute each on a CPU); the longer ones are marked slow."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    ("01_first_coils", ["certified: True"], False),
    ("02_limits", ["ordering     ok", "with the force limit: certified True"], True),
    ("03_own_boundary", ["(c) VMEC input file", "certified True"], False),
    ("04_starts", ["admissible starts", "best:", "certified True"], True),
    ("05_performance", ["no new compilation"], True),
    ("06_export", ["physical coils: (16, 128, 3)", "base coils: 4"], False),
    ("07_outer_optimization", ["relative difference"], True),
]


@pytest.mark.parametrize("name,expect", [pytest.param(n, e, marks=[pytest.mark.slow] if s else [], id=n)
                                          for n, e, s in CASES])
def test_tutorial(name, expect, tmp_path):
    env = dict(os.environ, XCOIL_USER_DIR=str(tmp_path / "boundaries"), JAX_PLATFORMS="cpu")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "examples", f"tutorial_{name}.py")], cwd=tmp_path,
                       env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for e in expect:
        assert e in r.stdout, (e, r.stdout[-3000:])
