"""Documentation completeness: every public function has a docstring, every command-line option a help text, every
HConfig field, solver-budget key and XCOIL_* environment variable a description (xcoil/options.py), and
docs/reference.md is generated from the current code."""
import argparse
import dataclasses
import importlib
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_public_api_documented():
    from xcoil.options import PUBLIC_API
    missing = [f"{m}.{n}" for m, names in PUBLIC_API for n in names
               if not (getattr(importlib.import_module(m), n).__doc__ or "").strip()]
    assert not missing, f"public functions without a docstring: {missing}"


def test_cli_options_documented():
    from xcoil import cli, check
    for p in (cli.build_parser(), check.build_parser()):
        bad = [a.dest for a in p._actions if not isinstance(a, argparse._HelpAction) and not (a.help or "").strip()]
        assert not bad, f"{p.prog}: options without help: {bad}"


def test_hconfig_fields_documented():
    from xcoil.homotopy.config import HConfig
    from xcoil.options import HCONFIG_DOC
    fields = {f.name for f in dataclasses.fields(HConfig)}
    assert fields - set(HCONFIG_DOC) == set(), "undocumented HConfig fields"
    assert set(HCONFIG_DOC) - fields == set(), "stale entries"


def test_budget_keys_documented():
    from xcoil.fixed.device_lm import DEFAULTS
    from xcoil.fixed.device_lm import MAIN
    from xcoil.options import BUDGET_DOC
    assert set(DEFAULTS) | set(MAIN) <= set(BUDGET_DOC)
    src = "".join(open(os.path.join(ROOT, "xcoil", "fixed", f)).read() for f in ("device_lm.py", "user_problem.py"))
    assert [k for k in BUDGET_DOC if f'"{k}"' not in src] == [], "stale budget keys"


def test_env_vars_documented():
    from xcoil.options import ENV_DOC
    found = set()
    for dp, _, fn in os.walk(os.path.join(ROOT, "xcoil")):
        for f in fn:
            if f.endswith(".py"):
                found |= set(re.findall(r"environ(?:\.get)?[\(\[]\s*[\"'](XCOIL_[A-Z0-9_]+)", open(os.path.join(dp, f)).read()))
    assert found - set(ENV_DOC) == set(), f"undocumented environment variables: {found - set(ENV_DOC)}"


def test_reference_up_to_date():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gen_reference.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
