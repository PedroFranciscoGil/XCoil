#!/usr/bin/env python
"""The xcoil-coils command as a script: python examples/xcoil_coils.py [options] (= xcoil-coils [options]).
See docs/cli.md and docs/tutorials/01_first_coils.md."""
import sys

from xcoil.cli import main

if __name__ == "__main__":
    sys.exit(main())
