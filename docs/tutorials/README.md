# Tutorials

Each tutorial has a script in [`examples/`](../../examples) that runs on a laptop CPU in the time given, and shows the
output it prints. Run them from any folder; they write their files into the current one.

| # | tutorial | script | CPU time |
|---|---|---|---|
| 1 | [First coils: solve, certify, export](01_first_coils.md) | `tutorial_01_first_coils.py` | 1 min |
| 2 | [Engineering limits and the certificate](02_limits.md) | `tutorial_02_limits.py` | 2 min |
| 3 | [Your own boundary](03_own_boundary.md) | `tutorial_03_own_boundary.py` | 2 min |
| 4 | [Starting coils, many starts, warm starts](04_starts.md) | `tutorial_04_starts.py` | 5 min |
| 5 | [Performance: compilation, precision, GPU](05_performance.md) | `tutorial_05_performance.py` | 5 min |
| 6 | [Exporting coils](06_export.md) | `tutorial_06_export.py` | 1 min |
| 7 | [XCoil inside an outer optimization](07_outer_optimization.md) | `tutorial_07_outer_optimization.py` | 5 min |

On a GPU every tutorial runs in well under a minute after compilation. Times and the last digits of the numbers vary
with the hardware.

`pytest tests/test_tutorials.py` runs tutorials 1, 3 and 6; `pytest -m slow tests/test_tutorials.py` all seven.
