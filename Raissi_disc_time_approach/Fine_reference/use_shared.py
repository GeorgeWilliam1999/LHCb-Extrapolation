"""The path helper: copy this file into an experiment folder and import it first.

    import use_shared                      # noqa: F401  (puts _shared on the path)
    from _shared.reference import make_field, rk4_rows

It only puts the parent of `_shared` (i.e. `Raissi_disc_time_approach/`) on
`sys.path`; it imports nothing itself, so it cannot reorder anything.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SHARED_ROOT = _ROOT
