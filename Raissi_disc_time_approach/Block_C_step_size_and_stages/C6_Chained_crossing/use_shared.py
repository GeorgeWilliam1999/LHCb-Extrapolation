"""The path helper: copy this file into an experiment folder and import it first.

    import use_shared                      # noqa: F401  (puts _shared on the path)
    from _shared.reference import make_field, rk4_rows

It walks up from this file until it finds the directory that *contains*
`_shared/` (i.e. `Raissi_disc_time_approach/`) and puts that on `sys.path`.
Walking rather than assuming a fixed depth is deliberate: the experiments are
grouped one level deeper than they used to be (`Block_C_step_size_and_stages/
C6_Chained_crossing/` rather than `C6_Chained_crossing/`), and a fixed
`dirname(dirname(...))` silently pointed at the block folder instead of the
project root. It imports nothing itself, so it cannot reorder anything.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_root(start):
    d = start
    while True:
        if os.path.isdir(os.path.join(d, "_shared")):
            return d
        parent = os.path.dirname(d)
        if parent == d:                     # hit the filesystem root
            raise RuntimeError(
                "use_shared.py: no directory containing '_shared/' above %s" % start)
        d = parent


_ROOT = _find_root(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SHARED_ROOT = _ROOT
