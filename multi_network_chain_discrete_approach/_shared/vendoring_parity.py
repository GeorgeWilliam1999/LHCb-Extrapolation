#!/usr/bin/env python
"""The gate on the vendored field loader (A0.3).

`field_v8r1.py` in this package is a copy of the archive module that every
earlier script reached for with a `sys.path.insert`. A copy is only safe if it
is provably the same thing, so this script:

  1. imports the field loader from the archive path AND from this package;
  2. evaluates both on 200,000 random points drawn inside the map's own
     bounding box, and requires the largest absolute difference to be exactly
     zero (not "small": the code is identical, so the answer must be bit-equal);
  3. records the md5 of the field-map binary itself, which must match the value
     stamped into the training-set metadata,
     af284c6954d2273c637a5e766b82b58e.

Output: results/vendoring_parity.json
Run:  PYTHONNOUSERSITE=1 /data/bfys/gscriven/conda/envs/TE/bin/python vendoring_parity.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from _shared.field_v8r1 import FieldV8R1 as VendoredField          # noqa: E402
from _shared.reference import field_md5, field_bounds              # noqa: E402

ARCHIVE_CORE = ("/data/bfys/gscriven/Track_Extrapolation_work_archive/"
                "track-extrapolation-pinn/core")
ARCHIVE_COMMIT = "1faa97e085918082fd8ff15e60aa7932bb588786"
EXPECTED_MD5 = "af284c6954d2273c637a5e766b82b58e"
N_POINTS = 200_000
SEED = 20260905


def main():
    sys.path.insert(0, ARCHIVE_CORE)
    import field_v8r1 as archive_module          # noqa: E402  the original
    assert os.path.dirname(os.path.abspath(archive_module.__file__)) == ARCHIVE_CORE, \
        "the archive import resolved somewhere else: %s" % archive_module.__file__

    archive = archive_module.FieldV8R1()
    vendored = VendoredField()

    lo, hi = field_bounds(vendored)
    rng = np.random.default_rng(SEED)
    pts = lo + rng.random((N_POINTS, 3)) * (hi - lo)      # strictly inside the map
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    Ba = np.stack(archive(x, y, z), axis=1)
    Bv = np.stack(vendored(x, y, z), axis=1)
    worst = float(np.abs(Ba - Bv).max())
    identical = bool((Ba == Bv).all())

    md5 = field_md5("down")
    out = {
        "checked": "2026-09-05",
        "origin_module": os.path.join(ARCHIVE_CORE, "field_v8r1.py"),
        "origin_commit": ARCHIVE_COMMIT,
        "vendored_module": os.path.join(HERE, "field_v8r1.py"),
        "n_points": N_POINTS,
        "point_seed": SEED,
        "sampling_box_mm": {"lo": lo.tolist(), "hi": hi.tolist()},
        "max_abs_diff_T": worst,
        "bitwise_identical": identical,
        "field_map_file": archive_module.V8R1_DOWN,
        "field_map_md5": md5,
        "field_map_md5_expected": EXPECTED_MD5,
        "md5_matches": bool(md5 == EXPECTED_MD5),
        "map_info": vendored.info(),
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "vendoring_parity.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))

    assert worst == 0.0, "vendored field differs from the archive by %g T" % worst
    assert identical, "vendored field is not bit-identical to the archive"
    assert md5 == EXPECTED_MD5, "the field map on CVMFS is not the expected file"
    print("\nGATE PASS: vendored loader is bit-identical on %d points; "
          "map md5 %s" % (N_POINTS, md5))


if __name__ == "__main__":
    main()
