"""The v8r1 magnetic field map of LHCb, and its differentiable twin.

The map is stored on a regular grid and interpolated linearly in each of the
three directions. The interpolated field is continuous, but its derivative
jumps at every cell face.

Ported from the frozen code, with the arithmetic unchanged and in the same
order, so that the field values are identical to the last bit:

    FieldMap                from  _shared/field_v8r1.py   (FieldV8R1)
    DifferentiableFieldMap  from  _shared/field_torch.py  (FieldTorch)
    hash_of_field_map_file  from  _shared/reference.py    (field_md5)
    corners_of_field_map    from  _shared/reference.py    (field_bounds)

The map files are read from CVMFS and are not part of the repository.
Gates: tests/test_field_map_matches_the_frozen_code.py.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from rkpinn.registry import register

FIELD_MAP_FILES = {
    "v8r1_down": "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.down.bin",
    "v8r1_up": "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.up.bin",
}


class FieldMap:
    """The field (Bx, By, Bz) in tesla at any point, from one map file."""

    def __init__(self, path: str):
        self.path = path
        raw = np.fromfile(path, dtype=np.float32)
        inverse_cell_size = raw[0:3]
        number_of_points = raw[4:8].view(np.int32)[0:3]
        lowest_corner = raw[8:11]
        nx, ny, nz = (int(number_of_points[0]), int(number_of_points[1]),
                      int(number_of_points[2]))
        values = raw[12:12 + nx * ny * nz * 4].reshape(-1, 4)
        self.inverse_cell_size = inverse_cell_size.astype(np.float64)
        self.number_of_points = number_of_points.astype(np.int64)
        self.lowest_corner = lowest_corner.astype(np.float64)
        # x runs fastest in the file: index = ix + nx * (iy + ny * iz)
        self.field_x = values[:, 0].reshape(nz, ny, nx).astype(np.float64)
        self.field_y = values[:, 1].reshape(nz, ny, nx).astype(np.float64)
        self.field_z = values[:, 2].reshape(nz, ny, nx).astype(np.float64)
        # The unit is detected from the peak of |By|: near 1 the file is in
        # tesla; near 1e-3 it is in Gaudi units, where 1 T = 1e-3.
        peak = float(np.nanmax(np.abs(self.field_y)))
        if 0.1 < peak < 10:
            self.scale_to_tesla = 1.0
        elif 1e-4 < peak < 1e-2:
            self.scale_to_tesla = 1.0 / 1e-3
        else:
            raise ValueError("unrecognised field units, peak |By| = %r" % peak)
        self.peak_of_field_y_in_file = peak

    def description(self) -> str:
        cell_size = 1.0 / self.inverse_cell_size
        highest_corner = self.lowest_corner + (self.number_of_points - 1) * cell_size
        return ("points %s, cell size %s mm, lowest corner %s mm, highest corner %s mm, "
                "peak |By| in the file %.5g, scale to tesla %g"
                % (tuple(self.number_of_points), tuple(np.round(cell_size, 1)),
                   tuple(np.round(self.lowest_corner, 1)),
                   tuple(np.round(highest_corner, 1)),
                   self.peak_of_field_y_in_file, self.scale_to_tesla))

    def __call__(self, x, y, z):
        """(Bx, By, Bz) in tesla at the points (x, y, z) in millimetres."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        fx = (x - self.lowest_corner[0]) * self.inverse_cell_size[0]
        fy = (y - self.lowest_corner[1]) * self.inverse_cell_size[1]
        fz = (z - self.lowest_corner[2]) * self.inverse_cell_size[2]
        ix = np.clip(fx.astype(np.int64), 0, self.number_of_points[0] - 2)
        iy = np.clip(fy.astype(np.int64), 0, self.number_of_points[1] - 2)
        iz = np.clip(fz.astype(np.int64), 0, self.number_of_points[2] - 2)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        tz = np.clip(fz - iz, 0.0, 1.0)

        def interpolate(grid):
            c000 = grid[iz, iy, ix]
            c100 = grid[iz, iy, ix + 1]
            c010 = grid[iz, iy + 1, ix]
            c110 = grid[iz, iy + 1, ix + 1]
            c001 = grid[iz + 1, iy, ix]
            c101 = grid[iz + 1, iy, ix + 1]
            c011 = grid[iz + 1, iy + 1, ix]
            c111 = grid[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1 - tx) + c100 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c11 = c011 * (1 - tx) + c111 * tx
            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty
            return (c0 * (1 - tz) + c1 * tz) * self.scale_to_tesla

        return (interpolate(self.field_x), interpolate(self.field_y),
                interpolate(self.field_z))


class DifferentiableFieldMap(torch.nn.Module):
    """The same field on torch tensors, in double precision, differentiable
    with respect to the position.

    The grid is taken from a `FieldMap`, so the two share their numbers. The
    gradient exists everywhere except on the cell faces, where the field has a
    kink and torch returns the value from one side.
    """

    def __init__(self, field_map: FieldMap):
        super().__init__()
        self.field_map = field_map
        self.register_buffer("inverse_cell_size", torch.tensor(field_map.inverse_cell_size))
        self.register_buffer("lowest_corner", torch.tensor(field_map.lowest_corner))
        self.register_buffer("field_x", torch.tensor(field_map.field_x * field_map.scale_to_tesla))
        self.register_buffer("field_y", torch.tensor(field_map.field_y * field_map.scale_to_tesla))
        self.register_buffer("field_z", torch.tensor(field_map.field_z * field_map.scale_to_tesla))
        self.number_of_points = tuple(int(n) for n in field_map.number_of_points)

    def forward(self, x, y, z):
        fx = (x - self.lowest_corner[0]) * self.inverse_cell_size[0]
        fy = (y - self.lowest_corner[1]) * self.inverse_cell_size[1]
        fz = (z - self.lowest_corner[2]) * self.inverse_cell_size[2]
        ix = fx.detach().floor().long().clamp(0, self.number_of_points[0] - 2)
        iy = fy.detach().floor().long().clamp(0, self.number_of_points[1] - 2)
        iz = fz.detach().floor().long().clamp(0, self.number_of_points[2] - 2)
        tx = (fx - ix).clamp(0.0, 1.0)
        ty = (fy - iy).clamp(0.0, 1.0)
        tz = (fz - iz).clamp(0.0, 1.0)

        def interpolate(grid):
            c000 = grid[iz, iy, ix]
            c100 = grid[iz, iy, ix + 1]
            c010 = grid[iz, iy + 1, ix]
            c110 = grid[iz, iy + 1, ix + 1]
            c001 = grid[iz + 1, iy, ix]
            c101 = grid[iz + 1, iy, ix + 1]
            c011 = grid[iz + 1, iy + 1, ix]
            c111 = grid[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1 - tx) + c100 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c11 = c011 * (1 - tx) + c111 * tx
            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty
            return c0 * (1 - tz) + c1 * tz

        return (interpolate(self.field_x), interpolate(self.field_y),
                interpolate(self.field_z))


_loaded: dict[str, FieldMap] = {}


def load_field_map(name: str) -> FieldMap:
    """The field map registered under `name`, read once per process."""
    if name not in FIELD_MAP_FILES:
        raise KeyError("no field map is named %r; the names are %s"
                       % (name, ", ".join(sorted(FIELD_MAP_FILES))))
    if name not in _loaded:
        _loaded[name] = FieldMap(FIELD_MAP_FILES[name])
    return _loaded[name]


@register("field_map", "v8r1_up")
def v8r1_up() -> FieldMap:
    """The v8r1 map with the magnet polarity up."""
    return load_field_map("v8r1_up")


@register("field_map", "v8r1_down")
def v8r1_down() -> FieldMap:
    """The v8r1 map with the magnet polarity down."""
    return load_field_map("v8r1_down")


def hash_of_field_map_file(name: str, chunk_bytes: int = 1 << 20) -> str:
    """The md5 hash of the map file, recorded with every run."""
    digest = hashlib.md5()
    with open(FIELD_MAP_FILES[name], "rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def corners_of_field_map(field_map: FieldMap):
    """The lowest and highest corners of the map in millimetres, as two 3-vectors."""
    lowest = np.array([field_map.lowest_corner[i] for i in range(3)], dtype=np.float64)
    highest = lowest + ((np.asarray(field_map.number_of_points, dtype=np.float64) - 1.0)
                        / np.asarray(field_map.inverse_cell_size))
    return lowest, highest
