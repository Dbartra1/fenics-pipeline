# tests/test_bolt_seat.py
#
# Tests for the BoltSeatRegion primitive.
#
# Covers:
#   - Schema validation (valid / invalid configs)
#   - voxelize_domain() correctly applies seat collars at entry/exit only
#   - Through-hole voids span the full axis
#   - Middle of bolt path is neither forced solid nor forced void
#   - entry_seat / exit_seat toggles produce asymmetric masks
#
# Run from repo root:
#   python -m pytest tests/test_bolt_seat.py -v

from __future__ import annotations

import numpy as np
import pytest

from src.geometry.param_schema import (
    GeometryParams, PipelineParams, BoltSeatRegion,
)
from scripts.voxelize import voxelize_domain


# ─────────────────────────────────────────────────────────────────────────────
# Schema validation
# ─────────────────────────────────────────────────────────────────────────────

class TestBoltSeatSchema:

    def test_valid_two_sided(self):
        b = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.002,
            wall_radius_m=0.004,
            seat_depth_m=0.005,
        )
        b.validate()  # defaults: entry=True, exit=True

    def test_valid_one_sided(self):
        b = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.002,
            wall_radius_m=0.004,
            seat_depth_m=0.005,
            entry_seat=False,
            exit_seat=True,
        )
        b.validate()

    def test_invalid_no_seats(self):
        """A bolt with no seats at all would have no anchor — reject."""
        b = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.002,
            wall_radius_m=0.004,
            seat_depth_m=0.005,
            entry_seat=False,
            exit_seat=False,
        )
        with pytest.raises(AssertionError, match="at least one of"):
            b.validate()

    def test_invalid_axis(self):
        b = BoltSeatRegion(
            type="bolt_seat_diagonal",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.002,
            wall_radius_m=0.004,
            seat_depth_m=0.005,
        )
        with pytest.raises(AssertionError, match="type must be one of"):
            b.validate()

    def test_invalid_seat_depth(self):
        b = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.002,
            wall_radius_m=0.004,
            seat_depth_m=0.0,
        )
        with pytest.raises(AssertionError, match="seat_depth_m"):
            b.validate()

    def test_invalid_wall_smaller_than_void(self):
        b = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010]],
            void_radius_m=0.004,
            wall_radius_m=0.002,
            seat_depth_m=0.005,
        )
        with pytest.raises(AssertionError, match=">= void_radius_m"):
            b.validate()

    def test_parse_from_dict(self):
        """PipelineParams._from_raw should parse bolt_seats list."""
        raw = {
            "part_name": "test",
            "geometry": {"length": 80.0, "width": 80.0, "height": 80.0},
            "mesh_hints": {
                "target_element_size": 5.0,
                "opt_domain_element_size_mm": 2.5,
            },
            "load_hints": {"primary_face": "top", "load_magnitude_n": 1000.0},
            "export": {"stl_output_dir": "/tmp", "stl_ascii": False},
            "bolt_seats": [
                {
                    "type": "bolt_seat_x",
                    "centers_m": [[0.010, 0.010], [0.050, 0.050]],
                    "void_radius_m": 0.002,
                    "wall_radius_m": 0.004,
                    "seat_depth_m":  0.005,
                },
            ],
        }
        p = PipelineParams.from_dict(raw)
        p.validate()
        assert len(p.bolt_seats) == 1
        assert p.bolt_seats[0].type == "bolt_seat_x"
        assert p.bolt_seats[0].entry_seat is True   # default
        assert p.bolt_seats[0].exit_seat is True    # default


# ─────────────────────────────────────────────────────────────────────────────
# Voxelization behaviour
# ─────────────────────────────────────────────────────────────────────────────

def _grid(nx=70, ny=60, nz=80, voxel_mm=1.0):
    return {"nx": nx, "ny": ny, "nz": nz, "voxel_size": voxel_mm / 1000.0}


def _box_geom(L=70.0, W=60.0, H=80.0):
    return GeometryParams(length=L, width=W, height=H)


class TestBoltSeatVoxelization:

    def test_through_hole_spans_full_axis(self):
        """Void core must exist at every X slice along the bolt path."""
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.030, 0.040]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
        )
        nondesign, void_mask = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        # For every X slice, there must be SOME void voxel at the bolt position
        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        for ix in range(nx):
            slab = void_mask[:, :, ix]   # (nz, ny) because mask is (nz, ny, nx)
            assert slab.sum() > 0, (
                f"Bolt through-hole missing at ix={ix} — "
                f"void mask should span full X axis"
            )

    def test_collar_only_near_faces(self):
        """
        Nondesign collar must exist near x=0 and x=xmax but NOT in the middle.
        """
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.030, 0.040]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,   # 5mm collar at each face
        )
        nondesign, _ = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )

        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        # x=2 (inside entry collar, 0 < 2 < 5) should have nondesign around bolt
        assert nondesign[:, :, 2].sum() > 0, "entry collar missing"
        # x=67 (inside exit collar, 65 < 67 < 70) should have nondesign around bolt
        assert nondesign[:, :, 67].sum() > 0, "exit collar missing"
        # x=35 (middle of bolt) should have NO nondesign ring around the bolt
        assert nondesign[:, :, 35].sum() == 0, (
            "middle of bolt path should be free; found forced-solid voxels"
        )

    def test_entry_only_skips_exit_collar(self):
        """entry_seat=True, exit_seat=False → no collar at x_max."""
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.030, 0.040]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
            entry_seat=True,
            exit_seat=False,
        )
        nondesign, _ = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        # Entry collar at x=2 should exist
        assert nondesign[:, :, 2].sum() > 0
        # Exit collar at x=67 should NOT exist
        assert nondesign[:, :, 67].sum() == 0

    def test_exit_only_skips_entry_collar(self):
        """entry_seat=False, exit_seat=True → no collar at x_min."""
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.030, 0.040]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
            entry_seat=False,
            exit_seat=True,
        )
        nondesign, _ = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        assert nondesign[:, :, 2].sum() == 0
        assert nondesign[:, :, 67].sum() > 0

    def test_voxel_count_reasonable(self):
        """
        Bolt seat should produce far fewer nondesign voxels than the
        equivalent full-length cylinder_x NondesignRegion.

        4 bolts, 3mm void + 5mm collar, 5mm seat depth on each side, on
        a 70mm-long X axis:
            bolt_seat:   4 × π × 5² × (5+5) ≈ 3,140 nondesign voxels
            cylinder_x:  4 × π × 5² × 70    ≈ 22,000 nondesign voxels
        Ratio should be ~14%.
        """
        grid = _grid()
        geom = _box_geom()
        seats = [BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.010, 0.010], [0.050, 0.010],
                       [0.010, 0.070], [0.050, 0.070]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
        )]
        nondesign, _ = voxelize_domain(geom, grid, bolt_seats=seats)

        # Very loose bound: nondesign count should be in the right order of
        # magnitude (thousands, not tens of thousands)
        n_nd = int(nondesign.sum())
        assert 1_000 < n_nd < 10_000, (
            f"nondesign voxel count {n_nd} outside expected range "
            f"(~3,000 for 4 bolt seats with 5mm collar on each face)"
        )

    def test_bolt_seat_y(self):
        """Bolt along Y axis: collars at y=0 and y=ymax."""
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_y",
            centers_m=[[0.030, 0.040]],   # (x, z) for Y-axis bolt
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
        )
        nondesign, void_mask = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        # Void must span full Y
        for iy in range(grid["ny"]):
            assert void_mask[:, iy, :].sum() > 0, f"void missing at iy={iy}"
        # Collars at iy=2 and iy=57, not iy=30
        assert nondesign[:, 2, :].sum() > 0
        assert nondesign[:, 57, :].sum() > 0
        assert nondesign[:, 30, :].sum() == 0

    def test_bolt_seat_z(self):
        """Bolt along Z axis: collars at z=0 and z=zmax."""
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_z",
            centers_m=[[0.030, 0.040]],   # (x, y) for Z-axis bolt
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
        )
        nondesign, void_mask = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        for iz in range(grid["nz"]):
            assert void_mask[iz, :, :].sum() > 0, f"void missing at iz={iz}"
        assert nondesign[2, :, :].sum() > 0
        assert nondesign[77, :, :].sum() > 0
        assert nondesign[40, :, :].sum() == 0

    def test_void_takes_priority_over_collar(self):
        """
        Where the through-hole void overlaps the collar region, void wins.
        Concretely: at x=2 with r<void_radius, the voxel should be void,
        not nondesign.
        """
        grid = _grid()
        geom = _box_geom()
        seat = BoltSeatRegion(
            type="bolt_seat_x",
            centers_m=[[0.030, 0.040]],
            void_radius_m=0.003,
            wall_radius_m=0.005,
            seat_depth_m=0.005,
        )
        nondesign, void_mask = voxelize_domain(
            geom, grid, bolt_seats=[seat],
        )
        # In the entry collar slice at x=2, voxels right on the bolt axis
        # (y=30, z=40) should be void, not nondesign
        # (remember array axes are [z, y, x])
        iy_bolt = 30
        iz_bolt = 40
        assert void_mask[iz_bolt, iy_bolt, 2] == 1
        assert nondesign[iz_bolt, iy_bolt, 2] == 0