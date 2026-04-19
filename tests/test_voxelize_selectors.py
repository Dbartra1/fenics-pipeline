# tests/test_voxelize_selectors.py
#
# Tests that the leg_holes and center_disk selectors in voxelize.py
# produce geometrically correct DOF selections on known grids.
#
# Run from repo root:
#   python -m pytest tests/test_voxelize_selectors.py -v

from __future__ import annotations

import math

import numpy as np
import pytest

from src.geometry.param_schema import (
    GeometryParams, FixedFaceConfig, LoadFaceConfig, LoadCaseConfig,
)
from scripts.voxelize import build_load_case


def _make_grid(nx=80, ny=80, nz=25, voxel_mm=1.0):
    """Standard tripod-sized grid config."""
    return {
        "nx": nx, "ny": ny, "nz": nz,
        "voxel_size": voxel_mm / 1000.0,
    }


def _tripod_geom():
    """GeometryParams matching tripod_mount_base_params.json."""
    return GeometryParams(
        length=80.0, width=80.0, height=25.0,
        shape="disk", diameter=80.0,
        center_hole_d=7.0, center_hole_wall_mm=5.0,
        leg_hole_d=5.0, leg_hole_radius=28.0,
        leg_hole_wall_mm=3.0, num_legs=3, first_leg_angle=90.0,
    )


def _box_geom():
    """GeometryParams matching base_part_params.json."""
    return GeometryParams(
        length=100.0, width=60.0, height=20.0,
        wall_thickness=4.0, fillet_radius=2.0,
        mounting_hole_diameter=6.0, mounting_hole_inset=10.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# leg_holes selector
# ─────────────────────────────────────────────────────────────────────────────

class TestLegHolesSelector:

    def test_produces_nonzero_fixed_dofs(self):
        """leg_holes selector must select at least some DOFs."""
        geom = _tripod_geom()
        grid = _make_grid()
        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="leg_holes",
                                  disk_radius_m=0.004),
            load=LoadFaceConfig(face="z_max", selector="full",
                                magnitude_n=5000.0),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)
        assert len(result["fixed_dofs"]) > 0

    def test_three_fold_symmetry(self):
        """
        With num_legs=3 and first_leg_angle=90, the fixed nodes should
        exhibit approximate 3-fold rotational symmetry about the part centre.

        Because the 120° rotation of a discrete grid doesn't land exactly
        on grid nodes, we verify symmetry by checking that each leg's
        cluster has approximately the same node count rather than doing
        exact coordinate matching.
        """
        geom = _tripod_geom()
        grid = _make_grid()
        h = grid["voxel_size"]
        nx, ny = grid["nx"], grid["ny"]

        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="leg_holes",
                                  disk_radius_m=0.004),
            load=LoadFaceConfig(face="z_max", selector="full",
                                magnitude_n=5000.0),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)
        fixed_dofs = result["fixed_dofs"]

        # Extract unique node indices (every node contributes 3 DOFs)
        node_ids = sorted(set(int(d) // 3 for d in fixed_dofs))
        assert len(node_ids) > 0

        # Convert to XY coords on z_min face
        coords = []
        for nid in node_ids:
            ix = nid % (nx + 1)
            iy = (nid // (nx + 1)) % (ny + 1)
            coords.append((ix * h, iy * h))

        # Compute the 3 expected leg centres
        cx_m, cy_m = 0.040, 0.040
        leg_r_m = 0.028
        leg_centres = []
        for i in range(3):
            theta = math.radians(90.0 + i * 120.0)
            leg_centres.append((cx_m + leg_r_m * math.cos(theta),
                                cy_m + leg_r_m * math.sin(theta)))

        # Assign each node to its nearest leg centre
        counts = [0, 0, 0]
        for x, y in coords:
            dists = [math.sqrt((x - lx)**2 + (y - ly)**2)
                     for lx, ly in leg_centres]
            counts[dists.index(min(dists))] += 1

        # All three legs should have similar node counts (within 20%)
        avg = sum(counts) / 3.0
        for i, c in enumerate(counts):
            assert abs(c - avg) / avg < 0.20, (
                f"Leg {i} has {c} nodes vs average {avg:.0f} — "
                f"counts={counts}, symmetry broken"
            )

    def test_nodes_are_on_bottom_face(self):
        """All fixed DOFs from leg_holes on z_min must be iz=0 nodes."""
        geom = _tripod_geom()
        grid = _make_grid()
        nx, ny = grid["nx"], grid["ny"]

        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="leg_holes",
                                  disk_radius_m=0.004),
            load=LoadFaceConfig(face="z_max", selector="full",
                                magnitude_n=5000.0),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)

        for dof in result["fixed_dofs"]:
            nid = int(dof) // 3
            iz = nid // ((nx + 1) * (ny + 1))
            assert iz == 0, f"node {nid} has iz={iz}, expected 0 (z_min face)"


# ─────────────────────────────────────────────────────────────────────────────
# center_disk selector
# ─────────────────────────────────────────────────────────────────────────────

class TestCenterDiskSelector:

    def test_produces_nonzero_load_dofs(self):
        """center_disk must select at least some DOFs."""
        geom = _tripod_geom()
        grid = _make_grid()
        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="full"),
            load=LoadFaceConfig(face="z_max", selector="center_disk",
                                magnitude_n=5000.0, disk_radius_m=0.004),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)
        assert len(result["load_dofs"]) > 0

    def test_fewer_nodes_than_full_face(self):
        """center_disk must load fewer nodes than full face."""
        geom = _tripod_geom()
        grid = _make_grid()

        lc_full = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="full"),
            load=LoadFaceConfig(face="z_max", selector="full",
                                magnitude_n=5000.0),
        )
        lc_disk = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="full"),
            load=LoadFaceConfig(face="z_max", selector="center_disk",
                                magnitude_n=5000.0, disk_radius_m=0.004),
        )
        r_full = build_load_case(geom, None, grid, load_case_config=lc_full)
        r_disk = build_load_case(geom, None, grid, load_case_config=lc_disk)
        assert len(r_disk["load_dofs"]) < len(r_full["load_dofs"])

    def test_load_nodes_near_centre(self):
        """All loaded nodes must be within disk_radius_m of part centre."""
        geom = _tripod_geom()
        grid = _make_grid()
        h = grid["voxel_size"]
        nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
        disk_r = 0.004

        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="full"),
            load=LoadFaceConfig(face="z_max", selector="center_disk",
                                magnitude_n=5000.0, disk_radius_m=disk_r),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)

        cx_m, cy_m = 0.040, 0.040
        for dof in result["load_dofs"]:
            nid = int(dof) // 3
            ix = nid % (nx + 1)
            iy = (nid // (nx + 1)) % (ny + 1)
            x, y = ix * h, iy * h
            dist = math.sqrt((x - cx_m)**2 + (y - cy_m)**2)
            assert dist < disk_r + h, (
                f"node ({ix},{iy}) at dist={dist:.4f} exceeds "
                f"disk_radius_m={disk_r} + voxel tolerance"
            )

    def test_force_magnitude_preserved(self):
        """Total force across all loaded DOFs must equal magnitude_n."""
        geom = _tripod_geom()
        grid = _make_grid()
        mag = 5000.0

        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="full"),
            load=LoadFaceConfig(face="z_max", selector="center_disk",
                                direction=[0.0, 0.0, -1.0],
                                magnitude_n=mag, disk_radius_m=0.004),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)
        total = sum(result["load_vals"])
        assert abs(total - (-mag)) < 1e-6, (
            f"total force = {total}, expected {-mag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward compatibility: corners + full still work
# ─────────────────────────────────────────────────────────────────────────────

class TestBackwardCompat:

    def test_corners_full_box_part(self):
        """base_part-style corners/full config still works with geom threading."""
        geom = _box_geom()
        grid = _make_grid(nx=100, ny=60, nz=20)
        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="z_min", selector="corners",
                                  inset_m=0.010, disk_radius_m=0.005),
            load=LoadFaceConfig(face="z_max", selector="full",
                                magnitude_n=10000.0),
        )
        result = build_load_case(geom, None, grid, load_case_config=lc)
        assert len(result["fixed_dofs"]) > 0
        assert len(result["load_dofs"]) > 0
        assert len(result["load_vals"]) == len(result["load_dofs"])