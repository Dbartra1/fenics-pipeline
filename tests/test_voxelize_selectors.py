# tests/test_voxelize_selectors.py
#
# Tests that the load and fixed selectors in voxelize.py produce
# geometrically correct DOF selections on known grids.
#
# Covers: leg_holes, center_disk, bolt_pattern, corners/full (compat)
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


def _motor_mount_geom():
    """GeometryParams matching motor_mount_params.json."""
    return GeometryParams(
        length=70.0, width=60.0, height=80.0,
        wall_hole_diameter=6.0, wall_hole_inset=10.0,
        motor_hole_diameter=4.0, motor_hole_pitch=31.0,
        motor_center_y=30.0, motor_center_z=40.0,
    )


def _motor_mount_grid():
    """1mm voxel grid for the motor mount (70x60x80mm)."""
    return _make_grid(nx=70, ny=60, nz=80)


# NEMA-17 bolt centers in face-local (Y, Z) coords for the x_max face.
# 31mm pitch, center at Y=30mm / Z=40mm — standard NEMA-17 pattern.
_NEMA_CENTERS = [
    [0.0145, 0.0245],
    [0.0455, 0.0245],
    [0.0145, 0.0555],
    [0.0455, 0.0555],
]
_NEMA_DISK_R = 0.007   # matches wall_radius_m of bolt_seat collars


def _decode_xmax_node(nid: int, nx: int, ny: int, h: float):
    """
    Invert node_idx for a node known to be on the x_max face (ix = nx).

    node_idx = ix + iy*(nx+1) + iz*(nx+1)*(ny+1)
    On x_max:  nid = nx + iy*(nx+1) + iz*(nx+1)*(ny+1)
               nid - nx = (nx+1) * (iy + iz*(ny+1))

    So:
        q  = (nid - nx) // (nx+1)   →  iy + iz*(ny+1)
        iy = q % (ny+1)
        iz = q // (ny+1)

    Returns (ca, cb) = (Y, Z) face-local coordinates in metres,
    matching the convention of _face_node_coords for x_max.
    """
    q  = (nid - nx) // (nx + 1)
    iy = q % (ny + 1)
    iz = q // (ny + 1)
    return iy * h, iz * h   # ca=Y, cb=Z


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
# bolt_pattern selector
# ─────────────────────────────────────────────────────────────────────────────

class TestBoltPatternSelector:

    def _make_lc(self, disk_r=_NEMA_DISK_R, magnitude_n=5000.0):
        """Helper: motor-mount load case with bolt_pattern on x_max."""
        return LoadCaseConfig(
            fixed=FixedFaceConfig(
                face="x_min", selector="corners",
                inset_m=0.010, disk_radius_m=0.006,
            ),
            load=LoadFaceConfig(
                face="x_max",
                selector="bolt_pattern",
                bolt_centers_m=_NEMA_CENTERS,
                disk_radius_m=disk_r,
                direction=[0.0, 0.0, -1.0],
                magnitude_n=magnitude_n,
            ),
        )

    def test_produces_nonzero_load_dofs(self):
        """bolt_pattern must select at least some load DOFs."""
        result = build_load_case(
            _motor_mount_geom(), None, _motor_mount_grid(),
            load_case_config=self._make_lc(),
        )
        assert len(result["load_dofs"]) > 0

    def test_far_fewer_dofs_than_full_face(self):
        """
        bolt_pattern at 4 NEMA positions selects ~616 nodes vs ~4941 for full face.
        Assert bolt_pattern is < 20% of full-face DOF count.
        """
        geom = _motor_mount_geom()
        grid = _motor_mount_grid()

        lc_full = LoadCaseConfig(
            fixed=FixedFaceConfig(face="x_min", selector="corners",
                                  inset_m=0.010, disk_radius_m=0.006),
            load=LoadFaceConfig(face="x_max", selector="full",
                                magnitude_n=5000.0),
        )
        r_full = build_load_case(geom, None, grid, load_case_config=lc_full)
        r_bolt = build_load_case(geom, None, grid, load_case_config=self._make_lc())

        n_full = len(r_full["load_dofs"])
        n_bolt = len(r_bolt["load_dofs"])
        assert n_bolt < n_full * 0.20, (
            f"bolt_pattern DOF count {n_bolt} should be < 20% of full-face "
            f"count {n_full} — check bolt_centers_m and disk_radius_m"
        )

    def test_load_dof_count_in_expected_range(self):
        """
        4 bolts × π × (7mm)² ≈ 616 nodes at 1mm voxel.
        Assert within ±30% of that theoretical value.
        """
        result = build_load_case(
            _motor_mount_geom(), None, _motor_mount_grid(),
            load_case_config=self._make_lc(),
        )
        # Direction is pure -Z so each selected node contributes exactly 1 DOF.
        n_nodes  = len(result["load_dofs"])
        expected = int(4 * math.pi * (_NEMA_DISK_R * 1000) ** 2)  # ≈ 616
        assert expected * 0.70 < n_nodes < expected * 1.30, (
            f"bolt_pattern node count {n_nodes} outside ±30% of "
            f"theoretical {expected} — check disk_radius_m={_NEMA_DISK_R}"
        )

    def test_loaded_nodes_cluster_near_bolt_centers(self):
        """
        Every loaded node must be within disk_radius_m (+1 voxel tolerance)
        of at least one declared bolt center. No stray nodes elsewhere.

        Node index inversion for x_max face (ix = nx fixed):
            node_idx = nx + iy*(nx+1) + iz*(nx+1)*(ny+1)
            → (nid - nx) = (nx+1) * (iy + iz*(ny+1))
            → q  = (nid - nx) // (nx+1)
            → iy = q % (ny+1),  iz = q // (ny+1)
            → ca = Y = iy*h,    cb = Z = iz*h
        """
        geom = _motor_mount_geom()
        grid = _motor_mount_grid()
        h    = grid["voxel_size"]
        nx, ny = grid["nx"], grid["ny"]

        result = build_load_case(geom, None, grid,
                                 load_case_config=self._make_lc())

        for dof in result["load_dofs"]:
            nid     = int(dof) // 3
            ca, cb  = _decode_xmax_node(nid, nx, ny, h)
            min_dist = min(
                math.sqrt((ca - c[0]) ** 2 + (cb - c[1]) ** 2)
                for c in _NEMA_CENTERS
            )
            assert min_dist < _NEMA_DISK_R + h, (
                f"node at (Y={ca:.4f}m, Z={cb:.4f}m) is {min_dist:.4f}m from "
                f"nearest bolt center — exceeds disk_radius_m={_NEMA_DISK_R} "
                f"+ voxel tolerance {h}"
            )

    def test_force_magnitude_preserved(self):
        """Total load across all DOFs must equal magnitude_n exactly."""
        mag    = 5000.0
        result = build_load_case(
            _motor_mount_geom(), None, _motor_mount_grid(),
            load_case_config=self._make_lc(magnitude_n=mag),
        )
        total = sum(result["load_vals"])
        assert abs(total - (-mag)) < 1e-6, (
            f"total force = {total:.6f}N, expected {-mag}N. "
            f"Force not conserved across {len(result['load_dofs'])} DOFs."
        )

    def test_fixed_dof_count_in_expected_range(self):
        """
        corners selector on x_min is unchanged by the load selector change.
        4 corner disks × π × (6mm)² × 3 DOF/node ≈ 1356 fixed DOFs.
        """
        result = build_load_case(
            _motor_mount_geom(), None, _motor_mount_grid(),
            load_case_config=self._make_lc(),
        )
        n_fixed = len(result["fixed_dofs"])
        assert 800 < n_fixed < 2000, (
            f"fixed DOF count {n_fixed} outside expected range ~1356 "
            f"(4 corner disks, 6mm radius, 3 DOFs/node)"
        )

    def test_schema_validates_well_formed_bolt_pattern(self):
        """LoadFaceConfig.validate() must not raise for a correct bolt_pattern."""
        cfg = LoadFaceConfig(
            face="x_max",
            selector="bolt_pattern",
            bolt_centers_m=_NEMA_CENTERS,
            disk_radius_m=_NEMA_DISK_R,
            direction=[0.0, 0.0, -1.0],
            magnitude_n=5000.0,
        )
        cfg.validate()  # must not raise

    def test_schema_rejects_bolt_pattern_without_centers(self):
        """bolt_pattern with no bolt_centers_m must fail validation."""
        cfg = LoadFaceConfig(
            face="x_max",
            selector="bolt_pattern",
            disk_radius_m=_NEMA_DISK_R,
            direction=[0.0, 0.0, -1.0],
            magnitude_n=5000.0,
            # bolt_centers_m intentionally omitted
        )
        with pytest.raises(AssertionError, match="bolt_centers_m"):
            cfg.validate()

    def test_schema_rejects_bolt_pattern_with_empty_centers(self):
        """bolt_pattern with an empty bolt_centers_m list must fail validation."""
        cfg = LoadFaceConfig(
            face="x_max",
            selector="bolt_pattern",
            bolt_centers_m=[],
            disk_radius_m=_NEMA_DISK_R,
            direction=[0.0, 0.0, -1.0],
            magnitude_n=5000.0,
        )
        with pytest.raises(AssertionError, match="bolt_centers_m"):
            cfg.validate()

    def test_deduplication_of_overlapping_disks(self):
        """
        Two bolt centers close enough that their disks overlap should not
        double-count nodes. Total force must still equal magnitude_n.
        Force conservation is the observable: if deduplication is broken,
        force_per_node is computed on the pre-dedup count and the sum is wrong.
        """
        overlapping_centers = [[0.030, 0.040], [0.033, 0.040]]
        lc = LoadCaseConfig(
            fixed=FixedFaceConfig(face="x_min", selector="full"),
            load=LoadFaceConfig(
                face="x_max",
                selector="bolt_pattern",
                bolt_centers_m=overlapping_centers,
                disk_radius_m=_NEMA_DISK_R,
                direction=[0.0, 0.0, -1.0],
                magnitude_n=1000.0,
            ),
        )
        result = build_load_case(
            _motor_mount_geom(), None, _motor_mount_grid(),
            load_case_config=lc,
        )
        total = sum(result["load_vals"])
        assert abs(total - (-1000.0)) < 1e-6, (
            f"Overlapping disk deduplication broken: total={total:.6f}N, "
            f"expected -1000.0N"
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