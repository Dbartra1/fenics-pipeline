# tests/test_param_schema.py
#
# Schema-level regression tests for src/geometry/param_schema.py.
#
# Covers:
#   - to_openscad_defines() filters non-numeric geometry fields
#     (regression guard for the `-D SHAPE=disk` bug)
#   - FixedFaceConfig / LoadFaceConfig selector validation (Task 2)
#   - Backward compatibility: existing part JSONs (base_part, cantilever_arm)
#     parse without a `selector` on the load block and default to "full".
#
# Run from repo root:
#   python -m pytest tests/test_param_schema.py -v
#
# Notes:
#   - These tests deliberately avoid touching voxelize / the Rust solver;
#     they cover the pure-Python schema surface only.
#   - If these fail after editing param_schema.py, remember to clear
#     __pycache__ AND restart the kernel — both are required in WSL2.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.geometry.param_schema import (
    PipelineParams,
    GeometryParams,
    FixedFaceConfig,
    LoadFaceConfig,
    LoadCaseConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAD_DIR = REPO_ROOT / "scad"


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: to_openscad_defines() must filter non-numeric fields
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenSCADDefinesFilter:
    """
    Regression guard for the `-D SHAPE="disk"` bug: geometry-factory
    metadata (strings, lists, dicts) must never leak to OpenSCAD's -D flags.
    """

    def _make_params(self, **geometry_overrides) -> PipelineParams:
        """Minimal valid PipelineParams with overridable geometry."""
        raw = {
            "part_name": "test_part",
            "geometry": {
                "length": 80.0, "width": 80.0, "height": 25.0,
                **geometry_overrides,
            },
            "mesh_hints": {
                "target_element_size": 5.0,
                "opt_domain_element_size_mm": 2.5,
            },
            "load_hints": {"primary_face": "top", "load_magnitude_n": 1000.0},
            "export": {"stl_output_dir": "/tmp", "stl_ascii": False},
        }
        return PipelineParams.from_dict(raw)

    def test_string_field_is_filtered(self):
        """shape='disk' must not appear in -D defines."""
        p = self._make_params(shape="disk", diameter=80.0)
        defines = p.to_openscad_defines()
        assert "SHAPE" not in defines, \
            "String field 'shape' leaked through filter — would produce -D SHAPE=\"disk\""

    def test_numeric_fields_preserved(self):
        """Floats and ints must pass through with uppercase keys."""
        p = self._make_params(
            diameter=80.0,
            num_legs=3,            # int
            first_leg_angle=90.0,  # float
        )
        d = p.to_openscad_defines()
        assert d["DIAMETER"] == 80.0
        assert d["NUM_LEGS"] == 3 and isinstance(d["NUM_LEGS"], int)
        assert d["FIRST_LEG_ANGLE"] == 90.0

    def test_required_scalars_always_present(self):
        """length, width, height always survive the filter."""
        p = self._make_params(shape="disk")
        d = p.to_openscad_defines()
        assert {"LENGTH", "WIDTH", "HEIGHT"}.issubset(d.keys())

    def test_list_field_is_filtered(self):
        """A list-valued geometry field must not be emitted."""
        p = self._make_params(
            shape="box",
            refinement_centers=[[0.01, 0.01], [0.09, 0.05]],
        )
        d = p.to_openscad_defines()
        assert "REFINEMENT_CENTERS" not in d

    def test_dict_field_is_filtered(self):
        """A dict-valued geometry field must not be emitted."""
        p = self._make_params(
            shape="box",
            meta={"origin": "step_import", "version": "v2"},
        )
        d = p.to_openscad_defines()
        assert "META" not in d

    def test_bool_field_passes_through(self):
        """
        bool is a subclass of int in Python, so bools fall through the
        numeric filter. openscad_runner has a dedicated bool branch that
        formats them as `true`/`false`, so this is intentional.
        """
        p = self._make_params(chamfer_enabled=True)
        d = p.to_openscad_defines()
        assert d["CHAMFER_ENABLED"] is True

    def test_live_tripod_json_no_shape_leak(self):
        """End-to-end: real tripod_mount_base_params.json must not emit SHAPE."""
        raw = json.loads((SCAD_DIR / "tripod_mount_base_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        d = p.to_openscad_defines()

        assert "SHAPE" not in d
        for expected in ("DIAMETER", "HEIGHT", "CENTER_HOLE_D",
                         "LEG_HOLE_D", "LEG_HOLE_RADIUS",
                         "NUM_LEGS", "FIRST_LEG_ANGLE"):
            assert expected in d, f"expected numeric field {expected!r} missing from defines"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: FixedFaceConfig selector validation
# ─────────────────────────────────────────────────────────────────────────────

class TestFixedFaceConfig:

    def test_default_corners_still_valid(self):
        """Existing corners selector must keep working unchanged."""
        f = FixedFaceConfig(face="z_min", selector="corners",
                            inset_m=0.010, disk_radius_m=0.005)
        f.validate()  # must not raise

    def test_full_selector_valid(self):
        f = FixedFaceConfig(face="z_min", selector="full")
        f.validate()

    def test_leg_holes_selector_valid(self):
        """leg_holes selector must pass validation."""
        f = FixedFaceConfig(face="z_min", selector="leg_holes",
                            disk_radius_m=0.004)
        f.validate()

    def test_center_disk_selector_valid(self):
        """center_disk selector must pass validation."""
        f = FixedFaceConfig(face="z_min", selector="center_disk",
                            disk_radius_m=0.006)
        f.validate()

    def test_unknown_selector_rejected(self):
        with pytest.raises(AssertionError, match="selector must be one of"):
            FixedFaceConfig(face="z_min", selector="triangles").validate()

    def test_valid_selectors_contains_expected(self):
        """All known selectors must be present in VALID_SELECTORS."""
        # Test membership rather than exact equality so adding future selectors
        # doesn't require updating this test — only the rejection test matters
        # for exact-set semantics.
        for selector in ("full", "corners", "leg_holes", "center_disk"):
            assert selector in FixedFaceConfig.VALID_SELECTORS, \
                f"'{selector}' missing from FixedFaceConfig.VALID_SELECTORS"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: LoadFaceConfig selector validation
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFaceConfig:

    def test_default_selector_is_full(self):
        """Back-compat: JSONs without `selector` default to 'full'."""
        l = LoadFaceConfig(face="z_max", magnitude_n=1000.0)
        assert l.selector == "full"
        l.validate()

    def test_center_disk_selector_valid(self):
        """center_disk selector passes validation."""
        l = LoadFaceConfig(face="z_max", selector="center_disk",
                           magnitude_n=5000.0, disk_radius_m=0.004)
        l.validate()

    def test_bolt_pattern_selector_valid(self):
        """bolt_pattern selector passes validation when centers are provided."""
        l = LoadFaceConfig(
            face="x_max", selector="bolt_pattern",
            bolt_centers_m=[[0.0145, 0.0245], [0.0455, 0.0555]],
            disk_radius_m=0.007, direction=[0.0, 0.0, -1.0],
            magnitude_n=5000.0,
        )
        l.validate()  # must not raise

    def test_center_disk_requires_positive_radius(self):
        with pytest.raises(AssertionError, match="disk_radius_m must be > 0"):
            LoadFaceConfig(face="z_max", selector="center_disk",
                           magnitude_n=5000.0, disk_radius_m=0.0).validate()

    def test_unknown_selector_rejected(self):
        with pytest.raises(AssertionError, match="selector must be one of"):
            LoadFaceConfig(face="z_max", selector="spiral").validate()

    def test_valid_selectors_contains_expected(self):
        """All known selectors must be present in VALID_SELECTORS."""
        for selector in ("full", "center_disk", "bolt_pattern"):
            assert selector in LoadFaceConfig.VALID_SELECTORS, \
                f"'{selector}' missing from LoadFaceConfig.VALID_SELECTORS"


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: End-to-end parse of live JSONs
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveJSONCompat:
    """
    Every shipped part JSON must still parse cleanly after the schema
    extension. Tests that adding new optional fields with defaults did
    not break existing configurations.
    """

    @pytest.mark.parametrize("part_name", [
        "base_part", "cantilever_arm", "tripod_mount_base", "motor_mount",
    ])
    def test_parse_ok(self, part_name):
        path = SCAD_DIR / f"{part_name}_params.json"
        raw = json.loads(path.read_text())
        p = PipelineParams.from_dict(raw)
        p.validate()  # must not raise

    def test_motor_mount_uses_bolt_seats(self):
        """Motor mount should use bolt_seats, not cylinder_x nondesign."""
        raw = json.loads((SCAD_DIR / "motor_mount_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert len(p.bolt_seats) > 0, "motor_mount should declare bolt_seats"
        # All bolt seats should be along x axis (wall-to-motor direction)
        for bs in p.bolt_seats:
            assert bs.type == "bolt_seat_x"

    def test_motor_mount_uses_bolt_pattern_load(self):
        """Session 6: motor mount load selector must be bolt_pattern."""
        raw = json.loads((SCAD_DIR / "motor_mount_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert p.load_case_config.load.selector == "bolt_pattern", (
            "motor_mount load selector should be 'bolt_pattern' after Session 6 fix"
        )
        assert p.load_case_config.load.bolt_centers_m is not None
        assert len(p.load_case_config.load.bolt_centers_m) == 4, \
            "motor_mount should have 4 NEMA bolt centers"

    def test_motor_mount_uses_ring_not_slab(self):
        """Motor mount uses through-ring bolt seats, NOT attachment slabs.

        Rectangular attachment slabs were abandoned (non-manifold geometry at
        marching-cubes junctions). The motor mount must therefore declare ZERO
        attachment_regions; fixity comes from bolt-seat collars + the `fixed`
        corner disks. Each bolt seat carries a through_ring_radius_m — the ring
        that replaced the slab. Do NOT reintroduce attachment slabs to make a
        test pass.
        """
        raw = json.loads((SCAD_DIR / "motor_mount_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert len(p.attachment_regions) == 0, (
            "motor_mount must have NO attachment_regions — slabs were retired in "
            "favour of through-ring bolt seats."
        )
        assert len(p.bolt_seats) >= 1, "motor_mount should declare bolt_seats"
        for bs in p.bolt_seats:
            assert bs.through_ring_radius_m and bs.through_ring_radius_m > 0, (
                f"bolt_seat {bs.type} must set through_ring_radius_m (the ring "
                f"mechanism that replaced the slab), got {bs.through_ring_radius_m}"
            )

    def test_motor_mount_bolt_seats_cover_both_faces(self):
        """Bolt holes appear on BOTH faces via a matched seat pair.

        The both-faces hole comes from a complementary pair of bolt seats (one
        entry-side collar, one exit-side collar) plus the through-ring — NOT
        from every seat being entry_seat=True. This asserts the pair covers
        both faces; the per-seat entry/exit split is an intentional geometry
        knob, so individual booleans are not pinned here.
        """
        raw = json.loads((SCAD_DIR / "motor_mount_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert len(p.bolt_seats) >= 2, (
            "motor_mount needs at least a matched pair of bolt_seats to cover "
            "both faces"
        )
        assert any(bs.entry_seat for bs in p.bolt_seats), \
            "no bolt_seat provides an entry-side collar (low-coord face)"
        assert any(bs.exit_seat for bs in p.bolt_seats), \
            "no bolt_seat provides an exit-side collar (high-coord face)"

    def test_base_part_load_defaults_to_full(self):
        """base_part.json has no `selector` on load → should default to 'full'."""
        raw = json.loads((SCAD_DIR / "base_part_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert p.load_case_config is not None
        assert p.load_case_config.load.selector == "full"

    def test_pre_task2_tripod_still_parses(self):
        """
        The tripod JSON in the repo still carries the pre-Task-2 `corners`
        selector (not yet updated to leg_holes/center_disk). It must still
        parse so we never end up with a broken main between the schema
        change and the JSON update.
        """
        raw = json.loads((SCAD_DIR / "tripod_mount_base_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        assert p.load_case_config.fixed.selector in {"corners", "leg_holes"}
        assert p.load_case_config.load.selector in {"full", "center_disk"}