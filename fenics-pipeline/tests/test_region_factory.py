# tests/test_region_factory.py
#
# Tests for src/geometry/region_factory.py.
#
# Covers:
#   - part_center_m helper: correct dispatch for disk / box / absent shape
#   - resolve_geometry_regions: disk shape emits correct void + nondesign
#   - resolve_geometry_regions: box / absent shape emits no auto-regions
#   - Factory appends to (not replaces) JSON-declared regions
#
# Run from repo root:
#   python -m pytest tests/test_region_factory.py -v

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.geometry.param_schema import PipelineParams
from src.geometry.region_factory import part_center_m, resolve_geometry_regions

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAD_DIR = REPO_ROOT / "scad"


# ─────────────────────────────────────────────────────────────────────────────
# part_center_m
# ─────────────────────────────────────────────────────────────────────────────

class TestPartCenterM:

    def _geom(self, **kw):
        from src.geometry.param_schema import GeometryParams
        return GeometryParams(**kw)

    def test_disk_center(self):
        """Disk shape: centre = (diameter/2, diameter/2) in metres."""
        g = self._geom(length=80, width=80, height=25, shape="disk", diameter=80.0)
        cx, cy = part_center_m(g)
        assert cx == pytest.approx(0.040)
        assert cy == pytest.approx(0.040)

    def test_box_center(self):
        """Box shape: centre = (length/2, width/2) in metres."""
        g = self._geom(length=100, width=60, height=20)
        cx, cy = part_center_m(g)
        assert cx == pytest.approx(0.050)
        assert cy == pytest.approx(0.030)

    def test_box_explicit_shape(self):
        """Explicit shape='box' same result as absent."""
        g = self._geom(length=100, width=60, height=20, shape="box")
        cx, cy = part_center_m(g)
        assert cx == pytest.approx(0.050)
        assert cy == pytest.approx(0.030)

    def test_asymmetric_disk(self):
        """A disk with diameter != length/width still uses diameter."""
        g = self._geom(length=100, width=100, height=10, shape="disk", diameter=60.0)
        cx, cy = part_center_m(g)
        assert cx == pytest.approx(0.030)
        assert cy == pytest.approx(0.030)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_geometry_regions
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveGeometryRegions:

    def _make_params(self, **geometry_overrides) -> PipelineParams:
        raw = {
            "part_name": "test",
            "geometry": {
                "length": 80.0, "width": 80.0, "height": 25.0,
                **geometry_overrides,
            },
            "mesh_hints": {"target_element_size": 5.0, "opt_domain_element_size_mm": 2.5},
            "load_hints": {"primary_face": "top", "load_magnitude_n": 1000.0},
            "export": {"stl_output_dir": "/tmp", "stl_ascii": False},
        }
        return PipelineParams.from_dict(raw)

    def test_box_no_auto_regions(self):
        """Box shape (or absent) → factory adds nothing."""
        p = self._make_params()
        voids, nondesign = resolve_geometry_regions(p)
        assert voids == []
        assert nondesign == []

    def test_disk_emits_exterior_void(self):
        """Disk shape → at least one cylinder_z_exterior void."""
        p = self._make_params(shape="disk", diameter=80.0)
        voids, nondesign = resolve_geometry_regions(p)
        assert any(v.type == "cylinder_z_exterior" for v in voids)

    def test_disk_exterior_void_geometry(self):
        """Exterior void cylinder centre and radius match the disk."""
        p = self._make_params(shape="disk", diameter=80.0)
        voids, _ = resolve_geometry_regions(p)
        ext = [v for v in voids if v.type == "cylinder_z_exterior"][0]
        assert ext.cx == pytest.approx(0.040)
        assert ext.cy == pytest.approx(0.040)
        assert ext.radius == pytest.approx(0.040)

    def test_disk_with_leg_holes_emits_nondesign(self):
        """Disk + leg hole fields → nondesign cylinders at polar positions."""
        p = self._make_params(
            shape="disk", diameter=80.0,
            leg_hole_d=5.0, leg_hole_radius=28.0,
            num_legs=3, first_leg_angle=90.0,
        )
        _, nondesign = resolve_geometry_regions(p)
        # Should have at least one nondesign region with 3 centres
        leg_regions = [r for r in nondesign if len(r.centers_m) == 3]
        assert len(leg_regions) == 1
        # First leg at 90° → should be at (cx, cy + leg_r)
        centers = leg_regions[0].centers_m
        cx_m = 0.040
        cy_m = 0.040
        leg_r_m = 0.028
        expected_first = [cx_m + leg_r_m * math.cos(math.radians(90)),
                          cy_m + leg_r_m * math.sin(math.radians(90))]
        assert centers[0][0] == pytest.approx(expected_first[0], abs=1e-6)
        assert centers[0][1] == pytest.approx(expected_first[1], abs=1e-6)

    def test_disk_with_center_hole_emits_nondesign(self):
        """Disk + center_hole_d → nondesign cylinder at centre."""
        p = self._make_params(
            shape="disk", diameter=80.0,
            center_hole_d=7.0, center_hole_wall_mm=5.0,
        )
        _, nondesign = resolve_geometry_regions(p)
        center_regions = [r for r in nondesign if len(r.centers_m) == 1]
        assert len(center_regions) == 1
        c = center_regions[0].centers_m[0]
        assert c[0] == pytest.approx(0.040)
        assert c[1] == pytest.approx(0.040)

    def test_factory_appends_to_declared(self):
        """JSON-declared regions are preserved; factory appends, never replaces."""
        raw = {
            "part_name": "test",
            "geometry": {
                "length": 80.0, "width": 80.0, "height": 25.0,
                "shape": "disk", "diameter": 80.0,
            },
            "mesh_hints": {"target_element_size": 5.0, "opt_domain_element_size_mm": 2.5},
            "load_hints": {"primary_face": "top", "load_magnitude_n": 1000.0},
            "export": {"stl_output_dir": "/tmp", "stl_ascii": False},
            "void_regions": [
                {"type": "box", "z_min": 0.020},
            ],
        }
        p = PipelineParams.from_dict(raw)
        voids, _ = resolve_geometry_regions(p)
        # Should have the declared box + the factory's cylinder_z_exterior
        types = [v.type for v in voids]
        assert "box" in types
        assert "cylinder_z_exterior" in types

    def test_unknown_shape_raises(self):
        """Unknown shape → ValueError, not silent rectangular fallback."""
        p = self._make_params(shape="hexagon")
        with pytest.raises(ValueError, match="Unknown geometry.shape"):
            resolve_geometry_regions(p)

    def test_live_tripod_json(self):
        """End-to-end: real tripod JSON resolves correct regions."""
        raw = json.loads((SCAD_DIR / "tripod_mount_base_params.json").read_text())
        p = PipelineParams.from_dict(raw)
        voids, nondesign = resolve_geometry_regions(p)

        # Must have exterior void
        assert any(v.type == "cylinder_z_exterior" for v in voids)
        # Must have centre hole (1 centre) + leg holes (3 centres)
        center_nd = [r for r in nondesign if len(r.centers_m) == 1]
        leg_nd    = [r for r in nondesign if len(r.centers_m) == 3]
        assert len(center_nd) == 1
        assert len(leg_nd) == 1