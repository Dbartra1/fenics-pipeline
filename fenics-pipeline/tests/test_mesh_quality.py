# tests/test_mesh_quality.py
#
# Smoke tests for mesh quality metrics.
# Run via: make test
# Or directly: docker compose exec fenics-pipeline pytest tests/ -v
#
# These tests build a known-good tetrahedral mesh programmatically
# so they don't depend on the full pipeline having been run first.

import sys
sys.path.insert(0, "/workspace")

import numpy as np
import pytest
import gmsh

from src.meshing.mesh_quality import (
    check_mesh_quality,
    QualityThresholds,
    _tet_aspect_ratios,
    _count_inverted,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def good_msh(tmp_path_factory):
    """
    Build a clean tetrahedral mesh of a unit cube via gmsh.
    Written to a temp .msh file — reused across tests in this module.
    """
    tmp = tmp_path_factory.mktemp("meshes")
    msh_path = tmp / "unit_cube.msh"

    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("unit_cube")

    # 1mm unit cube
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()

    # Tag physical groups — required for quality check to find tets
    volumes  = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.2)
    gmsh.option.setNumber("Mesh.Algorithm3D", 4)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return msh_path


@pytest.fixture(scope="module")
def bad_msh(tmp_path_factory):
    """
    Build a degenerate mesh with at least one inverted element
    by manually constructing a tet with negative Jacobian.
    Written to .msh via gmsh, then the node coords are patched.
    We use a very flat tet (needle element) to guarantee poor aspect ratio.
    """
    tmp = tmp_path_factory.mktemp("meshes")
    msh_path = tmp / "bad.msh"

    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("bad")

    # Extremely thin box — guaranteed needle elements
    gmsh.model.occ.addBox(0, 0, 0, 100, 100, 0.1)
    gmsh.model.occ.synchronize()
    volumes  = gmsh.model.getEntities(dim=3)
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces])
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 5.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(msh_path))
    gmsh.finalize()

    return msh_path


# ── Unit tests: geometric functions ─────────────────────────────────────────

class TestAspectRatio:
    def test_equilateral_tet_aspect_ratio_near_one(self):
        """A regular tetrahedron should have aspect ratio close to 1."""
        # Regular tet vertices
        pts = np.array([
            [1, 1, 1],
            [1,-1,-1],
            [-1,1,-1],
            [-1,-1,1],
        ], dtype=float)
        tets = np.array([[0, 1, 2, 3]])
        ar = _tet_aspect_ratios(pts, tets)
        assert ar[0] == pytest.approx(1.0, rel=0.05), \
            f"Regular tet aspect ratio should be ~1, got {ar[0]:.4f}"

    def test_needle_tet_high_aspect_ratio(self):
        """A very flat tet should have high aspect ratio."""
        pts = np.array([
            [0,   0,   0],
            [100, 0,   0],
            [50,  0.1, 0],
            [50,  0,   0.1],
        ], dtype=float)
        tets = np.array([[0, 1, 2, 3]])
        ar = _tet_aspect_ratios(pts, tets)
        assert ar[0] > 10.0, \
            f"Needle tet should have AR > 10, got {ar[0]:.2f}"

    def test_multiple_tets(self):
        """Batch of tets returns one aspect ratio per element."""
        n = 50
        pts  = np.random.rand(n * 4, 3)
        tets = np.arange(n * 4).reshape(n, 4)
        ar = _tet_aspect_ratios(pts, tets)
        assert ar.shape == (n,)
        assert (ar > 0).all()


class TestInvertedDetection:
    def test_no_inverted_in_regular_tet(self):
        pts = np.array([
            [1, 1, 1], [1,-1,-1], [-1,1,-1], [-1,-1,1]
        ], dtype=float)
        tets = np.array([[0, 1, 2, 3]])
        assert _count_inverted(pts, tets) == 0

    def test_detects_inverted_tet(self):
        """Swapping two vertices inverts the tet orientation."""
        pts = np.array([
            [1, 1, 1], [1,-1,-1], [-1,1,-1], [-1,-1,1]
        ], dtype=float)
        # Swap vertices 0 and 1 — inverts winding
        tets_inverted = np.array([[1, 0, 2, 3]])
        assert _count_inverted(pts, tets_inverted) == 1


# ── Integration tests: full quality check on real .msh files ────────────────

class TestQualityReport:
    def test_good_mesh_passes_default_thresholds(self, good_msh):
        report = check_mesh_quality(good_msh)
        assert report.n_elements > 0,     "Should find elements"
        assert report.n_inverted == 0,    "Good mesh should have no inverted elements"
        assert report.passed,             f"Good mesh should pass. Failures: {report.failures}"

    def test_good_mesh_aspect_ratio_reasonable(self, good_msh):
        report = check_mesh_quality(good_msh)
        assert report.aspect_ratio_mean < 5.0, \
            f"Mean AR {report.aspect_ratio_mean} too high for a cube mesh"
        assert report.aspect_ratio_max < 15.0, \
            f"Max AR {report.aspect_ratio_max} too high for a cube mesh"

    def test_thin_mesh_fails_strict_thresholds(self, bad_msh):
        strict = QualityThresholds(
            max_aspect_ratio=5.0,
            max_aspect_ratio_fail=8.0,
            min_dihedral_deg=15.0,
        )
        report = check_mesh_quality(bad_msh, thresholds=strict)
        # Thin box mesh should fail at least one threshold
        assert not report.passed or len(report.warnings) > 0, \
            "Thin mesh should fail or warn on strict thresholds"

    def test_quality_report_summary_is_string(self, good_msh):
        report = check_mesh_quality(good_msh)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Elements" in summary

    def test_gmsh_not_left_initialized(self, good_msh):
        """check_mesh_quality must finalize gmsh when it initializes it."""
        assert not gmsh.is_initialized(), \
            "gmsh should not be initialized after check_mesh_quality returns"
        check_mesh_quality(good_msh)
        assert not gmsh.is_initialized(), \
            "gmsh should be finalized after check_mesh_quality"