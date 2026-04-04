# src/meshing/mesh_quality.py
#
# Mesh quality metrics for tetrahedral meshes.
# Called by 02_mesh_gmsh.ipynb and tests/test_mesh_quality.py.
#
# Three metrics matter for FEniCSx linear elasticity:
#   - Aspect ratio:   longest edge / shortest altitude. Ideal = 1.0 (equilateral tet).
#                     High values cause conditioning issues in the stiffness matrix.
#   - Min dihedral:   angle between faces. Too small → near-degenerate elements.
#   - Jacobian:       negative Jacobian → inverted element → FEniCSx will abort.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import gmsh


@dataclass
class QualityThresholds:
    max_aspect_ratio:     float = 5.0   # warn above, fail above 10.0
    min_dihedral_deg:     float = 10.0  # degrees — fail below this
    max_aspect_ratio_fail: float = 10.0
    min_jacobian:         float = 0.0   # negative = inverted element


@dataclass
class QualityReport:
    n_elements:           int
    aspect_ratio_mean:    float
    aspect_ratio_max:     float
    aspect_ratio_p95:     float
    min_dihedral_deg:     float
    n_inverted:           int           # elements with negative Jacobian
    passed:               bool
    failures:             list[str]
    warnings:             list[str]

    def summary(self) -> str:
        lines = [
            f"Elements:          {self.n_elements}",
            f"Aspect ratio:      mean={self.aspect_ratio_mean:.2f}  "
            f"max={self.aspect_ratio_max:.2f}  p95={self.aspect_ratio_p95:.2f}",
            f"Min dihedral:      {self.min_dihedral_deg:.1f}°",
            f"Inverted elements: {self.n_inverted}",
            f"Passed:            {self.passed}",
        ]
        if self.warnings:
            lines += [f"  ⚠ {w}" for w in self.warnings]
        if self.failures:
            lines += [f"  ✗ {f}" for f in self.failures]
        return "\n".join(lines)


def _tet_aspect_ratios(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """
    Compute aspect ratio for each tet.
    Aspect ratio = longest edge / (2 * sqrt(2/3) * inradius).
    Vectorized over all tets.
    """
    v0 = points[tets[:, 0]]
    v1 = points[tets[:, 1]]
    v2 = points[tets[:, 2]]
    v3 = points[tets[:, 3]]

    # 6 edge vectors
    edges = [
        v1 - v0, v2 - v0, v3 - v0,
        v2 - v1, v3 - v1, v3 - v2,
    ]
    edge_lengths = np.array([np.linalg.norm(e, axis=1) for e in edges])
    longest_edge = edge_lengths.max(axis=0)

    # Volume via scalar triple product
    a, b, c = edges[0], edges[1], edges[2]
    volume = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0
    volume = np.maximum(volume, 1e-15)  # guard against degenerate tets

    # Inradius = 3 * volume / surface_area
    def tri_area(e1, e2):
        return np.linalg.norm(np.cross(e1, e2), axis=1) / 2

    surface_area = (
        tri_area(edges[0], edges[1]) +
        tri_area(edges[0], edges[2]) +
        tri_area(edges[1], edges[2]) +
        tri_area(edges[3], edges[4])
    )
    surface_area = np.maximum(surface_area, 1e-15)
    inradius = 3.0 * volume / surface_area

    return longest_edge / (2 * np.sqrt(2/3) * inradius + 1e-15)


def _min_dihedral_angle(points: np.ndarray, tets: np.ndarray) -> float:
    """
    Compute minimum dihedral angle across all tets, in degrees.
    Checks all 6 edges per tet (4 face-pairs).
    Expensive — samples 10% of tets for large meshes.
    """
    if len(tets) > 50_000:
        idx = np.random.choice(len(tets), size=len(tets) // 10, replace=False)
        tets = tets[idx]

    min_angle = 180.0
    face_triples = [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]

    for i in range(len(tets)):
        tet_pts = points[tets[i]]
        for a, b, c in face_triples:
            n1 = np.cross(tet_pts[b]-tet_pts[a], tet_pts[c]-tet_pts[a])
            n1_norm = np.linalg.norm(n1)
            if n1_norm < 1e-15:
                continue
            n1 /= n1_norm
            for d in range(4):
                if d in (a, b, c):
                    continue
                n2 = np.cross(tet_pts[b]-tet_pts[d], tet_pts[c]-tet_pts[d])
                n2_norm = np.linalg.norm(n2)
                if n2_norm < 1e-15:
                    continue
                n2 /= n2_norm
                cos_a = np.clip(np.dot(n1, n2), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_a))
                min_angle = min(min_angle, angle)

    return min_angle


def _count_inverted(points: np.ndarray, tets: np.ndarray) -> int:
    """Count tets with negative Jacobian (inverted orientation)."""
    v0 = points[tets[:, 0]]
    v1 = points[tets[:, 1]]
    v2 = points[tets[:, 2]]
    v3 = points[tets[:, 3]]
    a, b, c = v1 - v0, v2 - v0, v3 - v0
    det = np.einsum('ij,ij->i', a, np.cross(b, c))
    return int((det <= 0).sum())


def check_mesh_quality(
    msh_path: str | Path,
    thresholds: Optional[QualityThresholds] = None,
) -> QualityReport:
    """
    Load a .msh file and compute quality metrics.
    Does NOT require gmsh to be initialized — handles init/finalize internally.
    """
    thresholds = thresholds or QualityThresholds()
    msh_path = Path(msh_path)

    was_initialized = gmsh.is_initialized()
    if not was_initialized:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)

    try:
        gmsh.open(str(msh_path))

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        points = coords.reshape(-1, 3)

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)

        all_tets = []
        for etype, enodes in zip(elem_types, elem_nodes):
            if etype == 4:  # linear tet
                all_tets.append(enodes.reshape(-1, 4) - 1)  # 0-indexed

        if not all_tets:
            return QualityReport(
                n_elements=0, aspect_ratio_mean=0, aspect_ratio_max=0,
                aspect_ratio_p95=0, min_dihedral_deg=0, n_inverted=0,
                passed=False,
                failures=["No tetrahedral elements found in mesh"],
                warnings=[],
            )

        tets = np.vstack(all_tets)
        n_elements = len(tets)

        aspect_ratios = _tet_aspect_ratios(points, tets)
        min_dihedral  = _min_dihedral_angle(points, tets)
        n_inverted    = _count_inverted(points, tets)

        failures = []
        warnings = []

        if n_inverted > 0:
            failures.append(f"{n_inverted} inverted elements — FEniCSx will abort")
        if aspect_ratios.max() > thresholds.max_aspect_ratio_fail:
            failures.append(f"Max aspect ratio {aspect_ratios.max():.1f} "
                            f"exceeds hard limit {thresholds.max_aspect_ratio_fail}")
        if min_dihedral < thresholds.min_dihedral_deg:
            failures.append(f"Min dihedral {min_dihedral:.1f}° below threshold "
                            f"{thresholds.min_dihedral_deg}°")
        if aspect_ratios.mean() > thresholds.max_aspect_ratio:
            warnings.append(f"Mean aspect ratio {aspect_ratios.mean():.2f} is high "
                            f"— consider reducing target_element_size in params.json")

        return QualityReport(
            n_elements=n_elements,
            aspect_ratio_mean=round(float(aspect_ratios.mean()), 3),
            aspect_ratio_max=round(float(aspect_ratios.max()), 3),
            aspect_ratio_p95=round(float(np.percentile(aspect_ratios, 95)), 3),
            min_dihedral_deg=round(min_dihedral, 2),
            n_inverted=n_inverted,
            passed=len(failures) == 0,
            failures=failures,
            warnings=warnings,
        )

    finally:
        gmsh.clear()
        if not was_initialized:
            gmsh.finalize()