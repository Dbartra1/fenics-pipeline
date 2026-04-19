# src/fea/boundary_conditions.py
#
# Maps gmsh physical group tags to FEniCSx boundary conditions.
# Dirichlet (displacement = 0) applied to "bottom" tag.
# Neumann (traction force) applied to "top" or face named in load_hints.
#
# Two BC builders are provided:
#   build_boundary_conditions()           — tag-based, used by solver.py (Stage 3)
#   build_boundary_conditions_geometric() — geometry-based, used by simp.py (Stage 4)
#     All behaviour is driven by a BoundaryConditions dataclass read from
#     params.json so nothing is hardcoded.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import dolfinx.mesh as dmesh
from dolfinx.io import XDMFFile
import ufl


@dataclass
class BoundaryConditionSet:
    """
    Container for all BCs for a single load case.

    dirichlet:     list of dolfinx DirichletBC objects
    traction_tag:  integer tag where traction is applied
    traction_vec:  traction vector [Fx, Fy, Fz] in N/m²
    ds:            ufl surface measure restricted to boundary tags
    """
    dirichlet:    list
    traction_tag: int
    traction_vec: np.ndarray
    ds:           ufl.Measure


# Physical group tag constants — must match tag_physical_groups() in gmsh_pipeline.py
TAG_VOLUME = 1
TAG_TOP    = 2
TAG_BOTTOM = 3
TAG_SIDES  = 4


def load_boundary_mesh(
    boundaries_xdmf: str,
    domain: dolfinx.mesh.Mesh,
) -> dolfinx.mesh.MeshTags:
    """
    Load boundary facet tags from the _boundaries.xdmf file written in Stage 2.
    """
    with XDMFFile(domain.comm, boundaries_xdmf, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(domain, name="Grid")
    return facet_tags


def build_dirichlet_bc(
    V: fem.FunctionSpace,
    facet_tags: dolfinx.mesh.MeshTags,
    fixed_tag: int = TAG_BOTTOM,
) -> list:
    """
    Zero-displacement Dirichlet BC on all DOFs belonging to fixed_tag facets.
    """
    domain       = V.mesh
    fdim         = domain.topology.dim - 1
    fixed_facets = facet_tags.find(fixed_tag)
    fixed_dofs   = fem.locate_dofs_topological(V, fdim, fixed_facets)
    u_zero = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=np.float64))
    bc     = fem.dirichletbc(u_zero, fixed_dofs, V)
    return [bc]


def build_traction_bc(
    domain: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    load_hints: dict,
) -> tuple[int, np.ndarray, ufl.Measure]:
    """
    Neumann traction BC from load_hints dict.
    """
    face_to_tag = {
        "top":    TAG_TOP,
        "bottom": TAG_BOTTOM,
        "sides":  TAG_SIDES,
    }

    primary_face = load_hints.get("primary_face", "top")
    traction_tag = face_to_tag.get(primary_face, TAG_TOP)
    load_n       = float(load_hints.get("load_magnitude_n", 1000.0))

    coords    = domain.geometry.x
    bb_min    = coords.min(axis=0)
    bb_max    = coords.max(axis=0)
    face_area = (bb_max[0] - bb_min[0]) * (bb_max[1] - bb_min[1])
    face_area = max(face_area, 1e-6)

    traction_vec = np.array([0.0, 0.0, -load_n / face_area], dtype=np.float64)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    return traction_tag, traction_vec, ds


def build_boundary_conditions(
    V: fem.FunctionSpace,
    domain: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    load_hints: dict,
) -> BoundaryConditionSet:
    """
    Tag-based BC builder — used by solver.py (Stage 3).
    Requires a boundary mesh with gmsh physical group tags.
    """
    dirichlet = build_dirichlet_bc(V, facet_tags)
    traction_tag, traction_vec, ds = build_traction_bc(domain, facet_tags, load_hints)
    return BoundaryConditionSet(
        dirichlet=dirichlet,
        traction_tag=traction_tag,
        traction_vec=traction_vec,
        ds=ds,
    )


def build_boundary_conditions_geometric(
    V: fem.FunctionSpace,
    domain: dolfinx.mesh.Mesh,
    load_hints: dict,
    bc_params=None,
    geometry_params=None,
) -> BoundaryConditionSet:
    """
    Geometry-based BC builder — used by simp.py (Stage 4).
    Does not require gmsh tags — locates boundaries by coordinate.

    All behaviour is driven by bc_params (a BoundaryConditions dataclass)
    read from params.json. If bc_params is None, sensible defaults apply:
        fixed_face="corners", load_face="top", load_direction=[0,0,-1]

    Supported fixed_face values:
        "top", "bottom", "left", "right", "front", "back"
            — fully fix the named face (encastre)
        "corners"
            — fix only the 4 corner mounting-hole regions on the bottom face.
              When geometry_params (a GeometryParams dataclass) is provided,
              uses a disk predicate centred on each hole with radius
              (hole_diameter/2 + 2mm washer margin) — physically correct for
              a bolted joint. Falls back to a rectangular corner patch sized by
              bc_params.hole_inset_fraction when geometry_params is None.

    geometry_params: Optional[GeometryParams]
        When provided and fixed_face=="corners", drives the disk predicate from
        mounting_hole_diameter and mounting_hole_inset in the params. All values
        are in mm (converted to metres internally to match domain coordinates).
    """
    from dolfinx.mesh import locate_entities_boundary, meshtags

    # Use BoundaryConditions dataclass defaults if not provided
    try:
        from src.geometry.param_schema import BoundaryConditions
        if bc_params is None:
            bc_params = BoundaryConditions()
    except ImportError:
        bc_params = _DefaultBC()

    coords = domain.geometry.x
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    tol = 1e-6

    def face_predicate(face_name):
        """Return a coordinate predicate for a named face."""
        if face_name == "bottom":
            return lambda x: np.isclose(x[2], z_min, atol=tol)
        elif face_name == "top":
            return lambda x: np.isclose(x[2], z_max, atol=tol)
        elif face_name == "left":
            return lambda x: np.isclose(x[0], x_min, atol=tol)
        elif face_name == "right":
            return lambda x: np.isclose(x[0], x_max, atol=tol)
        elif face_name == "front":
            return lambda x: np.isclose(x[1], y_min, atol=tol)
        elif face_name == "back":
            return lambda x: np.isclose(x[1], y_max, atol=tol)
        else:
            raise ValueError(f"Unknown face name: '{face_name}'. "
                             f"Valid options: top, bottom, left, right, front, back")

    fdim = domain.topology.dim - 1

    # ── Dirichlet (fixed) boundary ────────────────────────────────────────
    if bc_params.fixed_face == "corners":
        if geometry_params is not None:
            # ── Disk predicate — physically correct for bolted joints ──────
            # Constrain a disk of radius (hole_r + 2mm washer margin) centred
            # on each mounting hole. Avoids the spurious stress concentration
            # from over-constraining a large rectangular corner patch.
            #
            # geometry_params values are in mm; domain coordinates are in
            # metres (mm→m conversion happens in simp.py before this call).
            WASHER_MARGIN_M = 0.002   # 2 mm expressed in metres
            hole_r_m   = (geometry_params.mounting_hole_diameter / 2.0) / 1000.0
            inset_m    = geometry_params.mounting_hole_inset / 1000.0
            bc_disk_r  = hole_r_m + WASHER_MARGIN_M

            hole_centers = [
                (x_min + inset_m, y_min + inset_m),
                (x_max - inset_m, y_min + inset_m),
                (x_min + inset_m, y_max - inset_m),
                (x_max - inset_m, y_max - inset_m),
            ]

            # Use a slightly relaxed z-tolerance: gmsh Delaunay can introduce
            # coordinate noise at ~1e-7m; 1e-4 (0.1mm) is safe for any part
            # taller than 1mm while avoiding spurious facet capture.
            z_tol_bc = 1e-4

            def corner_predicate(x):
                on_bottom  = np.isclose(x[2], z_min, atol=z_tol_bc)
                near_hole  = np.zeros(x.shape[1], dtype=bool)
                for hx, hy in hole_centers:
                    near_hole |= (
                        np.sqrt((x[0] - hx)**2 + (x[1] - hy)**2) < bc_disk_r
                    )
                return on_bottom & near_hole

            print(f"  Corner BC: disk predicate, r={bc_disk_r*1000:.1f}mm, "
                  f"{len(hole_centers)} holes at inset={inset_m*1000:.1f}mm")
        else:
            # ── Rectangular fallback — used when geometry_params unavailable ─
            # Falls back to the original rectangular corner patch sized by
            # hole_inset_fraction. Less physically accurate but always works.
            frac  = bc_params.hole_inset_fraction
            x_tol = (x_max - x_min) * frac
            y_tol = (y_max - y_min) * frac

            def corner_predicate(x):
                on_bottom   = np.isclose(x[2], z_min, atol=tol)
                near_x_edge = (x[0] < x_min + x_tol) | (x[0] > x_max - x_tol)
                near_y_edge = (x[1] < y_min + y_tol) | (x[1] > y_max - y_tol)
                return on_bottom & near_x_edge & near_y_edge

            print(f"  Corner BC: rectangular fallback "
                  f"(pass geometry_params for disk predicate), "
                  f"frac={frac:.2f}")

        fixed_facets = locate_entities_boundary(domain, fdim, corner_predicate)
    else:
        fixed_facets = locate_entities_boundary(
            domain, fdim, face_predicate(bc_params.fixed_face)
        )

    if len(fixed_facets) == 0:
        raise ValueError(
            f"No facets found for fixed_face='{bc_params.fixed_face}'. "
            f"Check params.json boundary_conditions.fixed_face."
        )

    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
    u_zero     = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=np.float64))
    dirichlet  = [fem.dirichletbc(u_zero, fixed_dofs, V)]

    print(f"  Fixed BCs: '{bc_params.fixed_face}' "
          f"({len(fixed_facets)} facets, {len(fixed_dofs)} DOFs)")

    # ── Neumann (traction) boundary ───────────────────────────────────────
    load_facets = locate_entities_boundary(
        domain, fdim, face_predicate(bc_params.load_face)
    )

    if len(load_facets) == 0:
        raise ValueError(
            f"No facets found for load_face='{bc_params.load_face}'. "
            f"Check params.json boundary_conditions.load_face."
        )

    facet_vals     = np.ones(len(load_facets), dtype=np.int32)
    facet_tags_obj = meshtags(domain, fdim, load_facets, facet_vals)
    ds             = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags_obj)

    # Face area for traction magnitude conversion (N → N/m²)
    face_name = bc_params.load_face
    if face_name in ("top", "bottom"):
        face_area = (x_max - x_min) * (y_max - y_min)
    elif face_name in ("left", "right"):
        face_area = (y_max - y_min) * (z_max - z_min)
    else:  # front, back
        face_area = (x_max - x_min) * (z_max - z_min)
    face_area = max(face_area, 1e-10)

    load_n    = float(load_hints.get("load_magnitude_n", 10000.0))
    direction = np.array(bc_params.load_direction, dtype=np.float64)
    norm      = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    traction = direction * (load_n / face_area)

    print(f"  Load BCs:  '{bc_params.load_face}' face, "
          f"{load_n:.0f} N, dir={direction.tolist()}, "
          f"area={face_area*1e6:.1f} mm²")

    return BoundaryConditionSet(
        dirichlet=dirichlet,
        traction_tag=1,
        traction_vec=traction,
        ds=ds,
    )


class _DefaultBC:
    """Fallback defaults when param_schema cannot be imported."""
    fixed_face          = "corners"
    load_face           = "top"
    load_direction      = [0.0, 0.0, -1.0]
    hole_inset_fraction = 0.15
    shell_thickness_mm  = 2.0
