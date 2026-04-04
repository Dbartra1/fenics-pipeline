# src/fea/boundary_conditions.py
#
# Maps gmsh physical group tags to FEniCSx boundary conditions.
# Dirichlet (displacement = 0) applied to "bottom" tag.
# Neumann (traction force) applied to "top" or face named in load_hints.
#
# FEniCSx BC note: DirichletBC in dolfinx 0.8 requires locating DOFs via
# meshtags, not via subdomain markers. The API changed significantly from
# legacy FEniCS — do not use locate_dofs_geometrical for tag-based BCs.

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

    dirichlet:     list of dolfinx DirichletBC objects (applied via petsc solver)
    traction_tag:  integer physical group tag where traction is applied
    traction_vec:  traction vector as numpy array [Fx, Fy, Fz] in N/m²
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
    Returns a MeshTags object mapping facets → physical group integers.

    The MeshTags object is what FEniCSx uses to locate DOFs and define
    surface integrals — it's the bridge between gmsh tags and ufl measures.
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
    Represents a fully fixed (encastre) boundary — no translation in any direction.

    For a more realistic fixture (e.g. fixed in Z only), replace the full-vector
    zero with a component-wise constraint using locate_dofs_topological per axis.
    """
    domain = V.mesh
    fdim = domain.topology.dim - 1

    fixed_facets = facet_tags.find(fixed_tag)
    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)

    u_zero = fem.Constant(domain, np.zeros(domain.geometry.dim, dtype=np.float64))
    bc = fem.dirichletbc(u_zero, fixed_dofs, V)
    return [bc]


def build_traction_bc(
    domain: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    load_hints: dict,
) -> tuple[int, np.ndarray, ufl.Measure]:
    """
    Neumann traction BC from load_hints dict (carried from scad/params.json).

    load_hints["primary_face"] maps to a physical group tag.
    load_hints["load_magnitude_n"] is total force in Newtons — converted to
    traction (N/m²) by dividing by the approximate face area.

    Returns (traction_tag, traction_vector, ds_measure).
    """
    face_to_tag = {
        "top":    TAG_TOP,
        "bottom": TAG_BOTTOM,
        "sides":  TAG_SIDES,
    }

    primary_face = load_hints.get("primary_face", "top")
    traction_tag = face_to_tag.get(primary_face, TAG_TOP)
    load_n = float(load_hints.get("load_magnitude_n", 1000.0))

    # Approximate face area from bounding box — good enough for traction conversion
    # A more accurate approach: integrate ufl.Constant(1)*ds(traction_tag)
    coords = domain.geometry.x
    bb_min = coords.min(axis=0)
    bb_max = coords.max(axis=0)
    face_area = (bb_max[0] - bb_min[0]) * (bb_max[1] - bb_min[1])
    face_area = max(face_area, 1e-6)  # guard against degenerate geometry

    # Apply load in -Z direction (compression from top)
    traction_vec = np.array([0.0, 0.0, -load_n / face_area], dtype=np.float64)

    # Surface measure restricted to boundary tags
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    return traction_tag, traction_vec, ds


def build_boundary_conditions(
    V: fem.FunctionSpace,
    domain: dolfinx.mesh.Mesh,
    facet_tags: dolfinx.mesh.MeshTags,
    load_hints: dict,
) -> BoundaryConditionSet:
    """
    Top-level BC builder — called by solver.py.
    Assembles Dirichlet and Neumann BCs into a single BoundaryConditionSet.
    """
    dirichlet = build_dirichlet_bc(V, facet_tags)
    traction_tag, traction_vec, ds = build_traction_bc(domain, facet_tags, load_hints)

    return BoundaryConditionSet(
        dirichlet=dirichlet,
        traction_tag=traction_tag,
        traction_vec=traction_vec,
        ds=ds,
    )