# src/geometry/region_factory.py
#
# Geometry-driven region resolver.
#
# Reads declared geometry fields (shape, leg_hole_radius, num_legs, etc.)
# and synthesizes the void / nondesign regions they imply.
#
# The factory APPENDS to whatever the JSON already declared — it never
# silently overrides user-provided regions. This means explicit entries
# in the JSON remain authoritative and can be used as overrides or for
# one-off parts that don't fit a known shape.
#
# Supported shapes:
#   "box" / None / absent   — no auto-voids (rectangular parts, default)
#   "disk"                  — emits cylinder_z_exterior void around the
#                             circular platform.
#                             Optionally emits nondesign cylinders for:
#                               - center hole (if center_hole_d present)
#                               - N evenly-spaced leg holes (if leg_hole_d
#                                 and leg_hole_radius present)
#
# Design conventions this resolver assumes (matching the SCAD files):
#   - disk is centered at (diameter/2, diameter/2) in the XY grid
#   - diameter, *_d, *_radius fields are in mm; coords in regions are metres
#   - leg holes are arranged at num_legs equal angular intervals starting
#     at first_leg_angle (degrees, measured CCW from +X), matching
#     tripod_mount_base.scad's convention
from __future__ import annotations

import math
from typing import List, Tuple

from src.geometry.param_schema import (
    PipelineParams,
    VoidRegion,
    NondesignRegion,
)

# ── Shared geometry helpers ───────────────────────────────────────────────

def part_center_m(geom) -> Tuple[float, float]:
    """
    Return the XY centre of a part in metres, based on its declared shape.

    This is the single source of truth for "where is the middle of this
    part in physical coordinates."  Used by:
      - _resolve_disk()          (void + nondesign region placement)
      - voxelize.py leg_holes    (fixed BC disk centres)
      - voxelize.py center_disk  (load BC disk centre)

    Shape dispatch:
      "disk"          → (diameter/2, diameter/2)  mm → m
      "box" / None    → (length/2,  width/2)      mm → m
    """
    shape = geom.get("shape", None)
    if shape == "disk":
        d_mm = float(geom.diameter)
        c = (d_mm / 2.0) / 1000.0
        return (c, c)
    else:
        # box / absent / any future rectangular shape
        return (float(geom.length) / 2.0 / 1000.0,
                float(geom.width)  / 2.0 / 1000.0)


# ── Per-shape resolvers ───────────────────────────────────────────────────

def _resolve_disk(geom) -> Tuple[List[VoidRegion], List[NondesignRegion]]:
    """
    Derive void + nondesign regions for a circular disk platform.

    Required geometry fields: diameter (mm)
    Optional:
        center_hole_d           (mm) — adds a central nondesign cylinder
        center_hole_wall_mm     (mm) — wall thickness around center hole
                                       [default 5.0]
        leg_hole_d              (mm) — diameter of leg holes
        leg_hole_radius         (mm) — radial distance center → leg hole
        leg_hole_wall_mm        (mm) — wall thickness around each leg hole
                                       [default 3.0]
        num_legs                (int) [default 3]
        first_leg_angle         (deg) [default 90.0]
    """
    diameter_mm = float(geom.diameter)
    r_m = (diameter_mm / 2.0) / 1000.0
    cx_m, cy_m = part_center_m(geom)

    voids: List[VoidRegion] = [
        VoidRegion(
            type="cylinder_z_exterior",
            cx=cx_m, cy=cy_m, radius=r_m,
        )
    ]

    nondesign: List[NondesignRegion] = []

    # Central hole
    center_hole_d = geom.get("center_hole_d", None)
    if center_hole_d is not None:
        wall_mm = float(geom.get("center_hole_wall_mm", 5.0))
        nondesign.append(NondesignRegion(
            type="cylinder_z",
            centers_m=[[cx_m, cy_m]],
            void_radius_m=(float(center_hole_d) / 2.0) / 1000.0,
            wall_radius_m=((float(center_hole_d) / 2.0) + wall_mm) / 1000.0,
        ))

    # Leg holes
    leg_hole_d      = geom.get("leg_hole_d", None)
    leg_hole_radius = geom.get("leg_hole_radius", None)
    if leg_hole_d is not None and leg_hole_radius is not None:
        num_legs        = int(geom.get("num_legs", 3))
        first_angle_deg = float(geom.get("first_leg_angle", 90.0))
        leg_wall_mm     = float(geom.get("leg_hole_wall_mm", 3.0))
        leg_r_m         = (float(leg_hole_radius)) / 1000.0

        centers = []
        for i in range(num_legs):
            theta = math.radians(first_angle_deg + i * (360.0 / num_legs))
            lx = cx_m + leg_r_m * math.cos(theta)
            ly = cy_m + leg_r_m * math.sin(theta)
            centers.append([lx, ly])

        nondesign.append(NondesignRegion(
            type="cylinder_z",
            centers_m=centers,
            void_radius_m=(float(leg_hole_d) / 2.0) / 1000.0,
            wall_radius_m=((float(leg_hole_d) / 2.0) + leg_wall_mm) / 1000.0,
        ))

    return voids, nondesign


# ── Registry ──────────────────────────────────────────────────────────────

_SHAPE_RESOLVERS = {
    "disk": _resolve_disk,
    # add future shapes here: "hex", "annulus", etc.
}


# ── Public API ────────────────────────────────────────────────────────────

def resolve_geometry_regions(
    params: PipelineParams,
) -> Tuple[List[VoidRegion], List[NondesignRegion]]:
    """
    Return (void_regions, nondesign_regions) with any geometry-derived
    regions APPENDED to those declared in the JSON.

    Behavior:
      - If geometry.shape is absent, "box", or None → no auto-regions.
      - If geometry.shape is a known shape → its resolver runs and appends.
      - Unknown shape raises ValueError (fail fast rather than silently
        produce a rectangular optimization on a non-rectangular part).

    The original params object is not mutated; callers should use the
    returned lists directly when voxelizing.
    """
    shape = params.geometry.get("shape", None)

    declared_voids     = list(params.void_regions)
    declared_nondesign = list(params.nondesign_regions)

    if shape in (None, "box"):
        return declared_voids, declared_nondesign

    if shape not in _SHAPE_RESOLVERS:
        raise ValueError(
            f"Unknown geometry.shape '{shape}'. "
            f"Known shapes: {sorted(_SHAPE_RESOLVERS.keys())} "
            f"(or use 'box' / omit for rectangular parts)."
        )

    auto_voids, auto_nondesign = _SHAPE_RESOLVERS[shape](params.geometry)

    # Validate before returning — catches bad geometry values early.
    for r in auto_voids:
        r.validate()
    for r in auto_nondesign:
        r.validate()

    return declared_voids + auto_voids, declared_nondesign + auto_nondesign
