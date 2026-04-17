# scripts/voxelize.py
#
# Converts geometry params into flat binary masks for the Rust solver.
#
# Phase 1: GeometryParams accepts arbitrary fields.
# Phase 2: voxelize_domain() reads nondesign_regions from params if present,
#           falls back to legacy geometry_params-based hole logic if absent.
#           build_load_case() reads load_case_config from params if present,
#           falls back to legacy top-load / corner-fix behavior if absent.

import math

import numpy as np
from src.geometry.param_schema import (
    GeometryParams, LoadCaseConfig, NondesignRegion, VoidRegion
)
from src.geometry.region_factory import part_center_m


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def _centroid_grids(grid_config: dict):
    """Return (Z, Y, X) meshgrid of element centroid coordinates in metres."""
    nx = grid_config["nx"]
    ny = grid_config["ny"]
    nz = grid_config["nz"]
    h  = grid_config["voxel_size"]
    xs = (np.arange(nx) + 0.5) * h
    ys = (np.arange(ny) + 0.5) * h
    zs = (np.arange(nz) + 0.5) * h
    return np.meshgrid(zs, ys, xs, indexing='ij')


def _node_idx(ix, iy, iz, nx, ny):
    return ix + iy * (nx + 1) + iz * (nx + 1) * (ny + 1)


# ─── voxelize_domain ─────────────────────────────────────────────────────────

def voxelize_domain(
    geometry_params: GeometryParams,
    grid_config: dict,
    nondesign_regions=None,   # list[NondesignRegion] or None  (Phase 2)
    void_regions=None,        # list[VoidRegion] or None       (Phase 3)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (nondesign, void_mask), each shape (nz, ny, nx), dtype uint8.

    If nondesign_regions is provided (Phase 2 declarative path), uses those.
    Otherwise falls back to legacy corner-hole logic derived from geometry_params.
    If void_regions is provided (Phase 3), those box regions are forced void
    unconditionally — used to mask the empty space in non-rectangular parts.
    """
    nx = grid_config["nx"]
    ny = grid_config["ny"]
    nz = grid_config["nz"]
    h  = grid_config["voxel_size"]

    nondesign = np.zeros((nz, ny, nx), dtype=np.uint8)
    void_mask  = np.zeros((nz, ny, nx), dtype=np.uint8)

    Z, Y, X = _centroid_grids(grid_config)

    if nondesign_regions:
        # ── Phase 2: declarative regions ─────────────────────────────────
        for region in nondesign_regions:
            for ca, cb in region.centers_m:
                if region.type == "cylinder_z":
                    dist = np.sqrt((X - ca)**2 + (Y - cb)**2)
                elif region.type == "cylinder_x":
                    dist = np.sqrt((Y - ca)**2 + (Z - cb)**2)
                elif region.type == "cylinder_y":
                    dist = np.sqrt((X - ca)**2 + (Z - cb)**2)
                else:
                    raise ValueError(f"Unknown region type: {region.type}")

                void_mask[ dist <  region.void_radius_m]  = 1
                nondesign[ dist <  region.wall_radius_m]  = 1

    else:
        # ── Legacy: derive holes from geometry_params ─────────────────────
        hole_d = geometry_params.get("mounting_hole_diameter", 0.0)
        inset  = geometry_params.get("mounting_hole_inset", 0.0)

        if hole_d and inset:
            hole_r_m = hole_d / 2.0 / 1000.0
            inset_m  = inset / 1000.0
            L = geometry_params.length / 1000.0
            W = geometry_params.width  / 1000.0

            HOLE_CLEARANCE = 0.0015
            WALL_M         = 0.006

            hole_centers = [
                (inset_m,     inset_m),
                (L - inset_m, inset_m),
                (inset_m,     W - inset_m),
                (L - inset_m, W - inset_m),
            ]
            for hx, hy in hole_centers:
                dist_xy = np.sqrt((X - hx)**2 + (Y - hy)**2)
                void_mask[ dist_xy <  hole_r_m + HOLE_CLEARANCE]            = 1
                nondesign[ dist_xy <  hole_r_m + HOLE_CLEARANCE + WALL_M]   = 1

    # ── Phase 3: void regions (box or cylinder_z_exterior) ────────────────
    if void_regions:
        for region in void_regions:
            if region.type == "box":
                # Build a boolean mask — None on any bound means unbounded
                mask = np.ones((nz, ny, nx), dtype=bool)
                if region.x_min is not None:
                    mask &= (X >= region.x_min)
                if region.x_max is not None:
                    mask &= (X <= region.x_max)
                if region.y_min is not None:
                    mask &= (Y >= region.y_min)
                if region.y_max is not None:
                    mask &= (Y <= region.y_max)
                if region.z_min is not None:
                    mask &= (Z >= region.z_min)
                if region.z_max is not None:
                    mask &= (Z <= region.z_max)

            elif region.type == "cylinder_z_exterior":
                # Everything OUTSIDE the cylinder becomes void.
                r2 = (X - region.cx) ** 2 + (Y - region.cy) ** 2
                mask = r2 > (region.radius ** 2)

            else:
                raise ValueError(f"Unknown VoidRegion type: {region.type}")

            void_mask[mask] = 1
            nondesign[mask] = 0   # void takes priority

    # void always takes priority over nondesign (catches any remaining overlap)
    nondesign[void_mask == 1] = 0

    return nondesign, void_mask


# ─── build_load_case ─────────────────────────────────────────────────────────

def build_load_case(
    geometry_params: GeometryParams,
    load_hints,
    grid_config: dict,
    load_case_config=None,    # LoadCaseConfig or None
) -> dict:
    """
    Returns {"fixed_dofs": u32, "load_dofs": u32, "load_vals": f64}.

    If load_case_config is provided (Phase 2), uses declarative face selection.
    Otherwise falls back to legacy top-load / corner-fix behavior.
    """
    nx = grid_config["nx"]
    ny = grid_config["ny"]
    nz = grid_config["nz"]
    h  = grid_config["voxel_size"]

    if load_case_config is not None:
        # ── Phase 2: declarative load case ───────────────────────────────
        fixed_dofs = _fixed_dofs_from_config(
            load_case_config.fixed, geometry_params, nx, ny, nz, h
        )
        load_dofs, load_vals = _load_dofs_from_config(
            load_case_config.load, geometry_params, nx, ny, nz, h
        )
    else:
        # ── Legacy: top-load, corner-fix ─────────────────────────────────
        load_n = float(load_hints.load_magnitude_n)

        # Load: z DOFs on top face (iz=nz)
        top_nodes = [
            _node_idx(ix, iy, nz, nx, ny)
            for iy in range(ny + 1) for ix in range(nx + 1)
        ]
        n_top = len(top_nodes)
        load_dofs = np.array([3 * n + 2 for n in top_nodes], dtype=np.uint32)
        load_vals = np.full(n_top, -load_n / n_top, dtype=np.float64)

        # Fixed: corner disks on bottom face (iz=0)
        hole_d = geometry_params.get("mounting_hole_diameter", 0.0)
        inset  = geometry_params.get("mounting_hole_inset", 0.0)
        hole_r_m = (hole_d / 2.0 / 1000.0) if hole_d else 0.003
        inset_m  = inset / 1000.0 if inset else 0.010
        L = geometry_params.length / 1000.0
        W = geometry_params.width  / 1000.0
        disk_r = hole_r_m + 0.002

        hole_centers = [
            (inset_m,     inset_m),
            (L - inset_m, inset_m),
            (inset_m,     W - inset_m),
            (L - inset_m, W - inset_m),
        ]
        fixed_nodes = []
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                xc, yc = ix * h, iy * h
                if any(np.sqrt((xc - hx)**2 + (yc - hy)**2) < disk_r
                       for hx, hy in hole_centers):
                    fixed_nodes.append(_node_idx(ix, iy, 0, nx, ny))

        fixed_dofs = np.array(
            [3 * n + d for n in fixed_nodes for d in range(3)],
            dtype=np.uint32,
        )

    return {
        "fixed_dofs": fixed_dofs,
        "load_dofs":  load_dofs,
        "load_vals":  load_vals,
    }


# ─── Internal helpers for declarative load case ───────────────────────────────

def _face_nodes(face: str, nx: int, ny: int, nz: int):
    """
    Return list of node indices on the named face.
    Face names: x_min, x_max, y_min, y_max, z_min, z_max
    """
    nodes = []
    if face == "z_min":
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, iy, 0, nx, ny))
    elif face == "z_max":
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, iy, nz, nx, ny))
    elif face == "x_min":
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                nodes.append(_node_idx(0, iy, iz, nx, ny))
    elif face == "x_max":
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                nodes.append(_node_idx(nx, iy, iz, nx, ny))
    elif face == "y_min":
        for iz in range(nz + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, 0, iz, nx, ny))
    elif face == "y_max":
        for iz in range(nz + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, ny, iz, nx, ny))
    else:
        raise ValueError(f"Unknown face: '{face}'")
    return nodes


def _face_node_coords(face: str, nx: int, ny: int, nz: int, h: float):
    """Return (node_indices, coord_a, coord_b) for the two in-plane axes of a face."""
    nodes = []
    ca_list = []
    cb_list = []

    if face == "z_min":
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, iy, 0, nx, ny))
                ca_list.append(ix * h)
                cb_list.append(iy * h)
    elif face == "z_max":
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, iy, nz, nx, ny))
                ca_list.append(ix * h)
                cb_list.append(iy * h)
    elif face == "x_min":
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                nodes.append(_node_idx(0, iy, iz, nx, ny))
                ca_list.append(iy * h)
                cb_list.append(iz * h)
    elif face == "x_max":
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                nodes.append(_node_idx(nx, iy, iz, nx, ny))
                ca_list.append(iy * h)
                cb_list.append(iz * h)
    elif face == "y_min":
        for iz in range(nz + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, 0, iz, nx, ny))
                ca_list.append(ix * h)
                cb_list.append(iz * h)
    elif face == "y_max":
        for iz in range(nz + 1):
            for ix in range(nx + 1):
                nodes.append(_node_idx(ix, ny, iz, nx, ny))
                ca_list.append(ix * h)
                cb_list.append(iz * h)
    else:
        raise ValueError(f"Unknown face: '{face}'")

    return (np.array(nodes),
            np.array(ca_list, dtype=np.float64),
            np.array(cb_list, dtype=np.float64))


def _fixed_dofs_from_config(cfg, geom, nx, ny, nz, h):
    """Build fixed DOF array from FixedFaceConfig.

    Args:
        cfg:  FixedFaceConfig (face, selector, inset_m, disk_radius_m)
        geom: GeometryParams  (needed by 'leg_holes' selector)
        nx, ny, nz, h: grid dimensions and voxel size
    """
    nodes_arr, ca, cb = _face_node_coords(cfg.face, nx, ny, nz, h)

    if cfg.selector == "full":
        selected = nodes_arr
    elif cfg.selector == "corners":
        # Find the 4 corners of the face bounding box
        a_min, a_max = ca.min(), ca.max()
        b_min, b_max = cb.min(), cb.max()
        corners = [
            (a_min + cfg.inset_m, b_min + cfg.inset_m),
            (a_max - cfg.inset_m, b_min + cfg.inset_m),
            (a_min + cfg.inset_m, b_max - cfg.inset_m),
            (a_max - cfg.inset_m, b_max - cfg.inset_m),
        ]
        mask = np.zeros(len(nodes_arr), dtype=bool)
        for corner_a, corner_b in corners:
            dist = np.sqrt((ca - corner_a)**2 + (cb - corner_b)**2)
            mask |= dist < cfg.disk_radius_m
        selected = nodes_arr[mask]
    elif cfg.selector == "leg_holes":
        # N disks at polar positions around the part centre.
        # Geometry fields: leg_hole_radius (mm), num_legs, first_leg_angle (deg)
        cx_m, cy_m = part_center_m(geom)
        num_legs        = int(geom.get("num_legs", 3))
        leg_r_m         = float(geom.get("leg_hole_radius")) / 1000.0
        first_angle_deg = float(geom.get("first_leg_angle", 90.0))

        centers = []
        for i in range(num_legs):
            theta = math.radians(first_angle_deg + i * (360.0 / num_legs))
            centers.append((cx_m + leg_r_m * math.cos(theta),
                            cy_m + leg_r_m * math.sin(theta)))

        mask = np.zeros(len(nodes_arr), dtype=bool)
        for lx, ly in centers:
            dist = np.sqrt((ca - lx)**2 + (cb - ly)**2)
            mask |= dist < cfg.disk_radius_m
        selected = nodes_arr[mask]
    else:
        raise ValueError(f"Unknown selector: '{cfg.selector}'")

    return np.array(
        [3 * int(n) + d for n in selected for d in range(3)],
        dtype=np.uint32,
    )


def _load_dofs_from_config(cfg, geom, nx, ny, nz, h):
    """Build load DOF and value arrays from LoadFaceConfig.

    Args:
        cfg:  LoadFaceConfig (face, selector, direction, magnitude_n, disk_radius_m)
        geom: GeometryParams (needed by 'center_disk' selector)
        nx, ny, nz, h: grid dimensions and voxel size
    """
    nodes_arr, ca, cb = _face_node_coords(cfg.face, nx, ny, nz, h)

    # Select which nodes receive load
    if cfg.selector == "full":
        selected = nodes_arr
    elif cfg.selector == "center_disk":
        cx_m, cy_m = part_center_m(geom)
        dist = np.sqrt((ca - cx_m)**2 + (cb - cy_m)**2)
        mask = dist < cfg.disk_radius_m
        selected = nodes_arr[mask]
    else:
        raise ValueError(f"Unknown load selector: '{cfg.selector}'")

    n_selected = len(selected)
    assert n_selected > 0, (
        f"No nodes selected for load (selector={cfg.selector!r}, "
        f"disk_radius_m={cfg.disk_radius_m}, face={cfg.face}). "
        f"Check that disk_radius_m is large enough relative to voxel size."
    )

    # Normalise direction vector
    direction = np.array(cfg.direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    # Apply force to all 3 DOFs per node, proportional to direction
    force_per_node = direction * cfg.magnitude_n / n_selected

    dofs = []
    vals = []
    for n in selected:
        for axis in range(3):
            if abs(force_per_node[axis]) > 1e-12:
                dofs.append(3 * int(n) + axis)
                vals.append(force_per_node[axis])

    return (
        np.array(dofs, dtype=np.uint32),
        np.array(vals, dtype=np.float64),
    )