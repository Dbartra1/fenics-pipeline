# scripts/voxelize.py
#
# Converts continuous geometry params into flat binary masks for the Rust solver.
# Replaces generate_opt_domain.py for the Rust path.
#
# Two public functions:
#   voxelize_domain()  → (nondesign, void_mask) as uint8 arrays, shape (nz,ny,nx)
#   build_load_case()  → {"fixed_dofs": u32, "load_dofs": u32, "load_vals": f64}

import numpy as np
from src.geometry.param_schema import GeometryParams


def voxelize_domain(
    geometry_params: GeometryParams,
    grid_config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (nondesign, void_mask), each shape (nz, ny, nx), dtype uint8.
    z-major order matches Rust solver expectation.

    nondesign[e] = 1  → forced solid (hole walls)
    void_mask[e] = 1  → always void (hole interiors)
    void takes priority over nondesign.
    """
    nx = grid_config["nx"]
    ny = grid_config["ny"]
    nz = grid_config["nz"]
    h  = grid_config["voxel_size"]

    nondesign = np.zeros((nz, ny, nx), dtype=np.uint8)
    void_mask  = np.zeros((nz, ny, nx), dtype=np.uint8)

    # Element centroid coordinates (z-major indexing)
    xs = (np.arange(nx) + 0.5) * h
    ys = (np.arange(ny) + 0.5) * h
    zs = (np.arange(nz) + 0.5) * h
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing='ij')

    hole_r_m  = geometry_params.mounting_hole_diameter / 2.0 / 1000.0
    inset_m   = geometry_params.mounting_hole_inset / 1000.0
    L = geometry_params.length / 1000.0
    W = geometry_params.width  / 1000.0
    WALL_M = 0.004   # 4mm nondesign ring around each hole

    hole_centers = [
        (inset_m,     inset_m),
        (L - inset_m, inset_m),
        (inset_m,     W - inset_m),
        (L - inset_m, W - inset_m),
    ]

    for hx, hy in hole_centers:
        dist_xy = np.sqrt((X - hx)**2 + (Y - hy)**2)
        void_mask[ dist_xy <  hole_r_m]            = 1
        nondesign[ dist_xy <  hole_r_m + WALL_M]   = 1

    # void takes priority
    nondesign[void_mask == 1] = 0

    return nondesign, void_mask


def build_load_case(
    geometry_params: GeometryParams,
    load_hints: dict,
    grid_config: dict,
) -> dict:
    """
    Returns {"fixed_dofs": u32 array, "load_dofs": u32 array, "load_vals": f64 array}.

    Top-face traction (downward z), corner-disk fixed BCs at mounting holes.
    Matches the boundary_conditions.py logic used in Stage 03.
    """
    nx = grid_config["nx"]
    ny = grid_config["ny"]
    nz = grid_config["nz"]
    h  = grid_config["voxel_size"]

    # Top-face load — z DOFs of all nodes on top face (iz = nz)
    top_nodes = [
        ix + iy*(nx+1) + nz*(nx+1)*(ny+1)
        for iy in range(ny+1) for ix in range(nx+1)
    ]
    n_top  = len(top_nodes)
    load_n = float(load_hints.get("load_magnitude_n", 10000.0))
    load_dofs = np.array([3*n + 2 for n in top_nodes], dtype=np.uint32)
    load_vals = np.full(n_top, -load_n / n_top, dtype=np.float64)  # negative = downward

    # Corner-disk fixed BCs on bottom face (iz = 0)
    hole_r_m = geometry_params.mounting_hole_diameter / 2.0 / 1000.0
    inset_m  = geometry_params.mounting_hole_inset / 1000.0
    L = geometry_params.length / 1000.0
    W = geometry_params.width  / 1000.0
    disk_r   = hole_r_m + 0.002   # 2mm washer margin

    hole_centers = [
        (inset_m,     inset_m),
        (L - inset_m, inset_m),
        (inset_m,     W - inset_m),
        (L - inset_m, W - inset_m),
    ]

    fixed_nodes = []
    for iy in range(ny+1):
        for ix in range(nx+1):
            xc, yc = ix * h, iy * h
            if any(np.sqrt((xc-hx)**2 + (yc-hy)**2) < disk_r
                   for hx, hy in hole_centers):
                fixed_nodes.append(ix + iy*(nx+1))   # iz=0 bottom layer

    fixed_dofs = np.array(
        [3*n + d for n in fixed_nodes for d in range(3)],
        dtype=np.uint32
    )

    return {
        "fixed_dofs": fixed_dofs,
        "load_dofs":  load_dofs,
        "load_vals":  load_vals,
    }