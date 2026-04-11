"""
generate_opt_domain.py
----------------------
Generates the optimization domain mesh for Stage 4 SIMP.

Design domain strategy:
  - Box = full part envelope
  - Mounting holes = permanent voids (subtracted from mesh)
  - Non-design regions = regions that must stay solid:
      * Thin shell around each hole (material that holds the fastener)
      * Outer surface shell of the part
  - Design region = everything else (optimizer can remove freely)

The mesh tags used:
  Tag 1 = design volume       (optimizer controls density here)
  Tag 2 = top face
  Tag 3 = bottom face
  Tag 4 = side faces

Usage:
    python3 generate_opt_domain.py [params_json_path]

Reads geometry from scad/params.json by default.
Writes to outputs/meshes/opt_domain.xdmf and opt_domain_boundaries.xdmf
"""

import sys
import json
import numpy as np
from pathlib import Path

import gmsh
import meshio


def generate_opt_domain(
    params_path: str = "scad/params.json",
    output_dir:  str = "outputs/meshes",
    element_size_m: float = None,   # if None, reads from params mesh_hints.opt_domain_element_size_mm
    hole_wall_thickness_m: float = 0.004,  # 4mm non-design ring around each hole
):
    params    = json.loads(Path(params_path).read_text())
    geom      = params["geometry"]
    bc_params = params.get("boundary_conditions", {})

    # Resolve element size — params is the source of truth, arg allows override
    if element_size_m is None:
        element_size_mm = params.get("mesh_hints", {}).get(
            "opt_domain_element_size_mm", 2.5
        )
        element_size_m = element_size_mm / 1000.0
    print(f"Opt domain element size: {element_size_m*1000:.1f}mm")

    # Part dimensions (convert mm → m)
    L = geom["length"]           / 1000.0
    W = geom["width"]            / 1000.0
    H = geom["height"]           / 1000.0
    hole_r     = geom["mounting_hole_diameter"] / 2.0 / 1000.0
    hole_inset = geom["mounting_hole_inset"]          / 1000.0
    hole_wall  = hole_wall_thickness_m

    # Hole center positions (XY plane, at Z=H/2 for cylinder axis)
    hole_centers = [
        (hole_inset,     hole_inset),
        (L - hole_inset, hole_inset),
        (hole_inset,     W - hole_inset),
        (L - hole_inset, W - hole_inset),
    ]

    print(f"Part dimensions: {L*1000:.0f} x {W*1000:.0f} x {H*1000:.0f} mm")
    print(f"Holes: {len(hole_centers)} x ⌀{hole_r*2000:.1f}mm "
          f"at {hole_inset*1000:.1f}mm inset")
    print(f"Non-design ring: {hole_wall*1000:.1f}mm around each hole")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("opt_domain")

    # ── Step 1: Create main box ───────────────────────────────────────────
    box_tag = gmsh.model.occ.addBox(0, 0, 0, L, W, H)

    # ── Step 2: Create hole cylinders (through entire height) ─────────────
    hole_cyls = []
    for (hx, hy) in hole_centers:
        cyl = gmsh.model.occ.addCylinder(hx, hy, -H*0.01, 0, 0, H*1.02, hole_r)
        hole_cyls.append(cyl)

    # ── Step 3: Create non-design rings around holes ──────────────────────
    # Each ring is an annulus cylinder: outer radius = hole_r + wall_thickness
    ring_cyls = []
    for (hx, hy) in hole_centers:
        outer = gmsh.model.occ.addCylinder(
            hx, hy, -H*0.1, 0, 0, H*1.2, hole_r + hole_wall
        )
        ring_cyls.append(outer)

    # ── Step 4: Boolean operations ────────────────────────────────────────
    # Step 4a: Cut holes from box → design_domain
    design_vols, _ = gmsh.model.occ.cut(
        [(3, box_tag)],
        [(3, c) for c in hole_cyls],
        removeObject=True,
        removeTool=True,
    )

    # Step 4b: Create ring volumes = (ring cylinders) minus (hole cylinders)
    # We need fresh hole cylinders since the originals were consumed
    fresh_holes = []
    for (hx, hy) in hole_centers:
        cyl = gmsh.model.occ.addCylinder(hx, hy, -H*0.1, 0, 0, H*1.2, hole_r)
        fresh_holes.append(cyl)

    ring_vols_list = []
    for ring_cyl, hole_cyl in zip(ring_cyls, fresh_holes):
        ring_vol, _ = gmsh.model.occ.cut(
            [(3, ring_cyl)],
            [(3, hole_cyl)],
            removeObject=True,
            removeTool=True,
        )
        ring_vols_list.extend(ring_vol)

    # Step 4c: Fragment everything so shared faces are conforming
    all_vols = design_vols + ring_vols_list
    gmsh.model.occ.fragment(all_vols, [])
    gmsh.model.occ.synchronize()

    # ── Step 5: Identify volumes by centroid ──────────────────────────────
    all_vols_after = gmsh.model.getEntities(dim=3)

    design_tags    = []
    nondesign_tags = []

    for dim, tag in all_vols_after:
        xc, yc, zc = gmsh.model.occ.getCenterOfMass(dim, tag)

        # Check if centroid is near any hole center (within hole_r + wall/2)
        near_hole = any(
            np.sqrt((xc - hx)**2 + (yc - hy)**2) < hole_r + hole_wall * 0.6
            for hx, hy in hole_centers
        )

        if near_hole:
            nondesign_tags.append(tag)
        else:
            design_tags.append(tag)

    print(f"Design volumes:     {len(design_tags)}")
    print(f"Non-design volumes: {len(nondesign_tags)}")

    # ── Step 6: Physical groups ───────────────────────────────────────────
    if design_tags:
        gmsh.model.addPhysicalGroup(3, design_tags, tag=1)
        gmsh.model.setPhysicalName(3, 1, "design")

    if nondesign_tags:
        gmsh.model.addPhysicalGroup(3, nondesign_tags, tag=2)
        gmsh.model.setPhysicalName(3, 2, "nondesign")

    # Tag boundary surfaces by face position
    all_surfs = gmsh.model.getEntities(dim=2)
    bottom_surfs = []
    top_surfs    = []
    side_surfs   = []

    for dim, tag in all_surfs:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_lo, z_hi = bb[2], bb[5]
        x_lo, x_hi = bb[0], bb[3]
        y_lo, y_hi = bb[1], bb[4]

        if z_hi < H * 0.05:
            bottom_surfs.append(tag)
        elif z_lo > H * 0.95:
            top_surfs.append(tag)
        elif (x_lo < 1e-6 or x_hi > L - 1e-6 or
              y_lo < 1e-6 or y_hi > W - 1e-6):
            side_surfs.append(tag)

    # Tags must match TAG_TOP=2, TAG_BOTTOM=3, TAG_SIDES=4 in boundary_conditions.py
    if bottom_surfs:
        gmsh.model.addPhysicalGroup(2, bottom_surfs, tag=3)
        gmsh.model.setPhysicalName(2, 3, "bottom")
    if top_surfs:
        gmsh.model.addPhysicalGroup(2, top_surfs, tag=2)
        gmsh.model.setPhysicalName(2, 2, "top")
    if side_surfs:
        gmsh.model.addPhysicalGroup(2, side_surfs, tag=4)
        gmsh.model.setPhysicalName(2, 4, "sides")

    # ── Step 7: Mesh ──────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size_m)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size_m)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    gmsh.model.mesh.optimize("")   # Laplacian pass — improves aspect ratios post-Netgen

    msh_path = str(Path(output_dir) / "opt_domain.msh")
    gmsh.write(msh_path)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    _, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
    n_elem = sum(len(e) for e in elem_tags)
    print(f"Nodes:    {len(node_tags):,}")
    print(f"Elements: {n_elem:,}")
    gmsh.finalize()

    # ── Step 8: Export to XDMF via meshio ────────────────────────────────
    msh = meshio.read(msh_path)
    phys = msh.cell_data_dict.get("gmsh:physical", {})

    # Volume mesh
    tet_blocks = [b.data for b in msh.cells if b.type == "tetra"]
    if not tet_blocks:
        raise RuntimeError("No tetrahedral cells found — meshing failed")
    tet_data = np.vstack(tet_blocks)
    tet_phys = phys.get("tetra", np.ones(len(tet_data), dtype=int))

    xdmf_path = str(Path(output_dir) / "opt_domain.xdmf")
    meshio.write(xdmf_path, meshio.Mesh(
        points=msh.points,
        cells={"tetra": tet_data},
        cell_data={"gmsh:physical": [tet_phys]},
    ))
    print(f"Written: {xdmf_path}")

    # Boundary mesh
    tri_blocks = [b.data for b in msh.cells if b.type == "triangle"]
    if tri_blocks:
        tri_data = np.vstack(tri_blocks)
        tri_phys = phys.get("triangle", np.ones(len(tri_data), dtype=int))
        bnd_path = str(Path(output_dir) / "opt_domain_boundaries.xdmf")
        meshio.write(bnd_path, meshio.Mesh(
            points=msh.points,
            cells={"triangle": tri_data},
            cell_data={"gmsh:physical": [tri_phys]},
        ))
        print(f"Written: {bnd_path}")

    # Report design vs non-design element counts
    design_elems    = (tet_phys == 1).sum()
    nondesign_elems = (tet_phys == 2).sum()
    print(f"\nDesign elements:     {design_elems:,} "
          f"({100*design_elems/len(tet_data):.1f}%)")
    print(f"Non-design elements: {nondesign_elems:,} "
          f"({100*nondesign_elems/len(tet_data):.1f}%)")
    print("\nDone — opt_domain mesh ready for Stage 4")
    return xdmf_path


if __name__ == "__main__":
    params_path = sys.argv[1] if len(sys.argv) > 1 else "scad/params.json"
    generate_opt_domain(params_path)
