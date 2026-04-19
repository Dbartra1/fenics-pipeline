# src/meshing/gmsh_pipeline.py
#
# Phase 1: Import STL into gmsh and classify surfaces
# Phase 2: Generate volumetric tetrahedral mesh
# Phase 3: Export to .msh and .xdmf for FEniCSx
#
# gmsh API note: gmsh.initialize() / gmsh.finalize() must bookend ALL gmsh
# operations in a process. Never call initialize() twice in the same kernel
# without finalizing first — it silently corrupts the model state.

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gmsh


@dataclass
class MeshResult:
    success:          bool
    msh_path:         Optional[Path]
    xdmf_path:        Optional[Path]
    n_nodes:          Optional[int]
    n_elements:       Optional[int]
    n_boundary_tris:  Optional[int]
    duration_s:       float
    warnings:         list[str]
    error:            Optional[str]

    def raise_if_failed(self) -> None:
        if not self.success:
            raise RuntimeError(f"Meshing failed: {self.error}")


def _initialize_gmsh(verbosity: int = 2) -> None:
    """
    Safe gmsh initialization.
    Verbosity: 0=silent, 2=warnings, 5=full debug.
    Always pair with _finalize_gmsh().
    """
    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", verbosity)


def _finalize_gmsh() -> None:
    if gmsh.is_initialized():
        gmsh.finalize()


def import_and_repair_stl(
    stl_path: Path,
    angle_deg: float = 20.0,
) -> tuple[bool, list[str]]:
    """
    Import STL and classify surfaces using gmsh's angle-based classifier.

    angle_deg controls the dihedral angle threshold for surface classification.
    Too low: over-segments flat faces. Too high: merges distinct surfaces.
    20 degrees is a reasonable default for mechanical parts.

    Returns (success, warnings).
    Called with gmsh already initialized.
    """
    warnings = []

    gmsh.model.add("part")
    gmsh.merge(str(stl_path))

    # Classify surfaces by dihedral angle — critical for correct BCs in Stage 3
    # This is what distinguishes top/bottom/side faces for boundary conditions
    angle_rad = angle_deg * 3.14159 / 180
    gmsh.model.mesh.classifySurfaces(
        angle_rad,
        True,
        False,
        angle_rad / 2,
    )

    # Create a geometric surface from the classified mesh
    gmsh.model.mesh.createGeometry()

    # Validate: we should have at least one surface after classification
    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        return False, ["No surfaces found after STL classification"]

    volumes = gmsh.model.getEntities(dim=3)
    if not volumes:
        # Try to create volume from classified surfaces
        surface_tags = [s[1] for s in surfaces]
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
        gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()
        volumes = gmsh.model.getEntities(dim=3)
        if not volumes:
            return False, ["Could not create volume from STL surfaces — "
                           "check for holes or non-manifold edges in the STL"]

    if len(volumes) > 1:
        warnings.append(f"Multiple volumes detected ({len(volumes)}) — "
                        f"using largest. Check base_part.scad for geometry errors.")

    return True, warnings


def configure_mesh_algorithm(
    target_size: float,
    min_size: Optional[float] = None,
    max_size: Optional[float] = None,
    algorithm_3d: int = 4,
) -> None:
    """
    Configure gmsh meshing algorithm and size constraints.

    algorithm_3d options:
        1 = Delaunay (fast, lower quality)
        4 = Frontal (slower, better quality for thin walls) ← default
        10 = HXT (parallel, fastest for large meshes)

    For the SIMP optimizer in Stage 4, mesh quality matters more than speed —
    stick with algorithm 4 unless meshes exceed ~500k elements.
    """
    gmsh.option.setNumber("Mesh.Algorithm3D", algorithm_3d)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin",
                          min_size or target_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax",
                          max_size or target_size * 2.0)

    # Optimize for FEniCSx compatibility
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)

    # Tag physical groups — required for FEniCSx to identify boundaries
    gmsh.option.setNumber("Mesh.SaveAll", 0)

    # Set global mesh size field
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), target_size)


def tag_physical_groups() -> dict[str, int]:
    """
    Tag volume and boundary surfaces as named physical groups.
    FEniCSx reads these tags to apply boundary conditions in Stage 3.

    Returns dict mapping name → physical group tag.

    Physical group strategy:
    - Volume gets tag 1 (the bulk material domain)
    - Each classified surface gets its own tag (top=2, bottom=3, sides=4+)
    - Stage 3 reads these tags from boundary_conditions.py
    """
    tags = {}

    volumes = gmsh.model.getEntities(dim=3)
    if volumes:
        vol_tag = gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes])
        gmsh.model.setPhysicalName(3, vol_tag, "volume")
        tags["volume"] = vol_tag

    surfaces = gmsh.model.getEntities(dim=2)
    if not surfaces:
        return tags

    # Classify surfaces by their bounding box centroid Z position
    # Top surface = highest Z centroid, bottom = lowest
    surface_centroids = []
    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        # bb = (xmin, ymin, zmin, xmax, ymax, zmax)
        z_center = (bb[2] + bb[5]) / 2
        surface_centroids.append((tag, z_center, bb))

    surface_centroids.sort(key=lambda x: x[1])

    # Bottom surface — fixed boundary in Stage 3
    # TAG_BOTTOM=3 must match boundary_conditions.py constant
    bot_tag = surface_centroids[0][0]
    phys_bot = gmsh.model.addPhysicalGroup(2, [bot_tag], tag=3)
    gmsh.model.setPhysicalName(2, 3, "bottom")
    tags["bottom"] = 3

    # Top surface — load application point in Stage 3
    # TAG_TOP=2 must match boundary_conditions.py constant
    top_tag = surface_centroids[-1][0]
    phys_top = gmsh.model.addPhysicalGroup(2, [top_tag], tag=2)
    gmsh.model.setPhysicalName(2, 2, "top")
    tags["top"] = 2

    # Remaining surfaces — side walls
    # TAG_SIDES=4 must match boundary_conditions.py constant
    side_tags = [s[0] for s in surface_centroids[1:-1]]
    if side_tags:
        phys_sides = gmsh.model.addPhysicalGroup(2, side_tags, tag=4)
        gmsh.model.setPhysicalName(2, 4, "sides")
        tags["sides"] = 4

    return tags


def export_mesh(
    output_dir: Path,
    part_name: str,
) -> tuple[Path, Path]:
    """
    Export to .msh (gmsh native) and .xdmf/.h5 (FEniCSx native).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    msh_path = output_dir / f"{part_name}.msh"
    gmsh.write(str(msh_path))

    try:
        import meshio
        import numpy as np
        msh = meshio.read(str(msh_path))

        phys = msh.cell_data_dict.get("gmsh:physical", {})

        # Collect ALL triangle blocks and concatenate — gmsh writes one block
        # per surface, meshio merges the data but keeps blocks separate
        tri_blocks = [b.data for b in msh.cells if b.type == "triangle"]
        tet_blocks = [b.data for b in msh.cells if b.type == "tetra"]

        if not tet_blocks:
            raise ValueError("No tetrahedral cells found in mesh")

        tet_cells = np.vstack(tet_blocks)
        tet_data  = phys.get("tetra")

        # Write volume mesh
        xdmf_path = output_dir / f"{part_name}.xdmf"
        vol_data = {}
        if tet_data is not None:
            vol_data["gmsh:physical"] = [tet_data]
        meshio.write(
            str(xdmf_path),
            meshio.Mesh(
                points=msh.points,
                cells={"tetra": tet_cells},
                cell_data=vol_data,
            )
        )

        # Write boundary mesh — concatenate all triangle blocks
        bnd_path = output_dir / f"{part_name}_boundaries.xdmf"
        if tri_blocks:
            tri_cells = np.vstack(tri_blocks)
            tri_data  = phys.get("triangle")
            bnd_data  = {}
            if tri_data is not None:
                bnd_data["gmsh:physical"] = [tri_data]
            meshio.write(
                str(bnd_path),
                meshio.Mesh(
                    points=msh.points,
                    cells={"triangle": tri_cells},
                    cell_data=bnd_data,
                )
            )

    except ImportError:
        raise RuntimeError("meshio not found")

    return msh_path, xdmf_path


def run_meshing_pipeline(
    stl_path: str | Path,
    part_name: str,
    output_dir: str | Path,
    target_element_size: float,
    verbosity: int = 2,
    angle_deg: float = 20.0,
    algorithm_3d: int = 4,
) -> MeshResult:
    """
    Full meshing pipeline: import STL → repair → mesh → tag → export.
    Always finalizes gmsh even on failure.
    """
    stl_path   = Path(stl_path)
    output_dir = Path(output_dir)
    warnings   = []
    t0         = time.perf_counter()

    try:
        _initialize_gmsh(verbosity)

        # Phase 1: import and repair
        ok, import_warnings = import_and_repair_stl(stl_path, angle_deg)
        warnings.extend(import_warnings)
        if not ok:
            return MeshResult(
                success=False, msh_path=None, xdmf_path=None,
                n_nodes=None, n_elements=None, n_boundary_tris=None,
                duration_s=round(time.perf_counter() - t0, 3),
                warnings=warnings, error=import_warnings[0],
            )

        # Phase 2: configure and generate mesh
        configure_mesh_algorithm(target_element_size, algorithm_3d=algorithm_3d)
        gmsh.model.mesh.generate(3)

        # Optimize mesh quality — reduces bad aspect ratios and flat tets
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize("")

        # Phase 3: tag physical groups for FEniCSx BCs
        physical_tags = tag_physical_groups()
        if "volume" not in physical_tags:
            warnings.append("No volume physical group — FEniCSx will not find domain")

        # Phase 4: export
        msh_path, xdmf_path = export_mesh(output_dir, part_name)

        # Collect stats
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
        n_elements = sum(len(et) for et in elem_tags)
        tri_types, tri_tags, _ = gmsh.model.mesh.getElements(dim=2)
        n_boundary_tris = sum(len(tt) for tt in tri_tags)

        return MeshResult(
            success=True,
            msh_path=msh_path,
            xdmf_path=xdmf_path,
            n_nodes=n_nodes,
            n_elements=n_elements,
            n_boundary_tris=n_boundary_tris,
            duration_s=round(time.perf_counter() - t0, 3),
            warnings=warnings,
            error=None,
        )

    except Exception as e:
        return MeshResult(
            success=False, msh_path=None, xdmf_path=None,
            n_nodes=None, n_elements=None, n_boundary_tris=None,
            duration_s=round(time.perf_counter() - t0, 3),
            warnings=warnings, error=str(e),
        )
    finally:
        _finalize_gmsh()