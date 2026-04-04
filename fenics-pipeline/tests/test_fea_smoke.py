# tests/test_fea_smoke.py
#
# Smoke tests for FEA solver and boundary conditions.
# Builds a minimal mesh programmatically — no dependency on pipeline outputs.
#
# These tests verify:
#   1. The dolfinx import chain works (MPI, petsc4py, ufl all load correctly)
#   2. A trivial linear elasticity problem assembles and solves without error
#   3. The solution is physically plausible (nonzero displacement, correct sign)
#   4. boundary_conditions.py correctly maps tags to DOFs

import sys
sys.path.insert(0, "/workspace")

import numpy as np
import pytest
from mpi4py import MPI
from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def unit_cube_mesh_and_tags(tmp_path_factory):
    """
    Build a unit cube mesh with tagged boundaries using gmsh,
    export to XDMF, and return paths.
    This is the minimal mesh that exercises the full FEA stack.
    """
    import gmsh
    import meshio

    tmp = tmp_path_factory.mktemp("fea")

    if gmsh.is_initialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("smoke_cube")
    gmsh.model.occ.addBox(0, 0, 0, 10, 10, 10)
    gmsh.model.occ.synchronize()

    # Tag volume
    gmsh.model.addPhysicalGroup(3, [1], tag=1)
    gmsh.model.setPhysicalName(3, 1, "volume")

    # Tag surfaces by Z position: bottom=3, top=2, sides=4
    surfaces = gmsh.model.getEntities(dim=2)
    for dim, tag in surfaces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        z_center = (bb[2] + bb[5]) / 2
        if z_center < 0.1:
            gmsh.model.addPhysicalGroup(2, [tag], tag=3)
            gmsh.model.setPhysicalName(2, 3, "bottom")
        elif z_center > 9.9:
            gmsh.model.addPhysicalGroup(2, [tag], tag=2)
            gmsh.model.setPhysicalName(2, 2, "top")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 3.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 3.0)
    gmsh.model.mesh.generate(3)

    msh_path = tmp / "smoke.msh"
    gmsh.write(str(msh_path))
    gmsh.finalize()

    # Convert to XDMF
    msh = meshio.read(str(msh_path))
    tet_block = next(b for b in msh.cells if b.type == "tetra")
    tri_block = next(b for b in msh.cells if b.type == "triangle")

    xdmf_path = tmp / "smoke.xdmf"
    meshio.write(str(xdmf_path), meshio.Mesh(
        points=msh.points,
        cells={"tetra": tet_block.data},
        cell_data={"gmsh:physical": [
            msh.cell_data_dict["gmsh:physical"].get("tetra",
            np.ones(len(tet_block.data), dtype=int))
        ]},
    ))

    bnd_path = tmp / "smoke_boundaries.xdmf"
    meshio.write(str(bnd_path), meshio.Mesh(
        points=msh.points,
        cells={"triangle": tri_block.data},
        cell_data={"gmsh:physical": [
            msh.cell_data_dict["gmsh:physical"].get("triangle",
            np.ones(len(tri_block.data), dtype=int))
        ]},
    ))

    return xdmf_path, bnd_path


# ── Import chain tests ────────────────────────────────────────────────────────

class TestImportChain:
    def test_dolfinx_imports(self):
        import dolfinx
        assert hasattr(dolfinx, "__version__")

    def test_mpi4py_functional(self):
        comm = MPI.COMM_WORLD
        assert comm.Get_size() >= 1
        assert comm.Get_rank() == 0

    def test_petsc4py_imports(self):
        from petsc4py import PETSc
        assert PETSc.COMM_WORLD.size >= 1

    def test_ufl_imports(self):
        import ufl
        assert hasattr(ufl, "inner")

    def test_src_fea_imports(self):
        from src.fea.boundary_conditions import (
            build_boundary_conditions,
            TAG_TOP, TAG_BOTTOM, TAG_SIDES,
        )
        assert TAG_BOTTOM == 3
        assert TAG_TOP    == 2

    def test_src_solver_imports(self):
        from src.fea.solver import run_fea, DEFAULT_MATERIAL
        assert DEFAULT_MATERIAL["youngs_modulus_pa"] == 210e9


# ── Boundary condition tests ──────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_dirichlet_bc_nonzero_dofs(self, unit_cube_mesh_and_tags):
        import dolfinx
        import dolfinx.fem as fem
        import ufl
        from dolfinx.io import XDMFFile
        from src.fea.boundary_conditions import (
            load_boundary_mesh, build_dirichlet_bc, TAG_BOTTOM
        )

        xdmf_path, bnd_path = unit_cube_mesh_and_tags
        comm = MPI.COMM_WORLD

        with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )

        facet_tags = load_boundary_mesh(str(bnd_path), domain)
        V  = fem.FunctionSpace(domain, ufl.VectorElement("CG", domain.ufl_cell(), 1))
        bcs = build_dirichlet_bc(V, facet_tags, fixed_tag=TAG_BOTTOM)

        assert len(bcs) == 1, "Should produce exactly one DirichletBC"
        # DOF array should be nonempty
        assert len(bcs[0]._cpp_object.dof_indices()[0]) > 0, \
            "Dirichlet BC should constrain at least some DOFs"

    def test_traction_tag_matches_load_hints(self, unit_cube_mesh_and_tags):
        import dolfinx
        import dolfinx.fem as fem
        import ufl
        from dolfinx.io import XDMFFile
        from src.fea.boundary_conditions import (
            load_boundary_mesh, build_traction_bc,
            TAG_TOP, TAG_BOTTOM
        )

        xdmf_path, bnd_path = unit_cube_mesh_and_tags
        comm = MPI.COMM_WORLD

        with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")

        facet_tags = load_boundary_mesh(str(bnd_path), domain)

        load_hints = {"primary_face": "top", "load_magnitude_n": 500.0}
        tag, vec, ds = build_traction_bc(domain, facet_tags, load_hints)

        assert tag == TAG_TOP,  f"Expected tag {TAG_TOP}, got {tag}"
        assert vec[2] < 0,      "Load should be in -Z direction (compression)"
        assert abs(vec[2]) > 0, "Traction magnitude should be nonzero"


# ── Full solve smoke test ─────────────────────────────────────────────────────

class TestFEASolve:
    def test_trivial_solve_completes(self, unit_cube_mesh_and_tags):
        """
        Full solve on a tiny cube — verifies the entire FEA stack
        without needing pipeline outputs.
        """
        from src.fea.solver import run_fea

        xdmf_path, bnd_path = unit_cube_mesh_and_tags

        result = run_fea(
            xdmf_path=xdmf_path,
            boundaries_xdmf=bnd_path,
            part_name="smoke_test",
            output_dir=xdmf_path.parent,
            load_hints={
                "primary_face":     "top",
                "load_magnitude_n": 1000.0,
            },
            material={
                "youngs_modulus_pa": 210e9,
                "poissons_ratio":    0.3,
                "name":              "steel",
            },
        )

        assert result.success, \
            f"FEA solve failed: {result.error}"
        assert result.u_max_mm is not None
        assert result.u_max_mm > 0, \
            "Max displacement should be positive under load"
        assert result.von_mises_max > 0, \
            "Von Mises stress should be positive"
        assert result.n_dofs > 0

    def test_solve_result_physically_plausible(self, unit_cube_mesh_and_tags):
        """
        A 10mm steel cube under 1000N should deflect on the order of microns.
        This guards against unit errors (m vs mm) in the solver.
        """
        from src.fea.solver import run_fea

        xdmf_path, bnd_path = unit_cube_mesh_and_tags

        result = run_fea(
            xdmf_path=xdmf_path,
            boundaries_xdmf=bnd_path,
            part_name="smoke_plausible",
            output_dir=xdmf_path.parent,
            load_hints={
                "primary_face":     "top",
                "load_magnitude_n": 1000.0,
            },
            material={
                "youngs_modulus_pa": 210e9,
                "poissons_ratio":    0.3,
                "name":              "steel",
            },
        )

        assert result.success
        # 10mm cube, steel, 1000N: deflection should be < 1mm and > 0.0001mm
        assert result.u_max_mm < 1.0, \
            f"Deflection {result.u_max_mm}mm suspiciously large — check units"
        assert result.u_max_mm > 1e-4, \
            f"Deflection {result.u_max_mm}mm suspiciously small — check units"

    def test_higher_load_gives_higher_stress(self, unit_cube_mesh_and_tags):
        """Linearity check: doubling the load should double the stress."""
        from src.fea.solver import run_fea

        xdmf_path, bnd_path = unit_cube_mesh_and_tags

        def solve(load):
            return run_fea(
                xdmf_path=xdmf_path,
                boundaries_xdmf=bnd_path,
                part_name=f"smoke_load_{int(load)}",
                output_dir=xdmf_path.parent,
                load_hints={"primary_face": "top", "load_magnitude_n": load},
                material={"youngs_modulus_pa": 210e9, "poissons_ratio": 0.3,
                          "name": "steel"},
            )

        r1 = solve(500.0)
        r2 = solve(1000.0)

        assert r1.success and r2.success
        ratio = r2.von_mises_max / r1.von_mises_max
        assert ratio == pytest.approx(2.0, rel=0.05), \
            f"Stress should scale linearly with load. Got ratio {ratio:.3f}"