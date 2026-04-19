# src/fea/solver.py
#
# Linear elasticity FEA solver using FEniCSx + PETSc.
#
# Weak form: find u in V such that a(u,v) = L(v) for all v in V
#   a(u,v) = integral of sigma(u) : epsilon(v) dV     (bilinear — stiffness)
#   L(v)   = integral of T · v dS                     (linear — traction load)
#
# PETSc solver note: we use the direct LU solver (MUMPS) for robustness.
# For meshes > 1M DOFs, switch to iterative (CG + AMG) via petsc_options.
# MUMPS is already available in the dolfinx Docker image.

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.io as io
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc

from src.fea.boundary_conditions import BoundaryConditionSet


@dataclass
class SolverResult:
    success:          bool
    displacement_path: Optional[Path]
    stress_path:       Optional[Path]
    u_max_mm:          Optional[float]   # max displacement magnitude in mm
    von_mises_max:     Optional[float]   # max von Mises stress in MPa
    von_mises_mean:    Optional[float]
    n_dofs:            Optional[int]
    duration_s:        float
    error:             Optional[str]

    def raise_if_failed(self) -> None:
        if not self.success:
            raise RuntimeError(f"FEA solve failed: {self.error}")


# ── Material parameters (SI units throughout) ─────────────────────────────────
# Default: structural steel. Override by passing material_props to run_fea().
DEFAULT_MATERIAL = {
    "youngs_modulus_pa": 210e9,   # 210 GPa
    "poissons_ratio":    0.3,
    "name":              "steel",
}


def _build_function_space(domain: dolfinx.mesh.Mesh) -> fem.FunctionSpace:
    """
    Vector CG1 (continuous Galerkin, linear) function space.
    CG1 = trilinear on tets — matches the mesh order from gmsh.
    CG2 would give better stress accuracy but 8x more DOFs — not worth it
    for the SIMP stage which operates on element-averaged quantities anyway.
    """
    return fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))


def _build_weak_form(
    V: fem.FunctionSpace,
    domain: dolfinx.mesh.Mesh,
    bc_set: BoundaryConditionSet,
    material: dict,
) -> tuple[ufl.Form, ufl.Form]:
    """
    Construct the bilinear form a(u,v) and linear form L(v).

    sigma(u) = lambda * tr(eps(u)) * I + 2 * mu * eps(u)   (Hooke's law)
    eps(u)   = 0.5 * (grad(u) + grad(u).T)                 (small strain)
    """
    E  = material["youngs_modulus_pa"]
    nu = material["poissons_ratio"]

    # Lamé parameters
    lam = fem.Constant(domain, E * nu / ((1 + nu) * (1 - 2 * nu)))
    mu  = fem.Constant(domain, E / (2 * (1 + nu)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(len(w)) + 2 * mu * eps(w)

    # Bilinear form — stiffness
    dx = ufl.Measure("dx", domain=domain)
    a  = ufl.inner(sigma(u), eps(v)) * dx

    # Linear form — traction load on primary face
    T  = fem.Constant(domain, bc_set.traction_vec)
    ds = bc_set.ds
    L  = ufl.inner(T, v) * ds(bc_set.traction_tag)

    return a, L


def _compute_von_mises(
    u: fem.Function,
    domain: dolfinx.mesh.Mesh,
    material: dict,
) -> fem.Function:
    """
    Project von Mises stress onto a scalar DG0 (discontinuous, piecewise constant)
    function space — one value per element.

    DG0 is used rather than CG1 because:
    1. Von Mises is derived from stress which is discontinuous at element boundaries
    2. Stage 4 (SIMP) needs per-element density fields — DG0 matches this naturally
    3. Projection to CG1 artificially smooths stress concentrations
    """
    E  = material["youngs_modulus_pa"]
    nu = material["poissons_ratio"]
    lam_val = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_val  = E / (2 * (1 + nu))

    lam = fem.Constant(domain, lam_val)
    mu  = fem.Constant(domain, mu_val)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(len(w)) + 2 * mu * eps(w)

    s    = sigma(u) - (1/3) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    von_mises_expr = ufl.sqrt(3/2 * ufl.inner(s, s))

    # DG0 function space — piecewise constant per element
    W   = fem.functionspace(domain, ("DG", 0))
    vm  = fem.Function(W, name="von_mises")
    vm_expr = fem.Expression(von_mises_expr, W.element.interpolation_points())
    vm.interpolate(vm_expr)

    return vm


def run_fea(
    xdmf_path: str | Path,
    boundaries_xdmf: str | Path,
    part_name: str,
    output_dir: str | Path,
    load_hints: dict,
    material: Optional[dict] = None,
    petsc_options: Optional[dict] = None,
) -> SolverResult:
    """
    Full FEA pipeline: load mesh → build BCs → assemble → solve → export.

    petsc_options: dict of PETSc solver options.
    Default: MUMPS direct LU — robust for meshes up to ~500k DOFs.
    For larger meshes pass:
        {"ksp_type": "cg", "pc_type": "hypre", "pc_hypre_type": "boomeramg"}
    """
    material     = material or DEFAULT_MATERIAL
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    petsc_options = petsc_options or {
        "ksp_type":  "preonly",
        "pc_type":   "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    t0 = time.perf_counter()

    try:
        comm = MPI.COMM_WORLD

        # ── Load mesh ───────────────────────────────────────────────────────
        with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )

        # Convert mesh coordinates from mm to m — gmsh outputs in mm,
        # FEniCSx assumes SI units (metres) throughout
        
        domain.geometry.x[:] /= 1000.0

        # ── Load boundary tags ──────────────────────────────────────────────
        from src.fea.boundary_conditions import (
            load_boundary_mesh, build_boundary_conditions
        )
        facet_tags = load_boundary_mesh(str(boundaries_xdmf), domain)

        # ── Function space and BCs ──────────────────────────────────────────
        V      = _build_function_space(domain)
        bc_set = build_boundary_conditions(V, domain, facet_tags, load_hints)

        n_dofs = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        print(f"  DOFs: {n_dofs:,}")

        # ── Weak form ───────────────────────────────────────────────────────
        a, L = _build_weak_form(V, domain, bc_set, material)

        # ── Assemble and solve ──────────────────────────────────────────────
        problem = fem_petsc.LinearProblem(
            a, L,
            bcs=bc_set.dirichlet,
            petsc_options=petsc_options,
        )
        u = problem.solve()
        u.name = "displacement"

        # ── Von Mises stress ────────────────────────────────────────────────
        vm = _compute_von_mises(u, domain, material)

        # ── Export ──────────────────────────────────────────────────────────
        disp_path = output_dir / f"{part_name}_displacement.xdmf"
        with io.XDMFFile(comm, str(disp_path), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(u)

        stress_path = output_dir / f"{part_name}_stress.xdmf"
        with io.XDMFFile(comm, str(stress_path), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(vm)

        # ── Summary stats ────────────────────────────────────────────────────
        u_array  = u.x.array.reshape(-1, domain.geometry.dim)
        u_mag    = np.linalg.norm(u_array, axis=1)
        vm_array = vm.x.array

        return SolverResult(
            success=True,
            displacement_path=disp_path,
            stress_path=stress_path,
            u_max_mm=round(float(u_mag.max()) * 1000, 4),   # m → mm
            von_mises_max=round(float(vm_array.max()) / 1e6, 3),   # Pa → MPa
            von_mises_mean=round(float(vm_array.mean()) / 1e6, 3),
            n_dofs=n_dofs,
            duration_s=round(time.perf_counter() - t0, 3),
            error=None,
        )

    except Exception as e:
        return SolverResult(
            success=False,
            displacement_path=None, stress_path=None,
            u_max_mm=None, von_mises_max=None, von_mises_mean=None,
            n_dofs=None,
            duration_s=round(time.perf_counter() - t0, 3),
            error=str(e),
        )