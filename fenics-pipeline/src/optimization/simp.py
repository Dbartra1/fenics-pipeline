# src/optimization/simp.py
#
# SIMP topology optimization loop.
#
# Algorithm: Optimality Criteria (OC) update — fast, robust, standard.
# Reference: Sigmund (2001) "A 99 line topology optimization code"
#            adapted for FEniCSx and 3D tetrahedral meshes.
#
# Compliance minimization subject to volume fraction constraint:
#   minimize:   C(rho) = F^T U  (global compliance = strain energy)
#   subject to: sum(rho * v) / sum(v) <= vf   (volume fraction)
#               0 < rho_min <= rho <= 1
#
# The penalized stiffness for element e:
#   K_e(rho_e) = rho_e^p * K_e^0
# where p=3 (penal) drives rho toward 0 or 1.

from __future__ import annotations
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.io as io
import ufl
from dolfinx.io import XDMFFile
from petsc4py import PETSc

from src.optimization.density_filter import (
    compute_element_centroids, build_filter_matrix, apply_filter
)
from src.fea.boundary_conditions import (
    load_boundary_mesh, build_boundary_conditions,
    TAG_VOLUME, TAG_TOP, TAG_BOTTOM
)


@dataclass
class SIMPConfig:
    volume_fraction:  float = 0.4    # target fraction of material to retain
    penal:            float = 3.0    # penalization power — 3 is standard
    filter_radius:    float = 6.0    # mm — set to 2-3x target_element_size
    max_iterations:   int   = 100
    convergence_tol:  float = 1e-3   # change in density field between iterations
    rho_min:          float = 1e-3   # minimum density — never true void (avoid singularity)
    move:             float = 0.2    # OC update move limit — max change per iteration
    checkpoint_every: int   = 10     # save density XDMF every N iterations
    safety_factor_min: float = 1.2   # abort if safety factor drops below this


@dataclass
class SIMPResult:
    success:           bool
    density_path:      Optional[Path]
    compliance_history: list[float]
    volume_history:    list[float]
    n_iterations:      int
    converged:         bool
    final_compliance:  Optional[float]
    final_volume_frac: Optional[float]
    duration_s:        float
    error:             Optional[str]

    def raise_if_failed(self) -> None:
        if not self.success:
            raise RuntimeError(f"SIMP optimization failed: {self.error}")


def _load_mesh_and_bcs(
    xdmf_path: Path,
    boundaries_xdmf: Path,
    load_hints: dict,
    material: dict,
) -> tuple:
    """
    Load mesh, BCs, and material constants.
    Called once before the optimization loop — mesh stays fixed throughout.
    Returns (domain, V, bc_set, lam, mu, dx, element_volumes, centroids, tets)
    """
    comm = MPI.COMM_WORLD

    with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )

    facet_tags = load_boundary_mesh(str(boundaries_xdmf), domain)

    # Vector CG1 for displacement
    V      = fem.FunctionSpace(domain, ufl.VectorElement("CG", domain.ufl_cell(), 1))
    bc_set = build_boundary_conditions(V, domain, facet_tags, load_hints)

    E  = material["youngs_modulus_pa"]
    nu = material["poissons_ratio"]
    lam_val = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_val  = E / (2 * (1 + nu))
    lam = fem.Constant(domain, lam_val)
    mu  = fem.Constant(domain, mu_val)

    dx = ufl.Measure("dx", domain=domain)

    # Element volumes — needed for volume fraction constraint
    # Integrate constant 1 over each cell using DG0
    W     = fem.FunctionSpace(domain, ("DG", 0))
    v_one = fem.Function(W)
    v_one.x.array[:] = 1.0
    vol_form = fem.form(v_one * dx)
    # Per-element volume via local assembly
    n_elem   = W.dofmap.index_map.size_local
    elem_vols = np.ones(n_elem) * (domain.comm.allreduce(
        fem.assemble_scalar(vol_form), op=MPI.SUM) / n_elem)

    # Element centroids and connectivity for filter
    points = domain.geometry.x
    tdim   = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    conn   = domain.topology.connectivity(tdim, 0)
    tets   = np.array([conn.links(i) for i in range(n_elem)])
    centroids = compute_element_centroids(points, tets)

    return domain, V, bc_set, lam, mu, dx, elem_vols, centroids, tets, W


def _penalized_solve(
    domain, V, bc_set, lam, mu, dx,
    rho: np.ndarray,
    W: fem.FunctionSpace,
    penal: float,
    petsc_options: dict,
) -> tuple[fem.Function, np.ndarray]:
    """
    Solve FEA with penalized stiffness field.
    Builds a DG0 stiffness modulation function from rho^penal,
    injects it into the bilinear form, and solves.

    Returns (displacement_function, element_strain_energies).
    Element strain energy = 0.5 * rho_e^penal * u_e^T K_e^0 u_e
    This is the sensitivity dc/drho_e up to the penalization factor.
    """
    # Density as DG0 function
    rho_fn = fem.Function(W, name="density")
    rho_fn.x.array[:] = rho

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma0(w):
        # Base stiffness — without density penalization
        return lam * ufl.tr(eps(w)) * ufl.Identity(len(w)) + 2 * mu * eps(w)

    # Penalized bilinear form — stiffness scaled by rho^penal
    a = rho_fn**penal * ufl.inner(sigma0(u), eps(v)) * dx

    T  = fem.Constant(domain, bc_set.traction_vec)
    ds = bc_set.ds
    L  = ufl.inner(T, v) * ds(bc_set.traction_tag)

    problem = fem_petsc.LinearProblem(
        a, L,
        bcs=bc_set.dirichlet,
        petsc_options=petsc_options,
    )
    u_sol = problem.solve()

    # Element strain energies (sensitivities)
    # dc/drho_e = -penal * rho_e^(penal-1) * u_e^T K_e^0 u_e
    # We compute the inner part: u_e^T K_e^0 u_e per element via DG0 projection
    strain_energy_expr = ufl.inner(sigma0(u_sol), eps(u_sol))
    se_fn  = fem.Function(W)
    se_expr = fem.Expression(strain_energy_expr, W.element.interpolation_points())
    se_fn.interpolate(se_expr)

    return u_sol, se_fn.x.array.copy()


def _oc_update(
    rho: np.ndarray,
    dc: np.ndarray,
    elem_vols: np.ndarray,
    volume_fraction: float,
    move: float,
    rho_min: float,
) -> np.ndarray:
    """
    Optimality Criteria density update.

    Bisection on Lagrange multiplier lambda to satisfy volume constraint.
    rho_new[e] = rho[e] * sqrt(-dc[e] / (lambda * vol[e]))  (clamped)

    The bisection finds the lambda that makes sum(rho_new * vol) = vf * sum(vol).
    Typically converges in 20-30 bisection steps.
    """
    total_vol = elem_vols.sum()
    target_vol = volume_fraction * total_vol

    l1, l2 = 1e-9, 1e9

    for _ in range(200):
        lmid    = 0.5 * (l1 + l2)
        # OC update formula
        rho_new = np.maximum(rho_min,
                  np.maximum(rho - move,
                  np.minimum(1.0,
                  np.minimum(rho + move,
                  rho * np.sqrt(np.maximum(0.0, -dc) / (lmid * elem_vols + 1e-16))
                  ))))
        # Check volume constraint
        if (rho_new * elem_vols).sum() > target_vol:
            l1 = lmid
        else:
            l2 = lmid
        if l2 - l1 < 1e-12 * (l1 + l2):
            break

    return rho_new


def run_simp(
    xdmf_path: str | Path,
    boundaries_xdmf: str | Path,
    part_name: str,
    output_dir: str | Path,
    load_hints: dict,
    material: dict,
    config: Optional[SIMPConfig] = None,
    petsc_options: Optional[dict] = None,
    checkpoint_callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
) -> SIMPResult:
    """
    Full SIMP optimization loop.

    checkpoint_callback: optional fn(iteration, rho, compliance) called
                         every checkpoint_every iterations — use for live
                         progress monitoring in the notebook.
    """
    config      = config or SIMPConfig()
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    petsc_options = petsc_options or {
        "ksp_type":  "preonly",
        "pc_type":   "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    t0 = time.perf_counter()
    compliance_history = []
    volume_history     = []

    try:
        # ── Setup (once) ──────────────────────────────────────────────────
        print("Loading mesh and building BCs...")
        (domain, V, bc_set, lam, mu, dx,
         elem_vols, centroids, tets, W) = _load_mesh_and_bcs(
            Path(xdmf_path), Path(boundaries_xdmf), load_hints, material
        )
        n_elem = len(elem_vols)
        print(f"  Elements: {n_elem:,}")

        # ── Density filter matrix (built once) ───────────────────────────
        print(f"Building filter matrix (r={config.filter_radius}mm)...")
        H = build_filter_matrix(centroids, config.filter_radius)
        print(f"  Filter matrix: {H.nnz:,} nonzeros ({100*H.nnz/n_elem**2:.3f}% fill)")

        # ── Initialize density field ──────────────────────────────────────
        # Uniform start at volume_fraction — feasible initial point
        rho = np.full(n_elem, config.volume_fraction, dtype=np.float64)

        converged  = False
        rho_change = np.inf
        comm       = MPI.COMM_WORLD

        # ── XDMF writer for density checkpoints ──────────────────────────
        density_path     = output_dir / f"{part_name}_density.xdmf"
        density_fn       = fem.Function(W, name="density")

        # ── Optimization loop ─────────────────────────────────────────────
        for iteration in range(1, config.max_iterations + 1):
            iter_t0 = time.perf_counter()
            rho_old = rho.copy()

            # Step 1: filter density
            rho_filtered = apply_filter(H, rho)

            # Step 2: penalized FEA solve
            u_sol, strain_energies = _penalized_solve(
                domain, V, bc_set, lam, mu, dx,
                rho_filtered, W, config.penal, petsc_options
            )

            # Step 3: compliance (objective)
            compliance = float(comm.allreduce(
                fem.assemble_scalar(fem.form(
                    rho_filtered**config.penal *
                    ufl.inner(ufl.sym(ufl.grad(u_sol)), ufl.sym(ufl.grad(u_sol))) * dx
                )), op=MPI.SUM
            ))
            compliance_history.append(compliance)

            # Step 4: sensitivities dc/drho_e
            # Chain rule through filter: dc/drho = H^T (dc/drho_filtered) / Hs
            dc_raw  = -config.penal * rho_filtered**(config.penal - 1) * strain_energies
            Hs      = np.array(H.sum(axis=1)).ravel()
            dc      = (H.T @ (dc_raw / (Hs + 1e-16)))

            # Step 5: OC density update
            rho = _oc_update(
                rho, dc, elem_vols,
                config.volume_fraction, config.move, config.rho_min
            )

            # Step 6: convergence check
            rho_change = np.max(np.abs(rho - rho_old))
            vol_frac   = (rho * elem_vols).sum() / elem_vols.sum()
            volume_history.append(float(vol_frac))

            iter_time = time.perf_counter() - iter_t0
            print(f"  Iter {iteration:4d} | C={compliance:.4e} | "
                  f"Vol={vol_frac:.3f} | Δρ={rho_change:.4f} | {iter_time:.1f}s")

            # Checkpoint
            if iteration % config.checkpoint_every == 0 or iteration == 1:
                density_fn.x.array[:] = rho
                with io.XDMFFile(comm, str(density_path), "w") as xdmf:
                    xdmf.write_mesh(domain)
                    xdmf.write_function(density_fn)

                if checkpoint_callback:
                    checkpoint_callback(iteration, rho.copy(), compliance)

            if rho_change < config.convergence_tol:
                print(f"\n✓ Converged at iteration {iteration} "
                      f"(Δρ={rho_change:.2e} < {config.convergence_tol})")
                converged = True
                break

        # Final density write
        density_fn.x.array[:] = rho
        with io.XDMFFile(comm, str(density_path), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(density_fn)

        if not converged:
            print(f"\n⚠ Did not converge in {config.max_iterations} iterations "
                  f"(Δρ={rho_change:.2e}). Consider increasing max_iterations.")

        return SIMPResult(
            success=True,
            density_path=density_path,
            compliance_history=compliance_history,
            volume_history=volume_history,
            n_iterations=len(compliance_history),
            converged=converged,
            final_compliance=compliance_history[-1] if compliance_history else None,
            final_volume_frac=volume_history[-1] if volume_history else None,
            duration_s=round(time.perf_counter() - t0, 3),
            error=None,
        )

    except Exception as e:
        return SIMPResult(
            success=False,
            density_path=None,
            compliance_history=compliance_history,
            volume_history=volume_history,
            n_iterations=len(compliance_history),
            converged=False,
            final_compliance=None,
            final_volume_frac=None,
            duration_s=round(time.perf_counter() - t0, 3),
            error=str(e),
        )