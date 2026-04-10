# src/optimization/simp.py
#
# SIMP topology optimization — rewritten based on proven FEniCSx implementation.
# Reference: Yaghoobi (2025) "Topology Optimization with FEniCSx"
#            Sigmund (2001) "A 99 line topology optimization code"
#
# Key design decisions:
# - LinearProblem built ONCE outside the loop (no JIT recompilation per iteration)
# - rho is a dolfinx Function updated in-place each iteration
# - Sensitivity filter applied to raw sensitivities (not density)
# - OC bisection uses adaptive bounds [0, ocp.sum()] not fixed [1e-9, 1e9]
# - Cell volumes computed via vector assembly (accurate, no tet geometry hacks)
# - Coordinate auto-detection: mm vs m

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.io as io
import ufl
from dolfinx.io import XDMFFile
from dolfinx.fem import form, Function, functionspace, Constant, Expression
from dolfinx.fem.petsc import LinearProblem, assemble_vector
from petsc4py import PETSc

from src.fea.boundary_conditions import (
    load_boundary_mesh, build_boundary_conditions,
    build_boundary_conditions_geometric,
    TAG_VOLUME, TAG_TOP, TAG_BOTTOM
)


@dataclass
class SIMPConfig:
    volume_fraction:   float = 0.4
    penal:             float = 3.0
    filter_radius:     float = 6.0    # mm
    max_iterations:    int   = 200
    convergence_tol:   float = 0.01   # max change in density between iterations
    rho_min:           float = 1e-3
    move:              float = 0.2
    checkpoint_every:  int   = 10
    safety_factor_min: float = 1.2


@dataclass
class SIMPResult:
    success:            bool
    density_path:       Optional[Path]
    compliance_history: list[float]
    volume_history:     list[float]
    n_iterations:       int
    converged:          bool
    final_compliance:   Optional[float]
    final_volume_frac:  Optional[float]
    duration_s:         float
    error:              Optional[str]

    def raise_if_failed(self) -> None:
        if not self.success:
            raise RuntimeError(f"SIMP optimization failed: {self.error}")


def _build_filter(centroids: np.ndarray, filter_radius_m: float):
    """
    Build sensitivity filter using KDTree.
    Returns (omega, omega_sum) where omega is a sparse weight matrix.
    filter_radius_m is in metres.
    """
    tree = cKDTree(centroids)
    distance = tree.sparse_distance_matrix(tree, filter_radius_m).tocsr()
    distance.data = (filter_radius_m - distance.data) / filter_radius_m
    omega = distance
    omega_sum = np.array(omega.sum(axis=1)).flatten()
    return omega, omega_sum


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
    bc_params=None,
    geometry_params=None,
    x_init: Optional[np.ndarray] = None,
) -> SIMPResult:
    """
    Full SIMP optimization loop.
    LinearProblem is assembled ONCE and rho updated in-place — no JIT per iteration.
    """
    config     = config or SIMPConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    petsc_options = petsc_options or {
        "ksp_type": "cg",
        "pc_type":  "gamg",
        "ksp_rtol": 1e-8,
    }

    t0                 = time.perf_counter()
    compliance_history = []
    volume_history     = []

    try:
        comm = MPI.COMM_WORLD

        # ── Load mesh ─────────────────────────────────────────────────────
        with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )

        # Auto-detect coordinate system
        coord_max = domain.geometry.x.max()
        if coord_max > 1.0:
            domain.geometry.x[:] /= 1000.0
            print("  Converted coordinates mm → m")
        else:
            print("  Coordinates already in metres")

        # ── Function spaces ───────────────────────────────────────────────
        dim  = domain.geometry.dim
        CG1  = functionspace(domain, ("CG", 1, (dim,)))
        DG0  = functionspace(domain, ("DG", 0))

        n_elem = DG0.dofmap.index_map.size_local
        print(f"  Elements: {n_elem:,}")

        # pts must be defined before cell volumes and centroids
        pts = domain.geometry.x
        dx  = ufl.Measure("dx", domain=domain)

        # ── Cell volumes via tet geometry ─────────────────────────────────
        domain.topology.create_connectivity(domain.topology.dim, 0)
        conn_v       = domain.topology.connectivity(domain.topology.dim, 0)
        tets_v       = np.array([conn_v.links(i) for i in range(n_elem)])
        p0 = pts[tets_v[:, 0]]; p1 = pts[tets_v[:, 1]]
        p2 = pts[tets_v[:, 2]]; p3 = pts[tets_v[:, 3]]
        cell_volumes = np.abs(np.einsum(
            'ij,ij->i', p1-p0, np.cross(p2-p0, p3-p0)
        )) / 6.0
        total_volume = cell_volumes.sum()
        print(f"  Total volume: {total_volume*1e6:.2f} cm³")

        # ── Element centroids for filter ──────────────────────────────────
        tets      = tets_v   # reuse connectivity
        centroids = pts[tets].mean(axis=1)

        try:
            import h5py
            h5_path = str(xdmf_path).replace(".xdmf", ".h5")
            nondesign_mask = np.zeros(n_elem, dtype=bool)
            if Path(h5_path).exists():
                with h5py.File(h5_path, "r") as f:
                    if "data2" in f:
                        tags = f["data2"][:].ravel()
                        nondesign_mask = (tags == 2)
                        print(f"  Non-design elements: {nondesign_mask.sum():,} "
                              f"({100*nondesign_mask.mean():.1f}%) — forced solid")
                    else:
                        print("  No physical tags found — all elements are design")
            else:
                print("  No HDF5 file found — all elements are design")
        except Exception as e:
            print(f"  Non-design detection skipped: {e}")
            nondesign_mask = np.zeros(n_elem, dtype=bool)

        design_mask = ~nondesign_mask
        # ── Build filter ──────────────────────────────────────────────────
        filter_radius_m = config.filter_radius / 1000.0
        print(f"Building filter (r={config.filter_radius}mm)...")
        omega, omega_sum = _build_filter(centroids, filter_radius_m)
        print(f"  Filter: {omega.nnz:,} nonzeros, {omega.nnz/n_elem:.0f} avg neighbors")

        # ── Boundary conditions ───────────────────────────────────────────
        bc_set = build_boundary_conditions_geometric(
            CG1, domain, load_hints, bc_params, geometry_params
        )

        # ── Density function (updated in-place each iteration) ────────────
        rho = Function(DG0, name="density")
        rho.x.array[:] = config.volume_fraction

        # ── Build weak form ONCE ──────────────────────────────────────────
        u = ufl.TrialFunction(CG1)
        v = ufl.TestFunction(CG1)

        E  = material["youngs_modulus_pa"]
        nu = material["poissons_ratio"]
        lam_val = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_val  = E / (2 * (1 + nu))
        lam = Constant(domain, lam_val)
        mu  = Constant(domain, mu_val)

        def eps(w):
            return ufl.sym(ufl.grad(w))

        def sigma(w):
            return lam * ufl.tr(eps(w)) * ufl.Identity(dim) + 2 * mu * eps(w)

        a = rho**config.penal * ufl.inner(sigma(u), eps(v)) * dx

        T  = Constant(domain, bc_set.traction_vec)
        ds = bc_set.ds
        L  = ufl.inner(T, v) * ds(bc_set.traction_tag)

        problem = LinearProblem(
            a, L,
            bcs=bc_set.dirichlet,
            petsc_options=petsc_options,
        )

        # ── Strain energy expression (reused each iteration) ──────────────
        energy_fn   = Function(DG0, name="strain_energy")
        energy_expr = Expression(
            ufl.inner(sigma(problem.u), eps(problem.u)),
            DG0.element.interpolation_points()
        )

        # ── XDMF writer ───────────────────────────────────────────────────
        density_path = output_dir / f"{part_name}_density.xdmf"

        converged  = False
        rho_change = np.inf

        if x_init is not None:
            # Warm-start from provided density field
            if len(x_init) != n_elem:
                raise ValueError(
                    f"x_init length {len(x_init)} != n_elem {n_elem}"
                )
            x = x_init.copy().astype(np.float64)
            x = np.clip(x, config.rho_min, 1.0)
            print(f"  Warm-start: x_init provided "
                  f"(mean={x.mean():.3f}, min={x.min():.4f}, max={x.max():.4f})")
        else:
            x = rho.x.array.copy()

        x[nondesign_mask] = 1.0   # non-design always solid

        # ── Optimization loop ─────────────────────────────────────────────
        for iteration in range(1, config.max_iterations + 1):
            iter_t0 = time.perf_counter()
            x_old   = x.copy()

            # Step 1: update rho in-place
            rho.x.array[:] = np.maximum(config.rho_min, x)

            # Step 2: solve FEA
            u_sol = problem.solve()

            # Step 3: strain energies
            energy_fn.interpolate(energy_expr)
            strain_energies = energy_fn.x.array.copy()

            # Step 4: compliance — volume-integrated (physical Joules)
            # strain_energies is density [Pa]; multiply by cell_volumes to get [J]
            compliance = float(np.dot(x**config.penal * cell_volumes, strain_energies))
            compliance_history.append(compliance)

            # Step 5: raw sensitivities
            dc = -config.penal * x**(config.penal - 1) * strain_energies

            # Step 6: filter sensitivities
            dc_filtered = (omega @ dc) / (omega_sum + 1e-16)

            # Only optimize design elements — exclude non-design from OC
            # NOTE: dc is strain energy *density* (not volume-integrated), so cell_volumes
            # must NOT appear here. Adding 1/v_e would produce ranking inversions on
            # non-uniform meshes (small corner tets near pins get spuriously boosted).
            # Correct KKT-derived formula: ocp = x * sqrt(-dc_density_filtered)
            ocp  = x * np.sqrt(np.maximum(0.0, -dc_filtered))
            ocp[nondesign_mask] = 0.0
            l1   = 0.0
            l2   = ocp.sum() + 1e-30   # upper bound from all elements
            move = config.move

            for _ in range(200):
                l_mid = 0.5 * (l1 + l2)
                if l_mid < 1e-30:
                    break
                x_new = (ocp / l_mid).clip(x - move, x + move).clip(config.rho_min, 1.0)
                x_new[nondesign_mask] = 1.0   # force non-design solid
                # Volume constraint over design elements only
                vol = (x_new[design_mask] * cell_volumes[design_mask]).sum() / \
                      cell_volumes[design_mask].sum()
                if vol > config.volume_fraction:
                    l1 = l_mid
                else:
                    l2 = l_mid
                if (l2 - l1) < 1e-9 * (l1 + l2 + 1e-30):
                    break

            x_new[nondesign_mask] = 1.0
            x = x_new

            # Step 7: convergence
            rho_change = float(np.max(np.abs(x - x_old)))
            design_vol_total = cell_volumes[design_mask].sum()
            vol_frac = float((x[design_mask] * cell_volumes[design_mask]).sum()
                             / design_vol_total)
            volume_history.append(vol_frac)

            iter_time = time.perf_counter() - iter_t0
            print(f"  Iter {iteration:4d} | C={compliance:.4e} | "
                  f"Vol={vol_frac:.3f} | Δρ={rho_change:.4f} | {iter_time:.1f}s")

            if iteration % config.checkpoint_every == 0 or iteration == 1:
                rho.x.array[:] = x
                with io.XDMFFile(comm, str(density_path), "w") as xdmf:
                    xdmf.write_mesh(domain)
                    xdmf.write_function(rho)
                if checkpoint_callback:
                    checkpoint_callback(iteration, x.copy(), compliance)

            frac_intermediate = np.logical_and(
                0.15 < x[design_mask], x[design_mask] < 0.85
            ).sum() / design_mask.sum()
            if rho_change < config.convergence_tol or frac_intermediate < 0.01:
                print(f"\n✓ Converged at iteration {iteration} "
                      f"(Δρ={rho_change:.2e}, {frac_intermediate*100:.1f}% intermediate)")
                converged = True
                break

        # Final write
        rho.x.array[:] = x
        with io.XDMFFile(comm, str(density_path), "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(rho)

        if not converged:
            frac_intermediate = np.logical_and(
                0.15 < x[design_mask], x[design_mask] < 0.85
            ).sum() / design_mask.sum()
            print(f"\n⚠ Did not converge in {config.max_iterations} iterations "
                  f"(Δρ={rho_change:.2e}, {frac_intermediate*100:.1f}% intermediate)")

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
        import traceback
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
            error=f"{e}\n{traceback.format_exc()}",
        )