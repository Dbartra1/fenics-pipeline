# MATERIALS_SPEC.md
# fenics-pipeline — Mathematics, Materials, and Physical Foundations
#
# PURPOSE
# ───────
# This document is the authoritative reference for every piece of physics,
# mathematics, and material science that underlies the fenics-pipeline SIMP
# topology optimization system. It is written for use by Claude sessions working
# on this project — any session that touches the solver, the notebooks, the
# boundary conditions, or the diagnostic layer should load this document first.
#
# It is NOT a tutorial. It assumes comfort with linear algebra and basic
# mechanics. It IS complete — every formula used in the pipeline is derived
# here with enough context to understand what it means, why it is correct,
# and what goes wrong when it is violated.
#
# STRUCTURE
# ─────────
# 1.  The Physical Problem — what we are actually solving
# 2.  Linear Elasticity — the governing equations
# 3.  Finite Element Discretization — turning PDEs into matrix equations
# 4.  Compliance — the objective function and its physical meaning
# 5.  SIMP Penalization — the core topology optimization trick
# 6.  Sensitivity Analysis — how compliance changes with density
# 7.  Density Filtering — preventing checkerboard and mesh dependence
# 8.  Optimality Criteria Update — the update rule and bisection
# 9.  Two-Stage Penalization Continuation — why and how
# 10. Convergence — what "done" means and how to detect it
# 11. Boundary Conditions — fixed DOFs, loads, and their effect on results
# 12. Material Library — confirmed properties for this pipeline
# 13. Parameter Reference — every tunable parameter, its meaning and safe range
# 14. Diagnostic Interpretation — reading run quality from numbers alone
# 15. Known Failure Modes — what goes wrong and why
# 16. Pipeline-Specific Implementation Notes — decisions specific to this codebase
#
## LAST UPDATED: 2026-06-04
# PIPELINE VERSION: Rust solver — AMGCL AMG-PCG primary (smoothed aggregation +
#                  ILU(0) smoother, block_size=3, OpenMP), 140 tests passing.
#                  Two-stage penalization (p=2→p=3) at notebook level.
#                  VCycle preconditioner is experimental, NOT in dispatch — AMGCL
#                  is the live path (see SOLVER_STATE.md §3).
# ─────────────────────────────────────────────────────────────────────────────


---

## 1. The Physical Problem

### What We Are Optimizing

Given a structural domain Ω (the bounding box of a part), find the distribution
of material within Ω that minimizes structural compliance (maximizes stiffness)
subject to a constraint on the total volume of material used.

Formally:

    min  C(ρ) = ∫_Ω σ(u) : ε(u) dΩ       (compliance = strain energy × 2)
     ρ
    s.t. ∫_Ω ρ dΩ ≤ V*                    (volume constraint)
         0 ≤ ρ(x) ≤ 1  ∀x ∈ Ω            (density bounds)
         a(u,v;ρ) = l(v)  ∀v ∈ V          (equilibrium — the FEA problem)

Where:
- ρ(x) ∈ [0,1] is the local material density field (the design variable)
- u is the displacement field (state variable — computed by FEA)
- σ = stress tensor, ε = strain tensor
- V* = target volume = V_fraction × |Ω|
- a(u,v;ρ) is the bilinear form (stiffness), l(v) is the load linear form

### Why Compliance?

Compliance C = ∫ f·u dΓ_N = ∫ σ:ε dΩ = uᵀKu is the work done by external
forces. A stiffer structure deforms less under the same load, so minimizing
compliance = maximizing stiffness. It is the canonical objective for structural
optimization because:

1. It has a clean mathematical adjoint (sensitivity derivation is trivial)
2. It is physically meaningful — units are Joules (energy)
3. It is self-adjoint — the adjoint problem is identical to the primal problem,
   so no second solve is needed for sensitivities

### What the Result Physically Means

After optimization, ρ(x) ≈ 0 (void) or ρ(x) ≈ 1 (solid) everywhere. The
solid regions form a load path — the structural skeleton that carries force
from the load application points to the fixed supports with minimum energy.

The topology (connectivity) of this skeleton is what SIMP discovers. It cannot
be specified in advance; it emerges from the physics of the load case.


---

## 2. Linear Elasticity

### Governing PDE

The equilibrium equation (strong form) in a domain Ω ⊂ ℝ³:

    div(σ) + b = 0    in Ω
    u = 0             on Γ_D  (Dirichlet — fixed boundary)
    σ·n = t           on Γ_N  (Neumann — traction/load boundary)

Where b is the body force (gravity — typically neglected in topology
optimization), t is the applied traction, n is the outward normal.

### Constitutive Law (Hooke's Law for Linear Isotropic Elasticity)

    σ = C : ε

Where C is the 4th-order stiffness tensor. For isotropic materials:

    σ_ij = λ δ_ij ε_kk + 2μ ε_ij

Where:
- λ = Lamé's first parameter = Eν / ((1+ν)(1-2ν))
- μ = shear modulus = E / (2(1+ν))
- E = Young's modulus [Pa]
- ν = Poisson's ratio [dimensionless]

### Strain-Displacement Relation (small strain assumption)

    ε_ij = ½ (∂u_i/∂x_j + ∂u_j/∂x_i) = sym(∇u)

This is the linearized Green strain. Valid when displacements are small
relative to the domain size — always true for stiffness-optimized structures
where the whole point is minimizing displacement.

### Weak Form (what FEniCSx solves)

Multiply by test function v and integrate by parts:

    ∫_Ω σ(u) : ε(v) dΩ = ∫_Γ_N t·v dΓ    ∀v ∈ V

Left side = bilinear form a(u,v). Right side = linear form l(v).
This is the starting point for finite element discretization.

### Why We Use First-Order Elements (CG1 / Linear Hex)

The pipeline uses linear tetrahedral elements (FEniCSx path) and linear
hexahedral (voxel) elements (Rust path). The key tradeoff:

- Linear tets: simple, robust, but volume-lock on incompressible materials
  (ν → 0.5). At ν=0.3 (steel, aluminum) locking is negligible.
- Linear hex (voxel): optimal for structured grids, no hourglassing with
  full integration, ideal for SIMP because each element maps exactly to
  one density variable — no ambiguity in the density field.

The voxel (Rust) path is preferred for SIMP for this reason: one element =
one density variable = no interpolation artifacts at the boundary.


---

## 3. Finite Element Discretization

### Assembly

After discretizing Ω into elements and choosing basis functions φ_i, the
weak form becomes:

    K(ρ) · U = F

Where:
- K(ρ) is the global stiffness matrix (n_dof × n_dof)
- U is the displacement vector (n_dof × 1)
- F is the force vector (n_dof × 1)
- n_dof = 3 × n_nodes (3 displacement components per node)

The stiffness matrix assembles element-wise:

    K(ρ) = Σ_e ρ_e^p · K_e

Where K_e is the element stiffness matrix (computed once from geometry and
material properties) and p is the penalization exponent (SIMP power).

### Element Stiffness Matrix (Voxel Path)

For a unit cube hex element with voxel_size h, the element stiffness matrix
K_e is computed from:

    K_e = ∫_e Bᵀ C B dV

Where B is the strain-displacement matrix (maps node displacements to strains)
and C is the constitutive matrix. For a hex element this is a 24×24 matrix
(8 nodes × 3 DOFs/node).

Key property: K_e is the SAME for every element on a uniform voxel grid
(same geometry, same material). The pipeline computes it once in `ke_base.rs`
and reuses it for all elements. This is a major computational advantage of
the structured voxel approach.

### Degrees of Freedom Layout

From `types.rs` (authoritative):

    node_idx(ix, iy, iz) = ix + iy*(nx+1) + iz*(nx+1)*(ny+1)
    elem_idx(ix, iy, iz) = ix + iy*nx     + iz*nx*ny

X-fastest, Z-slowest. DOFs for node n: [3n, 3n+1, 3n+2] = [u_x, u_y, u_z].

When reshaping the flat density array from the solver output:

    density_3d = density_flat.reshape((nz, ny, nx))

Slice conventions:
    XY plane (top-down, constant Z): density_3d[iz_mid, :, :]   → shape (ny, nx)
    XZ plane (front,   constant Y): density_3d[:, iy_mid, :]    → shape (nz, nx)
    YZ plane (side,    constant X): density_3d[:, :, ix_mid]    → shape (nz, ny)

CRITICAL: Any code that reshapes or slices the density array must use
(nz, ny, nx) order, NOT (nx, ny, nz). Getting this wrong produces a valid
array with scrambled geometry — a silent failure that is only detectable
by visually inspecting the slice plots.

### Linear System Solution

The system K·U = F is solved by Conjugate Gradient (CG) in the Rust path,
preconditioned by:
- Jacobi (diagonal scaling) — CPU fallback
- VCycle multigrid (via vcycle_dispatch.rs) — CPU primary for large problems
- ILU(0) + cuSPARSE — GPU path

The matrix K is symmetric positive definite (SPD) when properly constrained,
making CG the correct choice. The condition number of K grows as material
becomes sparser during SIMP iterations, which is why the CG iteration count
typically increases in later iterations.

Dispatch threshold (from solver.rs): problems ≥ 50k DOFs use VCycle;
smaller problems use faer sparse Cholesky (direct, exact, fast for small N).


---

## 4. Compliance — The Objective Function

### Definition and Units

    C = Uᵀ F = Uᵀ K U                    [Joules]

This is the total strain energy × 2. It equals the work done by external
forces when the structure deforms to equilibrium under load F.

For the voxel solver, compliance is computed element-wise:

    C = Σ_e ρ_e^p · u_eᵀ · K_e · u_e    [Joules]

Where u_e is the 24-element displacement vector for element e (extracted
from the global U by DOF lookup).

### Physical Interpretation

- Low compliance → stiff structure (small deformation under load) → GOOD
- High compliance → flexible structure (large deformation under load) → BAD
- Compliance has units of Joules because it is energy (force × displacement)
- Compliance is ALWAYS positive for a loaded structure

### Compliance Scaling with Problem Size

Compliance scales with the physical load magnitude and domain size. For the
motor mount example (5000N load, steel, ~96k elements at 1mm voxels):

    Typical converged compliance: ~0.041 J

This value is NOT directly comparable across different parts, loads, or
materials. Only the RELATIVE compliance matters:
- Between iterations of the same run (must decrease)
- Between runs of the same part with different parameters (lower = better)
- Between Stage 1 and Stage 2 of the two-stage continuation

### Why Compliance Increases at the Start of a Run

The compliance history for a typical run shows:

    Iter 1:  0.033  ← initial uniform density (all ρ = VF)
    Iter 2:  0.029  ← compliance drops as uniform density is efficient
    Iter 3:  0.037  ← RISES as topology begins to form (material removed)
    Iter 4:  0.039  ← peaks here
    Iter 5+: monotone decrease toward convergence

The initial rise is expected and NOT a sign of a problem. When material is
first removed from low-sensitivity regions, compliance temporarily worsens
because the load path has been disrupted. The optimizer then reroutes the
load path through the remaining material, which produces a better structure
than the initial uniform field.

A run that does NOT show this initial rise is suspicious — it may indicate
that the sensitivity filter is too aggressive (smoothing out the gradient
signal) or that the move limit is too small to make progress.


---

## 5. SIMP Penalization

### The Core Idea

SIMP (Solid Isotropic Material with Penalization) uses a single scalar trick
to make the problem tractable: material stiffness is related to density by a
power law:

    E(ρ_e) = ρ_e^p · E_0

Where:
- ρ_e ∈ [ρ_min, 1] is the element density (design variable)
- p is the penalization exponent (typically 3.0)
- E_0 is the Young's modulus of the solid material
- ρ_min = 1e-3 (prevents singular stiffness matrix in void regions)

### Why the Power Law Works

At p = 1: E(ρ) = ρ·E_0. An element at ρ = 0.5 has 50% stiffness AND 50%
volume. It is equally efficient per unit material as solid — so the optimizer
has no reason to push toward binary 0/1. Result: gray mush everywhere.

At p = 3: An element at ρ = 0.5 has:
    - Stiffness: 0.5³ × E_0 = 0.125 × E_0 (12.5% of solid stiffness)
    - Volume:    0.5 × V_e  = 50% of element volume

The element contributes only 12.5% stiffness but costs 50% of the volume
budget. It is mechanically inefficient at intermediate density. The optimizer
is therefore strongly driven to make elements either 0 (void, costs nothing)
or 1 (solid, efficient). This is the penalization effect.

### Why p Must Be ≥ 3 for Binary Results

The condition for SIMP to produce binary designs is:

    p ≥ E₀/E_Hashin_upper

Where E_Hashin_upper is the upper Hashin-Shtrikman bound for a two-phase
composite at density ρ. For 3D problems, the theoretical minimum penalization
is p ≈ 3 for ν = 0.3. Below p = 3, intermediate densities are mechanically
realizable composites and the optimizer will use them. Above p = 3, they are
not, and the optimizer is forced toward 0/1.

In practice, p = 3 is the standard choice. p > 3 drives harder toward binary
but can cause numerical instability (very small stiffness in near-void elements
produces ill-conditioned K). p = 4 or 5 is occasionally used for very clean
results but requires finer convergence tolerances.

### Why We Use Two Stages (p=2 → p=3)

Starting directly at p=3 is problematic:
- The stiffness matrix is highly ill-conditioned from iteration 1 (near-void
  elements have stiffness ~ρ_min³ × E₀ ≈ 10⁻⁹ × E₀)
- The optimizer makes aggressive initial moves that can create disconnected
  topology — islands of solid with no load path
- Convergence is slower because the gradient landscape is highly non-convex
  from the start

Starting at p=2 then switching to p=3 (implemented at notebook level in
Cell 3 via run_stage):
- Stage 1 (p=2): explores the topology space efficiently, finds approximate
  load paths, converges quickly (low ill-conditioning)
- Stage 2 (p=3): commits the topology from Stage 1, drives ρ toward binary,
  refines the load path under stronger penalization

The Stage 2 warm-start from Stage 1's density field is critical. Without it,
Stage 2 at p=3 starting from uniform density is much slower to converge.

IMPORTANT BUG — GPU warm-start: The GPU CG solver zeroes u on entry to each
solve. On the CPU path, u is carried forward between iterations (warm-start),
which reduces CG iterations needed. This is documented tech debt — the GPU
path is functionally correct but uses more CG iterations in early SIMP
iterations than necessary. Fix is to pass the previous u as initial guess
to the GPU solver.

### ρ_min — Why Not Zero?

Setting ρ_e = 0 exactly makes K singular (element contributes zero stiffness,
row and column become zero). ρ_min = 1e-3 is the standard choice:

    E(ρ_min) = (1e-3)³ × E₀ = 1e-9 × E₀

This is 10⁻⁹ times the solid stiffness — mechanically negligible but
numerically sufficient to keep K positive definite. The value 1e-3 is
deliberately the same order as the convergence tolerance so that elements
that reach ρ_min don't continue affecting the solution.

DO NOT set ρ_min < 1e-4 without also tightening the CG tolerance and
reducing max_cg_iter, as the condition number of K scales as
κ(K) ~ O(ρ_min^(-p)), and smaller ρ_min dramatically worsens conditioning.


---

## 6. Sensitivity Analysis

### What We Need

To update densities, we need ∂C/∂ρ_e — how much does compliance change if
we add a tiny amount of material to element e? Negative sensitivity means
adding material to element e reduces compliance (good). Positive means it
increases compliance (removing material from e is beneficial).

### Derivation (Adjoint Method)

From C = Uᵀ F = Uᵀ K U:

    ∂C/∂ρ_e = ∂(Uᵀ K U)/∂ρ_e

Using the chain rule and the fact that K U = F (constant), the adjoint
problem is identical to the primal problem (self-adjointness of compliance):

    ∂C/∂ρ_e = -p · ρ_e^(p-1) · u_eᵀ · K_e · u_e

The term (u_eᵀ · K_e · u_e) is the element strain energy — always positive.
So ∂C/∂ρ_e is always NEGATIVE (adding material always reduces compliance or
has no effect in void regions).

In code (`sensitivity.rs` / `simp.py`):

    dc[e] = -p * ρ_e^(p-1) * strain_energy_e

Where strain_energy_e = u_eᵀ K_e u_e (NOT multiplied by ρ_e^p — that is
already accounted for in the OC update via ocp).

### Common Mistake: Double-Counting ρ in Sensitivity

The raw sensitivity from the chain rule is:

    ∂C/∂ρ_e = -p · ρ_e^(p-1) · (u_eᵀ · K_e · u_e)

Some implementations incorrectly use:

    dc[e] = -p * strain_energy_e / ρ_e     ← WRONG

This produces the correct relative ranking but incorrect magnitude, which
affects the OC bisection. The correct form above is what this pipeline uses.

### Sensitivity vs. Cell Volumes

A critical implementation decision: dc is strain energy DENSITY [Pa], not
volume-integrated strain energy [J]. The cell volumes must NOT appear in
the sensitivity computation.

Why: On a non-uniform mesh (or when cell volumes vary across the domain),
elements near boundary conditions tend to be smaller due to mesh refinement.
If we normalize by cell volume, these elements get artificially boosted
sensitivity (small volume → high density sensitivity per unit), causing the
optimizer to preferentially place or remove material near boundaries rather
than along the true load path.

On the voxel path all elements are the same size so this doesn't matter.
But the FEniCSx tet path can have variable element sizes, and this is why
the sensitivity filter uses the x-weighted form (see Section 7) rather
than a volume-weighted form.


---

## 7. Density Filtering

### Why Filtering Is Necessary

Without filtering, SIMP produces two pathological results:

**Checkerboard instability**: A pattern where alternating elements are solid
and void in a checkerboard arrangement. This artificial pattern has higher
numerically-computed stiffness than a physically realizable composite at the
same volume fraction (an artifact of the FEM discretization). The optimizer
finds checkerboard solutions preferentially. These solutions are NOT physically
manufacturable — a real checkerboard of solid and void has zero stiffness.

**Mesh dependence**: Without a length scale control, the optimal topology
changes as the mesh is refined. Finer mesh → thinner and more numerous struts.
This means results are not mesh-converged and cannot be trusted for
manufacturing without specifying the mesh.

The filter imposes a minimum length scale by spatially averaging sensitivities
over a neighborhood of radius r. Features smaller than r cannot form.

### The Sensitivity Filter (Sigmund 2001)

The filter used in this pipeline is the sensitivity filter from Sigmund (2001).
It operates on raw sensitivities dc before the OC update, replacing each
element's sensitivity with a weighted average of its neighbors:

    dc_filtered_e = Σ_f H_ef · x_f · dc_f
                    ─────────────────────────
                         Σ_f H_ef · x_f

Where:
- H_ef = max(0, r - |x_e - x_f|) is the cone-shaped weighting kernel
- r is the filter radius [meters]
- x_f is the density of neighbor element f (the x-weighted form)
- The sum is over all elements f within distance r of element e

This is the KKT-correct form (x-weighted denominator). The alternative
simple form (dividing by Σ H_ef without x-weighting) is a common mistake
that amplifies sensitivities at the boundary between solid and void regions,
causing the optimizer to preferentially erode void boundaries — producing
shells with hollow interiors rather than load-path-following struts.

In code (`simp.py`):

    dc_filtered = (omega @ (x * dc)) / (omega @ x + 1e-16)

Where omega is the sparse weight matrix built by `_build_filter()`.

In the Rust solver (`filter.rs`): same formula, implemented as a sparse
matrix-vector multiply using precomputed weights.

### Filter Radius Rules (CONFIRMED FOR THIS PIPELINE)

The filter radius must satisfy:

    r ≥ 2 × voxel_size    (minimum — one-element-wide features still possible)
    r ≤ 5 × voxel_size    (maximum — beyond this, results are over-smoothed)

CONFIRMED WORKING:
    VOXEL_SIZE_MM = 1.0mm  →  FILTER_RADIUS = 3.0mm  (3× voxel size)

CONFIRMED FAILURE:
    VOXEL_SIZE_MM = 1.0mm  →  FILTER_RADIUS = 8.0mm  (8× voxel size)
    Result: blob topology — mass concentrates at load application points
    rather than forming a load-path skeleton.

The 3× rule is a solid default. If the resulting STL shows:
- Spiky thin struts: increase FILTER_RADIUS (r too small → checkerboard leaking)
- Blob topology / over-smoothed with no clear struts: decrease FILTER_RADIUS
- Disconnected islands: usually a BC issue, but increasing r slightly can help

### The Density Filter (Alternative — Not Used in This Pipeline)

The density filter (Bruns & Tortorelli 2001) filters the density field ρ
rather than the sensitivities:

    ρ_filtered_e = Σ_f H_ef · ρ_f / Σ_f H_ef

This guarantees a minimum length scale on BOTH solid and void regions, which
the sensitivity filter does not. It is theoretically superior for
manufacturability. However it requires a modified sensitivity derivation
(chain rule through the filter), and the implementation is more complex.

This pipeline uses the sensitivity filter for simplicity and proven
convergence behavior. The density filter is a future upgrade path if
finer length scale control is needed.


---

## 8. Optimality Criteria (OC) Update

### The KKT Conditions

The optimization problem has KKT conditions (necessary conditions for
optimality). At a solution, for each design element e:

    If ρ_e = ρ_min:    ∂C/∂ρ_e - μ ≥ 0    (lower bound active)
    If ρ_e = 1:        ∂C/∂ρ_e - μ ≤ 0    (upper bound active)
    If ρ_min < ρ_e < 1: ∂C/∂ρ_e - μ = 0   (interior point)

Where μ is the Lagrange multiplier for the volume constraint. At the
interior point condition:

    ∂C/∂ρ_e = μ    for all active design elements

This means at optimality, the sensitivity ∂C/∂ρ_e is the same for all
non-bound elements. The optimizer is done when no element can profitably
exchange material with another.

### The OC Update Rule

The standard OC update (Sigmund 2001):

    ρ_e^(k+1) = clamp(ρ_e^k · B_e^η, ρ_e^k - move, ρ_e^k + move)

Where:
- B_e = (-∂C/∂ρ_e) / (μ · ρ_e) = ocp_e / (μ · ρ_e)   (optimality quotient)
- η = 0.5 (historical default — but see damping below)
- move is the move limit
- μ is found by bisection to satisfy the volume constraint

In this pipeline, B_e^η with η=0.5 is implemented as:

    ocp_e = ρ_e · sqrt(max(0, -dc_filtered_e))

Then:

    x_new_e = clamp(ocp_e / l_mid, x_e - move, x_e + move)
    x_new_e = clamp(x_new_e, ρ_min, 1.0)

Where l_mid is found by bisection. This is algebraically equivalent to the
B_e^η form above with η=0.5.

### OC Damping (Breaks 2-Cycle Oscillation)

A known failure mode of the raw OC update is 2-cycle oscillation: the
density alternates between two values without converging. This happens because
the update is aggressive — elements overshoot the optimal value and then
correct back past it.

The fix is damping: instead of fully accepting x_new, blend with the previous:

    x_e ← 0.5 × x_new_e + 0.5 × x_e      (damping factor = 0.5)

This is applied in `simp.py` and `oc_update.rs`. The 0.5 factor is the
standard choice and matches what is specified in `simp_config.damping`.

DO NOT remove the damping. Without it, many runs fail to converge — they
oscillate forever between two compliance values about 1% apart.

### Bisection for the Volume Lagrange Multiplier

We need to find μ such that Σ_e ρ_e^new(μ) = V*. The function
Σ_e ρ_e^new(μ) is monotone decreasing in μ (higher multiplier → more material
removed). Standard bisection:

    l1 = 0.0                                    (lower bound)
    l2 = max(ocp) / ρ_min                       (tight upper bound)

The tight upper bound for l2 is important. The naive choice l2 = 1e9 is
technically safe but causes ~30 extra bisection iterations per SIMP
iteration, which adds up. The tight bound ensures l2 is large enough that
all elements hit their lower bound (x_e - move or ρ_min) at l_mid = l2.

Bisection terminates when:
    (l2 - l1) < 1e-9 × (l1 + l2)   (relative tolerance)
    OR after 200 iterations (safety cap — should never be reached)

Non-design elements are EXCLUDED from the volume computation:

    vol = (x_new[design_mask] × cell_volumes[design_mask]).sum()
          / cell_volumes[design_mask].sum()

And non-design elements are FORCED to ρ = 1.0 before and after bisection.

### Move Limit

The move limit constrains how much density can change in a single iteration:

    ρ_e^new ∈ [ρ_e - move, ρ_e + move]

CONFIRMED WORKING: move = 0.3 (density can change by ±0.3 per iteration).

Too small (move < 0.1): Slow convergence — topology takes hundreds of extra
iterations to commit. Useful for very fine-grained control near convergence
but not during topology development.

Too large (move > 0.5): Aggressive topology changes that can cause:
- Disconnected topology islands (material removed too fast before rerouting)
- Compliance spikes between iterations
- Non-convergence (oscillation mode)

move = 0.2–0.3 is the sweet spot for 1mm voxel grids.


---

## 9. Two-Stage Penalization Continuation

### Implementation Location

Two-stage continuation is implemented at the NOTEBOOK level (Cell 3 of
04_simp_optimization.ipynb), NOT inside the Rust solver. The solver
always runs a single stage with a fixed penal value. The notebook calls
run_stage() twice with different penal values.

This is a deliberate architectural decision: keeping the solver stateless
makes it easier to test and reason about. The notebook orchestrates the
strategy; the solver implements the mechanics.

### Stage Configuration (Confirmed Working)

Stage 1 — Topology development:
    penal = 2.0
    max_iterations = MAX_ITERATIONS (typically 200)
    convergence_tol = 0.0005
    min_iterations = 30         (must explore before declaring convergence)

Stage 2 — Topology commitment:
    penal = 3.0
    max_iterations = 200
    convergence_tol = 0.0005
    min_iterations = 50         (must develop past initial plateau at p=3)
    x_init = Stage 1 final density (warm-start)

### The Stage Boundary Compliance Spike

When switching from p=2 to p=3, the compliance ALWAYS spikes upward at
the first iteration of Stage 2. This is expected:

- At p=2, intermediate densities (ρ ≈ 0.5) have stiffness 0.25 × E₀
- At p=3, the same elements now have stiffness 0.125 × E₀ (half)
- K drops → U increases → C = UᵀKU changes (complex interaction)

Typically compliance rises 10–30% at the stage boundary then decreases
monotonically. A spike larger than 50% suggests the Stage 1 topology was
poor (too many intermediate density elements) and Stage 2 may struggle.

### The min_iterations Parameter

min_iterations prevents premature convergence in early iterations when
the spread criterion can be satisfied trivially (e.g., compliance is flat
in iterations 1–10 of Stage 1 before topology has formed). Without it,
Stage 1 might declare convergence after 15 iterations with a gray field.

Setting: min_iterations = 30 for Stage 1, 50 for Stage 2.

These are FLOORS, not targets. The solver runs at least min_iterations
before checking either convergence criterion (spread or density change).

### Total Run Time

With confirmed working parameters at 1mm voxels:
    base_part (~120k elements): ~60 min total (Stage 1 + Stage 2)
    conrod    (~96k elements):  ~36 min total
    Typical iteration time: ~16s/iter on CPU CG path

GPU target (when GPU warm-start bug is fixed): ~3s/iter (5× improvement).

The iteration time scales roughly linearly with n_elements at fixed CG
iteration count, and the CG iteration count scales with κ(K), which
increases through the SIMP run as elements become more differentiated
(some near ρ_min, some near 1.0).


---

## 10. Convergence

### Two Convergence Criteria (Both Active)

**Criterion 1 — Compliance spread (primary)**

Check the relative spread of compliance over the last 10 iterations:

    spread = (max(C[-10:]) - min(C[-10:])) / max(C[-10:])

Declare convergence if: spread < spread_tol

Default spread_tol (from `types.rs` `spread_tol()` method): 1e-4

Physical meaning: compliance has changed less than 0.01% in the last
10 iterations. The objective has effectively stopped improving.

This is the preferred criterion because it directly measures whether the
objective is still making progress, regardless of whether individual element
densities are still changing (they can oscillate at very small amplitude
while compliance is flat).

**Criterion 2 — Density change (secondary)**

    rho_change = max_e |ρ_e^new - ρ_e^old|

Declare convergence if: rho_change < density_tol

Default density_tol (from `types.rs` `density_tol()` method): 0.0005

Physical meaning: no element changed its density by more than 0.05% in
the last iteration. Topology has frozen.

Both criteria require iteration count > min_iterations before checking.

### What Good Convergence Looks Like (Compliance History Shape)

**healthy_monotone** (best):
- Stage 1: initial dip, then rise over 5–10 iters, then monotone decrease
- Stage 2: spike at boundary, then strict monotone decrease to flat tail
- Final spread < 1e-4 before iteration limit
- This is what motor_mount and conrod produced

**mostly_monotone** (good):
- Like healthy_monotone but with occasional compliance increases of <1%
- Typical for larger problems where CG doesn't fully converge
- Acceptable if gray fraction is low

**early_plateau** (bad — trapped local minimum):
- Stage 2 compliance barely changes in iterations 10–30
- Run may not converge before iteration limit
- Action: decrease FILTER_RADIUS by ~30% OR change VOLUME_FRACTION ±5%

**oscillating** (bad — OC not working):
- Large compliance swings (>5%) between consecutive iterations
- Usually means: damping missing, move limit too large, or filter radius
  too small (raw sensitivities are noisy and the OC update overreacts)
- Action: reduce MOVE_LIMIT to 0.2, verify damping = 0.5

### Motor Mount Run Analysis (Reference Baseline)

From result.json (last confirmed good run):
    Stage 1 + Stage 2: 147 total iterations (from result.json for stage 2 alone)
    Stage 2 alone: 147 iterations, converged
    Final compliance: 0.04093 J
    Volume fraction: 0.38000 (target 0.38, error < 0.001%)
    Duration: 2066s (~34 min for stage 2 alone)

The compliance history shows clean monotone decrease from iter 8 onward
with spread < 1e-4 at convergence. This is the reference baseline for
"healthy" run behavior.


---

## 11. Boundary Conditions

### Fixed DOFs (Dirichlet Conditions)

Fixed DOFs represent the attachment points — where the part is bolted,
welded, or otherwise constrained to the world. They constrain displacements
to zero: U[fixed_dofs] = 0.

Implementation: Dirichlet conditions are imposed by row/column zeroing in K
with the diagonal set to diag_mean. This preserves the SPD property of K
while effectively removing the fixed DOFs from the system.

CRITICAL — Minimum Fixed DOF Count:
Too few fixed DOFs causes the condition number κ(K) to explode. Specifically,
if the fixed region is too small, the structure is nearly free to rotate
(near-rigid body modes), and the CG iteration count climbs to the max_cg_iter
cap without converging.

Confirmed threshold: ≥ 19,000 fixed DOFs needed for stable convergence on
disk-geometry parts (from conrod and motor_mount experience). Parts with
fewer fixed DOFs showed CG non-convergence and required either expanding the
fixed region or reducing the load.

Why DOF count matters: Rigid body modes in 3D have 6 degrees of freedom
(3 translation, 3 rotation). Each rigid body mode represents a direction in
which the structure can move without deforming. If K is not constrained
against all 6 rigid body modes, K is singular (or near-singular). The number
of fixed DOFs is not directly the count of constrained rigid body modes —
it is a proxy for the completeness of the constraint.

### Traction Loads (Neumann Conditions)

Loads are applied as surface tractions — force per unit area on the load
face, integrated to equivalent nodal forces.

Total force magnitude F_total [N] is specified in params.json as
load_magnitude_n. This is distributed uniformly across all load DOFs:

    load_vals[i] = F_total / n_load_dofs    for each DOF i in load face

For a 1mm voxel grid with a full face of area A mm², the number of load nodes
is approximately A mm² / (1mm²) = A. At 5000N over a 70mm×60mm face:

    Force per node ≈ 5000 / (70 × 60) ≈ 1.19 N/node

### Load Direction Convention

Load direction is specified by the primary_face in load_hints:
    "right" → force applied in -X direction (inward from right face)
    "top"   → force applied in -Z direction (downward from top face)

Force vector construction is in `voxelize.py` build_load_case(). The traction
is applied normal to the face, pointing inward (compressive convention).

### The BC-Compliance Relationship

The same part geometry with different BCs produces completely different
topology results:
- Fixed at bottom, loaded at top → vertical column structure
- Fixed at corners, loaded at center → arch/bridge structure
- Fixed at full face, loaded at point → fan/radiating structure

This is fundamental: SIMP finds the optimal topology FOR THE SPECIFIED LOAD
CASE. Changing the BCs requires a full re-run. There is no way to infer
"what if I changed the fixed face" from an existing result.

### The `center_disk` BC Selector

Parts using `center_disk` BCs (conrod, motor_mount) place fixed constraints
in disk regions centered on specific faces, matching physical bolt patterns.
The disk radius must be large enough to capture sufficient DOFs (≥19k threshold
above). The selector is implemented in `voxelize.py` `_fixed_dofs_from_config`.

Domain masking (void regions + non-design regions) is essential with disk BCs.
Without masking, SIMP will fill the bounding box including space that doesn't
exist on the physical part (inside the hole, outside the outer wall, etc.).


---

## 12. Material Library

### Steel (Structural / Mild) — CONFIRMED PIPELINE DEFAULT

    Young's modulus E:    210 GPa  = 210,000,000,000 Pa
    Poisson's ratio ν:    0.3      (dimensionless)
    Yield strength σ_y:   250 MPa  (ASTM A36, mild steel)
    Density ρ_mat:        7850 kg/m³

    Lamé parameters (derived):
        λ = E·ν / ((1+ν)(1-2ν)) = 210e9 × 0.3 / (1.3 × 0.4)
          = 63e9 / 0.52 = 121.15 GPa
        μ = E / (2(1+ν)) = 210e9 / 2.6 = 80.77 GPa

    In params.json / material dict:
        {"youngs_modulus_pa": 210000000000.0, "poissons_ratio": 0.3, "name": "steel"}

    In Rust solver material_rust dict:
        {"young": 210e9, "poisson": 0.3}

    These two representations are both in use. The Python path uses the
    snake_case `youngs_modulus_pa` form. The Rust path uses the short form.
    They must be kept in sync — the notebook Cell 1 defines `material` (Python
    form) and `material_rust` (Rust form) separately.

### Aluminum 6061-T6 — NOT YET IN PIPELINE (Reference)

    Young's modulus E:    68.9 GPa
    Poisson's ratio ν:    0.33
    Yield strength σ_y:   276 MPa
    Density ρ_mat:        2700 kg/m³

    Lamé parameters:
        λ = 68.9e9 × 0.33 / (1.33 × 0.34) = 22.74e9 / 0.4522 = 50.3 GPa
        μ = 68.9e9 / (2 × 1.33) = 25.9 GPa

    Use case: FDM or CNC aluminum parts where weight is critical.
    Note: Lower E means higher compliance at same load → different optimal
    topology. Cannot directly compare steel/aluminum compliance values.

### PLA (FDM Printing) — NOT YET IN PIPELINE (Reference)

    Young's modulus E:    ~3.5 GPa  (highly variable, depends on print params)
    Poisson's ratio ν:    0.36
    Yield strength σ_y:   ~50 MPa

    IMPORTANT: PLA is anisotropic when FDM-printed. The stiffness normal to
    print layers (Z-axis for a flat build plate) is typically 50–70% of the
    in-plane stiffness. SIMP with isotropic material properties will
    overestimate stiffness for FDM prints by 30–50%.

    For FDM: Use E = 2.0–2.5 GPa as a conservative estimate to account for
    layer adhesion weakness. Or multiply the safety factor threshold by 1.5.

### PETG (FDM Printing) — NOT YET IN PIPELINE (Reference)

    Young's modulus E:    ~2.1 GPa
    Poisson's ratio ν:    0.38
    Yield strength σ_y:   ~40 MPa

### How Material Choice Affects SIMP

The topology (structural skeleton shape) is independent of E for a given load
case and volume fraction. This is because compliance C = Uᵀ K U scales as
C ~ F²/(E·V), and the optimal topology minimizes C/E — which is the same for
any E.

However, changing ν affects the optimal topology (different Lamé parameters
mean different stress states). The effect is typically small for ν in [0.2, 0.4].

Safety factor computation depends heavily on E and σ_y — a PLA part designed
with steel parameters will appear to have enormous safety factors but is
actually much weaker than predicted.

### Safety Factor Estimation (Current NB06 Method)

The pipeline uses compliance-based safety factor estimation:

    σ_rms = sqrt(E · C / V)           [Pa]
    σ_est = K_t × σ_rms               [Pa, where K_t = 3.0 stress concentration]
    SF = σ_y / σ_est

This is a LOWER BOUND on safety factor. σ_rms < σ_max always, so the true
maximum stress is higher and the true SF is lower. The compliance-based
method is conservative in a specific sense: it gives the RMS stress, not the
peak stress. The K_t = 3.0 stress concentration factor partially corrects
for this by assuming worst-case geometric concentration.

For the motor_mount run at 5000N: SF = 10.94 (PASS, threshold 2.0).
A factor of 10+ is common for topology-optimized parts because the optimizer
finds near-globally-optimal load paths that avoid stress concentrations.


---

## 13. Parameter Reference

### Complete Parameter Table — Confirmed Production Values

| Parameter         | Value   | Unit   | Location            | Effect if Too High          | Effect if Too Low            |
|-------------------|---------|--------|---------------------|-----------------------------|-----------------------------|
| VOXEL_SIZE_MM     | 1.0     | mm     | Cell 0              | Fast but coarse features    | Slow but fine features       |
| VOLUME_FRACTION   | 0.45    | ratio  | Cell 0 / params.json| Heavy result, safe          | Light result, may fail SF   |
| PENAL (Stage 2)   | 3.0     | —      | Cell 0              | Binary, ill-conditioned     | Gray mush, not 0/1           |
| FILTER_RADIUS     | 3.0     | mm     | Cell 0 / params.json| Blob topology               | Checkerboard / spikes        |
| MAX_ITERATIONS    | 200     | iters  | Cell 0              | (wastes time if converged)  | May not converge             |
| CONVERGENCE_TOL   | 0.0005  | Δρ     | Cell 0              | Premature stop              | Wastes iterations at end     |
| MOVE_LIMIT        | 0.3     | Δρ/iter| Cell 0             | Oscillation, disconnection  | Very slow convergence        |
| CHECKPOINT_EVERY  | 10      | iters  | Cell 0              | I/O overhead                | Lost work on crash           |
| MAX_CG_ITER       | 2000    | iters  | Cell 0              | (no effect if converges)    | CG non-convergence           |
| damping           | 0.5     | ratio  | solver config       | Over-damped, slow           | 2-cycle oscillation          |
| min_iterations    | 30/50   | iters  | Cell 3 run_stage()  | Premature convergence       | Extra iters after convergence|
| ρ_min             | 1e-3    | ratio  | types.rs RHO_MIN    | Singular K (crash)          | Ill-conditioned K            |

### Per-Part Parameter Overrides

The "simp" block in `scad/{part_name}_params.json` overrides notebook defaults.
Only keys present in the block override — absent keys keep notebook defaults.
Applied in Cell 1 after params load.

Example params.json simp block:
    "simp": {
        "voxel_size_mm":   1.0,
        "volume_fraction": 0.45,
        "filter_radius":   3.0,
        "max_iterations":  200,
        "convergence_tol": 0.0005,
        "move_limit":      0.3,
        "penal":           3.0
    }

### Parameter Sensitivity Guide

**Volume fraction** is the most impactful parameter for the end user:
- Lower VF (0.3) → lighter, more skeletal, less safe margin
- Higher VF (0.5) → heavier, more solid, more conservative
- For FDM parts: start at 0.45, reduce if over weight budget
- For metal (CNC/SLS): can go as low as 0.25 if safety factor allows

**Filter radius** is the most impactful parameter for topology quality:
- Always set to 2–4× voxel size
- If results look spiky: increase by 50%
- If results look blobby: decrease by 30%
- Never go below 2× voxel size

**Penalization** should almost never be changed from 3.0 for the final stage.
Change it only if:
- p=3 produces non-convergence (very ill-conditioned K) → try p=2.5
- You want extremely binary results with no gray → try p=4 (with tighter
  CG tolerance, max_cg_iter=5000)


---

## 14. Diagnostic Interpretation

### Reading the `{part_name}_simp_diagnostic.json` Report

After each run, the diagnostic report contains:

**convergence block:**
- `shape`: The convergence curve classification (see Section 10)
- `final_spread_10iter`: Relative compliance spread over last 10 iters
  - < 1e-4: Converged cleanly
  - 1e-4 to 1e-3: Marginal — result likely okay
  - > 1e-3: Did not converge — treat result with caution
- `compliance_reduction_pct`: How much compliance dropped from iter 1 to end
  - < 10%: Run barely optimized — likely a bad starting point or parameter issue
  - 10–50%: Normal range for most problems
  - > 50%: Excellent optimization (simple load path, clear optimal topology)

**density_quality block:**
- `gray_fraction_design`: Fraction of design-domain elements with 0.2 < ρ < 0.8
  - < 5%: PASS — topology has committed to binary
  - 5–15%: WARN — partial commitment, STL will have over-smoothed regions
  - > 15%: FAIL — topology has NOT committed, do not trust the STL
- `solid_fraction`: Fraction of ALL elements (including non-design) with ρ > 0.5
  - Should be approximately VOLUME_FRACTION when non-design fraction is small

**volume_constraint block:**
- `error_pct`: |achieved_vf - target_vf| / target_vf × 100
  - Should be < 0.1% for a well-functioning OC bisection
  - If > 1%: bisection may have failed (l2 bound issue) or non-design mask is wrong

### Reading the Slice Plots

The three orthogonal mid-plane slices (XY, XZ, YZ) show density on a
0–1 grayscale (white = void, black = solid). What to look for:

**Good result:**
- Clear binary contrast — black struts on white background
- Continuous load path visible from load face to fixed face
- No isolated black islands disconnected from the main structure
- Strut width ≥ 2–3 voxels (anything thinner won't survive marching cubes)

**Bad result — checkerboard:**
- Alternating pixel-level black/white pattern
- Cause: filter radius too small
- Fix: increase FILTER_RADIUS by 50%, rerun from Cell 3

**Bad result — gray cloud:**
- Large regions of medium gray (ρ ≈ 0.5)
- Cause: penalization too weak or run didn't converge
- Fix: check gray_fraction_design; if > 15%, increase PENAL or rerun longer

**Bad result — blob:**
- Large solid blobs at load/fix points with thin connections
- Cause: filter radius too large
- Fix: decrease FILTER_RADIUS by 30%

**Bad result — disconnected islands:**
- Multiple solid regions with no connection
- Cause: BC or void mask issue (solid regions can't "see" each other)
  OR move limit too large (material was removed too aggressively)
- Fix: verify void_mask and nondesign in domain mask plot (Cell 3 output)

### Interpreting the Compliance Curve

The `{part_name}_convergence.png` dashboard has 6 panels. Most important:

**Top-left (compliance log scale):**
- Should show initial rise then monotone decrease
- Final value should be flat (horizontal) for at least 20 iterations

**Bottom-left (10-iter spread):**
- Crosses the red dashed line (1e-4 threshold) when converged
- If it never crosses: run didn't converge in allocated iterations

**Top-right (per-iteration improvement):**
- Most bars should be positive (compliance decreasing)
- A few negative bars are okay (< 20% of total)
- > 30% negative bars → oscillation problem


---

## 15. Known Failure Modes

### FM-01: CG Non-Convergence (SOLVE! flag in output)

**Symptom:** Line ends with `[SOLVE!]` in solver output:
    Iter   1 | C=0.0123 | Vol=0.450 | Δρ=0.4500 | p=2 | CG=2000 | res=1.2e-02 | 16.1s [SOLVE!]

**Cause:** CG hit max_cg_iter (2000) without reaching residual tolerance (1e-6).
The solution U is inaccurate, which corrupts the sensitivities.

**Not always fatal:** In early SIMP iterations, topology is developing and
small inaccuracies in U don't matter much. If `[SOLVE!]` appears only in
iterations 1–20 then disappears, the run is likely fine.

**Fatal if:** `[SOLVE!]` persists through iteration 50+. The sensitivity
field is too noisy to drive useful topology development.

**Fix options:**
1. Increase MAX_CG_ITER to 5000 (CPU path) — costs more time per iteration
2. Verify fixed_dofs count ≥ 19k (under-constrained K is the most common cause)
3. Reduce VOXEL_SIZE_MM slightly (finer mesh → smaller element stiffness
   ratios → better conditioning)
4. Verify void_mask is not creating a disconnected structure (disconnected
   subdomains make K block-singular)

### FM-02: Gray Result (High Gray Fraction)

**Symptom:** gray_fraction_design > 15% in diagnostic report

**Cause:** SIMP failed to drive elements to binary densities.

**Most common cause:** Compliance converged (spread criterion met) before
topology committed. Happens when:
- min_iterations too low → converged during the initial gray phase
- PENAL too low (p < 3 for Stage 2)
- Volume fraction too high (optimizer has so much material budget that
  intermediate densities ARE the optimal solution)

**Fix:**
1. Increase min_iterations for Stage 2 to 80+
2. Verify p=3.0 for Stage 2 (check stage02 problem JSON)
3. Reduce VOLUME_FRACTION by 0.05

### FM-03: Blob Topology (filter radius too large)

**Symptom:** STL looks like a solid brick with slightly rounded corners;
no struts visible. Compliance may still have converged.

**Cause:** Filter radius much larger than 3× voxel size. The filter averaging
region is so large that sensitivity variations are smoothed out — every element
sees approximately the same averaged sensitivity, so OC has no guidance on
where to place material. The optimizer defaults to uniform density distribution.

**Historical failure case (confirmed):**
    VOXEL_SIZE_MM=1.0mm, FILTER_RADIUS=8.0mm → blob topology
    VOXEL_SIZE_MM=1.0mm, FILTER_RADIUS=3.0mm → correct strut topology

**Fix:** Reduce FILTER_RADIUS. Rule: r = 3× voxel size as default.

### FM-04: Pycache/Kernel State Bug

**Symptom:** Python changes to .py files in src/ have no effect; old behavior
persists even after saving.

**Cause:** The Jupyter kernel has the old .pyc cached. The kernel must be
hard-restarted AND `__pycache__` cleared.

**Fix (always do this when editing .py source files):**
    # Inside container:
    find /workspace/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    find /workspace/scripts -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    # Then: Kernel → Restart Kernel and Clear All Outputs

### FM-05: Checkpoint Stale on Re-Run

**Symptom:** A fresh re-run of NB04 resumes from a previous run's checkpoint
rather than starting fresh.

**Cause:** `outputs/problem/checkpoint.bin` and `checkpoint_meta.json` from
a previous run are still on disk. The solver always checks for checkpoints
before reading x_init.

**Fix:**
    rm -f outputs/problem/checkpoint.bin outputs/problem/checkpoint_meta.json

The session_start.sh script prompts about this when a checkpoint is detected.

### FM-06: Stage04 Handoff Mismatch

**Symptom:** NB05 or NB06 reads stale config from a previous run's stage04.json

**Cause:** NB04 was interrupted after writing stage04.json but before the
run completed. The stage04.json reflects the config but not the actual result.

**Fix:** Always run NB04 to completion. The stage04.json is written at the
end of Cell 6, after all computation is complete. If NB04 crashed mid-run,
delete stage04.json and rerun NB04 fully.

### FM-07: Disconnected Topology Islands in STL

**Symptom:** STL contains multiple separate solid bodies (islands). Visible
in mesh validation as Euler number anomaly or in 3D viewer as floating pieces.

**Cause:** Material removal was too aggressive (topology disconnected) OR
void mask creates regions with no path between load and fixed faces.

**Fix:**
1. Check domain mask plot (Cell 3 output): verify there is a continuous
   design domain between load face and fixed face
2. Reduce MOVE_LIMIT to 0.2
3. Increase FILTER_RADIUS slightly (larger smoothing prevents local disconnection)
4. In NB05 STL export: island removal is applied (keeps only the largest
   connected component). If the main structure IS connected, islands are
   already removed in post-processing.

### FM-08: Circular Dependency Between Modules

This is a code architecture note, not a runtime failure.

**Pattern:** When module A imports module B and module B needs A, Python/Rust
raise circular import errors.

**Solution used:** `vcycle_dispatch.rs` shim — instead of multigrid.rs
importing solver.rs and solver.rs importing multigrid.rs, vcycle_dispatch.rs
imports both and provides the combined interface. Neither core module imports
the other.

This pattern should be followed for any future cross-module dependencies
in the Rust solver.


---

## 16. Pipeline-Specific Implementation Notes

### Note 1: Compliance Computation (Physical Units)

In `simp.py` (Python path), compliance is computed as:

    compliance = float(np.dot(x**p * cell_volumes, strain_energies))    [Joules]

The `cell_volumes` term converts from strain energy density [Pa] to strain
energy [J]. This gives physically meaningful compliance values that don't
depend on mesh resolution.

In `sensitivity.rs` (Rust path), the compliance is:

    C = Σ_e ρ_e^p · u_eᵀ · K_e · u_e

Where K_e already includes the voxel volume (from the FEM integration
over the element volume), so the compliance is in Joules without needing
an explicit cell_volumes multiplication.

Both paths produce compliance in Joules. Comparison between paths is valid.

### Note 2: Coordinate System and Units

The Rust solver works in METRES throughout:
    voxel_size [m] = VOXEL_SIZE_MM / 1000.0

The Python/FEniCSx path auto-detects:
    if max(coordinates) > 1.0: coordinates are in mm → convert to m

Compliance is in Joules [N·m] because forces are in Newtons and displacements
are in metres. If coordinates are in mm and forces are in N, compliance would
be in [N·mm] = [mJ]. The auto-conversion ensures consistency.

### Note 3: Non-Design Domain Handling

Non-design elements are ALWAYS forced to ρ = 1.0 and EXCLUDED from the
volume fraction computation. This is essential for parts with bolt holes,
walls, and other geometric features that must remain solid.

In the OC update:
    nondesign_mask elements: ocp set to 0.0 (no OC update)
    nondesign_mask elements: x_new forced to 1.0 after bisection

In the volume constraint:
    vol = (x_new[design_mask] × cell_volumes[design_mask]).sum()
          / cell_volumes[design_mask].sum()

The non-design volume is NOT counted toward the volume budget. This means
the actual total solid volume is:

    total_solid = (design_solid_volume) + (nondesign_volume)
                = VF × design_volume + nondesign_volume

Which can be larger than VF × total_volume. The NB06 validation reports
volumetric STL size in mm³ which reflects the full solid volume including
non-design regions.

### Note 4: Papermill Parameter Injection Limitation

Papermill can inject scalar parameters (int, float, string, bool) via `-p`.
It CANNOT inject dictionaries or nested structures via `-p`. This means
SIMP_OVERRIDES style injection of dict parameters fails silently.

Confirmed workaround: use per-part `params.json` simp blocks (Section 13)
instead of Papermill dict injection. The Cell 1 params loading reads these
automatically.

### Note 5: mtime-Based Handoff Auto-Detection

Cells that auto-detect the most recent stage handoff file use:

    candidates = sorted(glob("*_stageXX.json"), key=lambda p: p.stat().st_mtime)

This is mtime-based, NOT alphabetical. Alphabetical sorting fails when
two runs on the same day produce files with the same date prefix — the
wrong file gets picked up. mtime ensures the most recently written file
is selected. This was a confirmed bug that was fixed in the Tier 5 audit.

### Note 6: The GPU Warm-Start Bug (Tech Debt)

The GPU CG solver (`gpu_solver.rs`) zeros `u` (displacement vector) at the
start of each call. On the CPU path, `u` is initialized to the previous
iteration's displacement, which typically reduces CG iterations by 20–40%
(warm-starting is especially effective when displacements change slowly,
which they do in late SIMP iterations).

Impact: GPU path uses more CG iterations per solve than theoretically necessary.
The result is CORRECT — CG with zero initialization converges to the same
solution, just takes more iterations.

Fix required: pass `u` from the previous iteration as the initial guess for
the GPU CG solve. This requires surfacing `u` through the GPU solver API
as an in-out parameter.

This is the PRIMARY GPU performance optimization remaining after the GPU
SpMV implementation.

### Note 7: The Filter is Built Once, Reused Every Iteration

The filter sparse matrix omega (shape: n_elem × n_elem) is computed once
before the SIMP loop and reused at every iteration. It is an O(n_elem × k)
matrix where k ≈ (4/3)π(r/h)³ is the average number of neighbors per element.

At r=3mm, h=1mm: k ≈ 4/3 × π × 27 ≈ 113 neighbors per element.
For 96k elements: omega has ~10.8M nonzeros.

The filter application per iteration is two sparse matrix-vector multiplies:
    O(2 × nnz) = O(21.6M) operations — fast (< 1s on CPU).

Rebuilding the filter every iteration would be O(n_elem × k × log(n_elem))
due to the KDTree query — about 100× more expensive. Never rebuild mid-run.

---

## Appendix A: Mathematical Notation Reference

| Symbol   | Meaning                                        | Units       |
|----------|------------------------------------------------|-------------|
| Ω        | Structural domain (bounding box)               | m³          |
| ρ, ρ_e   | Element density (design variable)              | —           |
| ρ_min    | Minimum density (prevents singular K)          | — (= 1e-3)  |
| u        | Displacement field / vector                    | m           |
| U        | Global displacement vector (all DOFs)          | m           |
| σ        | Stress tensor                                  | Pa          |
| ε        | Strain tensor                                  | —           |
| E        | Young's modulus                                | Pa          |
| ν        | Poisson's ratio                                | —           |
| λ, μ     | Lamé parameters                                | Pa          |
| K        | Global stiffness matrix                        | N/m         |
| K_e      | Element stiffness matrix (24×24 for hex)       | N/m         |
| F        | Global force vector                            | N           |
| C        | Compliance (objective)                         | J           |
| p        | Penalization exponent (SIMP)                   | — (= 3.0)   |
| r        | Filter radius                                  | m or mm     |
| V*       | Target volume (= VF × |Ω|)                    | m³          |
| VF       | Volume fraction target                         | — (= 0.45)  |
| dc       | Raw sensitivity ∂C/∂ρ_e                        | J (≤ 0)     |
| omega    | Filter weight sparse matrix                    | —           |
| ocp      | OC numerator (= ρ_e × sqrt(-dc_filtered_e))   | —           |
| l_mid    | Lagrange multiplier (volume constraint)        | —           |
| move     | Move limit (max Δρ per iteration)              | — (= 0.3)   |
| κ(K)     | Condition number of stiffness matrix           | —           |

---

## Appendix B: Quick Reference — "What Parameter Do I Change?"

| Problem Observed                              | Change                              |
|-----------------------------------------------|-------------------------------------|
| Result is too heavy                           | Reduce VOLUME_FRACTION by 0.05      |
| Result is too light / fails safety factor     | Increase VOLUME_FRACTION by 0.05    |
| STL is spiky / has checkerboard texture       | Increase FILTER_RADIUS by 50%       |
| STL is blobby / no struts visible             | Decrease FILTER_RADIUS by 30%       |
| Gray fraction > 15%                           | Increase min_iterations or PENAL    |
| Run oscillates (compliance up and down)       | Reduce MOVE_LIMIT to 0.2            |
| CG [SOLVE!] in early iters only              | Normal — no action needed           |
| CG [SOLVE!] persists past iteration 50        | Check fixed DOF count (≥ 19k)       |
| Run converges in < 30 iterations             | Check min_iterations setting        |
| Run doesn't converge in 200 iterations        | Increase MAX_ITERATIONS to 400      |
| Disconnected islands in STL                   | Reduce MOVE_LIMIT; check void_mask  |
| Python source change has no effect            | Clear __pycache__, restart kernel   |
| Fresh run resumes old checkpoint              | rm outputs/problem/checkpoint.bin   |
| Stage04 has wrong params                      | Check if NB04 completed fully       |


---

## 17. Shell Enforcement (NB05 Post-Processing)

### What It Is

Shell enforcement is a post-processing step applied to the density array
BEFORE marching cubes runs in NB05. It forces the outer voxels of selected
faces to solid (density=1.0), creating a continuous closed skin that encloses
whatever the optimizer built inside. The key architectural property is that
it operates on the same density array that marching cubes consumes — the shell
and the interior structure are one continuous field, not two geometries being
joined. This avoids the junction artifacts that plagued the slab approach.

### Why It Is Necessary

SIMP compliance minimization has no connectivity constraint. For a motor mount
with 4 fixed disks on x_min and 4 load disks on x_max, the mathematically
optimal topology is 4 independent load paths (struts) — one connecting each
fixed point to its nearest load point. These struts are disconnected from each
other because no force requires them to be connected.

Shell enforcement solves this by giving all struts a common plate to attach
to at each end. The struts themselves are unchanged; they are enclosed within
the shell rather than replaced by it.

### Void-Aware Mechanism

The shell MUST respect the void_mask from the optimizer. If the shell blindly
forces all outer voxels to solid, it fills the bolt holes. The implementation
loads `outputs/problem/void.bin` and applies:

    density_shelled[face_slice] = np.where(void_mask[face_slice],
                                            density[face_slice],   # preserve void
                                            1.0)                   # force solid

This means: voxels that the solver declared void stay void; everything else
on the selected face gets forced solid.

### Parameters (Cell 0 of NB05)

```python
SHELL_ENFORCE_FACES = ["x_min", "x_max", "y_min", "y_max"]
SHELL_THICKNESS_MM  = 8.0   # mm — must match or exceed bolt_seat seat_depth_m
```

**SHELL_ENFORCE_FACES**: which faces to reinforce.
- `["x_min", "x_max"]` — wall plate + motor plate only (minimal)
- `["x_min", "x_max", "y_min", "y_max"]` — full frame including side walls
  (confirmed working for motor mount)
- `[]` — disabled (pass-through, no shell)

**SHELL_THICKNESS_MM**: must be ≥ `seat_depth_m` from bolt_seats in params.json.
If SHELL_THICKNESS_MM < seat_depth_m, the shell does not reach the strut
anchor zone and Taubin smoothing removes the thin shell between anchor points.

CONFIRMED WORKING: SHELL_THICKNESS_MM=8.0 with seat_depth_m=0.015 (15mm).
Shell thickness of 2mm was confirmed INSUFFICIENT — smoothing erased it.

### What SHELL_THICKNESS_MM Should NOT Be

Setting SHELL_THICKNESS_MM=15.0 (matching seat_depth_m exactly) creates a
very thick forced-solid region that the optimizer's struts must terminate
into. This works structurally but produces excessive forced material. 8mm is
the confirmed sweet spot: thick enough to survive smoothing, thin enough to
leave the optimizer with a meaningful interior design space.

---

## 18. Through-Ring Passive Elements

### The Concept

A through-ring is a thin forced-solid annulus around each bolt void that runs
the FULL axis length of the part. It is the answer to the question: "how do
we tell the optimizer the bolt path exists without consuming the volume budget?"

The optimizer sees:
- Void core (bolt hole): radius = void_radius_m, full axis length
- Forced-solid ring: void_radius_m < r < through_ring_radius_m, full axis
- Design space: r > through_ring_radius_m, optimizer decides

The ring is "invisible" in the sense that it does not become significant STL
geometry — it is 1mm thick at 1mm voxels, a single annular shell. But it
tells the optimizer that material must exist around the bolt path everywhere,
not just at the entry/exit collar zones.

### Why Previous Approaches Failed

**Full-depth collar (wall_radius_m full length):** Forces a thick cylinder of
nondesign material for the entire part depth. At 8mm collar radius, 70mm
depth, 8 bolts: ~93,000 mm³ = 40% of total material budget consumed before
optimization. The optimizer has no freedom and produces 8 solid columns with
minimal bridging material. CONFIRMED FAILURE.

**Entry/exit collar only (original bolt_seat):** The collar ring only exists
for 15mm at each end. The middle 40mm has a void core floating in free space
with no incentive for the optimizer to surround it. Marching cubes produces
clean bores at the collar zones and open channels in the middle. CONFIRMED
FAILURE (produces corner cutouts instead of through-holes).

**Through-ring (current implementation):** 1mm annulus, full depth. Volume
cost: π×((4²−3²))×70×8 ≈ 15,400 mm³ = 4.6% of budget. CONFIRMED WORKING.

### Volume Budget Math

At 1mm voxels for a motor mount (70×60×80mm = 336,000 element grid):

    through_ring volume = π × (r_ring² − r_void²) × axis_length × n_bolts
                       = π × (0.007² − 0.006²) × 0.070 × 8  [m³]
                       = π × (4.9e-5 − 3.6e-5) × 0.560
                       ≈ 2.29e-5 m³ = 22,900 mm³

    Grid total volume = 70 × 60 × 80 = 336,000 mm³
    Budget fraction   = 22,900 / 336,000 = 6.8%

6.8% of budget gives the optimizer clear anchor geometry without dominating.
By comparison, the failed full-collar approach consumed 40%.

### Implementation

**voxelize.py** — added after existing entry/exit collar logic in Phase 4:

```python
if hasattr(region, 'through_ring_radius_m') \
        and region.through_ring_radius_m is not None:
    ring_r2  = region.through_ring_radius_m ** 2
    ring_mask = (r2 >= void_r2) & (r2 < ring_r2)
    nondesign[ring_mask] = 1
```

No `axis_coord` constraint — the ring applies for the full axis length.
The `r2 >= void_r2` condition ensures the void core is not overwritten.

**param_schema.py** — `BoltSeatRegion` dataclass:

```python
through_ring_radius_m: Optional[float] = None
```

Optional field, defaults None (backward compatible — existing params.json
files without this field behave identically to before).

**motor_mount_params.json** — both bolt_seat groups:

```json
"through_ring_radius_m": 0.007
```

This sets a 1mm ring (void=6mm, ring=7mm) for all 8 bolts.

### Confirmed Results

Motor mount run with through_ring_radius_m=0.007:
- Both x_min (wall) and x_max (motor) faces show clean bolt holes
- Holes are surrounded by material rather than being corner cutouts
- Compliance: 0.03583 J (best result yet — optimizer building better
  structure around the ring anchor points)
- Safety factor: 11.09 (PASS)
- Watertight: true, open edges: 0

### Tuning Guide

| through_ring_radius_m | Effect |
|---|---|
| None | No ring — entry/exit collar only (old behavior) |
| void_radius + 0.001 | 1mm ring — minimal volume cost, clean bores |
| void_radius + 0.002 | 2mm ring — cleaner marching cubes surface at holes |
| void_radius + 0.005 | 5mm ring — approaches failed full-collar behavior |

Start at void_radius + 0.001. Increase by 0.001 at a time if holes are
still irregular in the STL. Do not exceed void_radius + 0.003 without
re-checking the volume budget fraction.

---

## 19. Confirmed Working Configuration (Motor Mount v3)

This is the confirmed production configuration as of 2026-05-24.
Use as the reference baseline for new parts.

### Optimization Parameters
```
VOXEL_SIZE_MM:    1.0
VOLUME_FRACTION:  0.38
FILTER_RADIUS:    3.0  (3× voxel size — confirmed rule)
MAX_ITERATIONS:   200
CONVERGENCE_TOL:  0.0005
MOVE_LIMIT:       0.3
PENAL stage1:     2.0
PENAL stage2:     3.0
```

### Bolt Seat Parameters
```
void_radius_m:         0.006  (6mm hole clearance)
wall_radius_m:         0.009  (9mm entry/exit collar)
seat_depth_m:          0.015  (15mm collar depth)
through_ring_radius_m: 0.007  (7mm full-depth ring)
```

### Shell Enforcement Parameters (NB05)
```
SHELL_ENFORCE_FACES:  ["x_min", "x_max", "y_min", "y_max"]
SHELL_THICKNESS_MM:   8.0
```

### Run Results
```
Iterations:     238 (converged)
Final C:        0.03583 J
Gray fraction:  <5% (PASS)
Safety factor:  11.09 (PASS)
Watertight:     true
Open edges:     0
```