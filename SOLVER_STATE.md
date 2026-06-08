# SOLVER_STATE.md
# fenics-pipeline — Solver Architecture, Module Map, and Implementation State
#
# PURPOSE
# ───────
# This document is the authoritative reference for the current state of the
# fenics-pipeline Rust solver and Python pipeline code. Any Claude session
# working on this project should load this document alongside MATERIALS_SPEC.md.
#
# MATERIALS_SPEC.md answers: "what is the math?"
# SOLVER_STATE.md  answers: "what is the code, where does it live, and what
#                            decisions were made and why?"
#
# STRUCTURE
# ─────────
# 1.  Repository Layout — file tree with purpose annotations
# 2.  Rust Solver Module Map — every .rs file, its role, its public API
# 3.  Solver Dispatch Architecture — how a linear solve gets routed
# 4.  Data Flow — from params.json to STL, step by step
# 5.  Python Pipeline Module Map — every .py file in src/ and scripts/
# 6.  Notebook Pipeline — cell-by-cell summary of each notebook
# 7.  Inter-Stage Handoff Schema — the JSON files that connect notebooks
# 8.  Key Architectural Decisions — why things are structured this way
# 9.  Test Suite — what is tested, where, and how to run
# 10. Known Tech Debt — confirmed issues deferred to future sessions
# 11. Git History Summary — major milestones and their commits
# 12. Development Environment — container, paths, build commands
# 13. Session Startup Protocol — the correct sequence to begin a work session
#
# LAST UPDATED: 2026-06-07
# CURRENT HEAD: main branch (R0 docs reconciled + R1 multi-load complete)
# TEST COUNT:   141 passing (cargo test --release)
# SOLVER STATE: AMGCL AMG-PCG is the primary solver for n_dof ≥ 50k (smoothed
#               aggregation + ILU(0) smoother, block_size=3, OpenMP); faer
#               sparse Cholesky below 50k; CPU Jacobi-PCG as fallback and
#               per-iteration corrector. Legacy GPU ILU(0)-PCG is deprecated
#               (diverges under SIMP contrast; not in any default build).
#               VCycle geometric multigrid is experimental and NOT in dispatch
#               (assumes power-of-2 grids). See §3 for the dispatch tree.
# ─────────────────────────────────────────────────────────────────────────────


---

## 1. Repository Layout

```
fenics-pipeline/
├── MATERIALS_SPEC.md          ← Physics and math reference (this project)
├── SOLVER_STATE.md            ← This document
├── README.md
├── PATCHES.md                 ← Change log / patch notes
├── Dockerfile
├── docker-compose.yml         ← v1 syntax (not v2) — do not upgrade without testing
├── Makefile                   ← build-solver, test, run targets
│
├── bin/
│   └── simp_solver            ← Compiled Rust binary (deployed here, not target/)
│
├── notebooks/
│   ├── 00_env_validation.ipynb
│   ├── 00_generate_test_cases.ipynb
│   ├── 00_import_step.ipynb
│   ├── 01_geometry_openscad.ipynb     ← NB01: SCAD → STL
│   ├── 02_mesh_gmsh.ipynb             ← NB02: STL → XDMF mesh (or stub)
│   ├── 03_fea_fenicsx.ipynb           ← NB03: FEA validation
│   ├── 04_simp_optimization.ipynb     ← NB04: SIMP (primary solver notebook)
│   ├── 05_stl_export.ipynb            ← NB05: density → STL
│   ├── 06_part_validation.ipynb       ← NB06: quality checks
│   ├── pipeline_full.ipynb            ← Orchestrator (calls NB01–NB06 via Papermill)
│   └── outputs/                       ← Executed notebook outputs (gitignored)
│
├── outputs/
│   ├── executed_nbs/                  ← Papermill output notebooks
│   ├── meshes/                        ← XDMF, NPY, STL, stage JSON files
│   ├── problem/                       ← Solver I/O (problem.json, binary files)
│   │   ├── fixed_dofs.bin             ← shared supports (all load cases)
│   │   ├── load_dofs_0.bin            ← per load case: load_dofs_{i}.bin
│   │   ├── load_vals_0.bin            ← per load case: load_vals_{i}.bin
│   │   ├── nondesign.bin
│   │   ├── void.bin
│   │   ├── problem_s1.json            ← Stage 1 problem spec
│   │   ├── problem_s2.json            ← Stage 2 problem spec
│   │   ├── density.bin                ← Solver output (float32)
│   │   ├── result.json                ← Solver output (per-stage summary)
│   │   ├── checkpoint.bin             ← Mid-run checkpoint (deleted on clean exit)
│   │   └── checkpoint_meta.json       ← Checkpoint metadata
│   └── reports/                       ← PNGs, validation JSON, diagnostic JSON
│
├── scad/
│   ├── {part_name}.scad               ← OpenSCAD geometry definition
│   └── {part_name}_params.json        ← Part parameters + SIMP overrides
│
├── scripts/
│   ├── voxelize.py                    ← Domain masking + load case builder
│   ├── build_solver.sh                ← Compiles Rust, copies binary to bin/
│   ├── session_start.sh               ← Session startup script
│   └── simp_dashboard.sh              ← tmux dashboard for monitoring runs
│
├── src/
│   ├── fea/
│   │   ├── boundary_conditions.py     ← FEniCSx BC construction
│   │   └── solver.py                  ← FEniCSx solve wrapper (legacy path)
│   ├── geometry/
│   │   ├── openscad_runner.py         ← SCAD → STL subprocess wrapper
│   │   ├── param_schema.py            ← PipelineParams dataclass hierarchy
│   │   └── region_factory.py          ← Resolves geometry to void/nondesign regions
│   ├── meshing/
│   │   ├── gmsh_pipeline.py           ← STL → XDMF via gmsh Python API
│   │   └── mesh_quality.py            ← Aspect ratio, Jacobian checks
│   └── optimization/
│       └── simp.py                    ← FEniCSx SIMP loop (legacy / fallback path)
│
├── solver/                            ← Rust solver crate
│   ├── Cargo.toml
│   ├── Cargo.lock
│   ├── build.rs                       ← amgcl C++ wrapper build script
│   └── src/
│       ├── main.rs                    ← Entry point: reads problem JSON, calls run_simp
│       ├── types.rs                   ← Grid, Problem, SimpConfig, SolveResult structs
│       ├── assembly.rs                ← K matrix assembly, Dirichlet BC application
│       ├── connectivity.rs            ← Element-node connectivity, DOF mapping
│       ├── filter.rs                  ← Sensitivity filter weight matrix
│       ├── io.rs                      ← Binary I/O, checkpoint read/write, JSON serde
│       ├── ke_base.rs                 ← Element stiffness matrix (computed once)
│       ├── oc_update.rs               ← OC bisection update rule
│       ├── sensitivity.rs             ← Compliance + sensitivity computation
│       ├── simp.rs                    ← Main SIMP optimization loop
│       ├── solver.rs                  ← Linear solvers (Cholesky, Jacobi-CG, GPU CG)
│       ├── multigrid.rs               ← VCycle GMG preconditioner
│       ├── vcycle_dispatch.rs         ← Dispatch shim (breaks multigrid↔solver cycle)
│       ├── preconditioner.rs          ← Preconditioner trait definition
│       ├── gpu_solver.rs              ← cuSPARSE ILU(0)-CG GPU path
│       └── amgcl_solver.rs            ← amgcl algebraic multigrid (experimental)
│
├── tests/                             ← Python integration tests
│   ├── test_bolt_seat.py
│   ├── test_fea_smoke.py
│   ├── test_mesh_quality.py
│   ├── test_param_schema.py
│   ├── test_region_factory.py
│   ├── test_skip_meshing_wiring.py
│   ├── test_slabs.py
│   ├── test_solver_interface.py
│   └── test_voxelize_selectors.py
│
└── test_cases/                        ← Parameterized test suite for pipeline sweeps
    ├── sweep_config.json
    └── {variant}/
        ├── base_part.scad
        └── params.json
```


---

## 2. Rust Solver Module Map

### `types.rs` — Core Data Structures

**Purpose**: All shared types. Every other module imports from here. Nothing
imports INTO here (it is a leaf in the dependency graph).

**Key types**:

```rust
pub struct Grid {
    pub nx: usize,       // elements in X
    pub ny: usize,       // elements in Y
    pub nz: usize,       // elements in Z
    pub voxel_size: f64, // metres
}
```

Grid methods (authoritative DOF layout):
- `node_idx(ix, iy, iz) = ix + iy*(nx+1) + iz*(nx+1)*(ny+1)`
- `elem_idx(ix, iy, iz) = ix + iy*nx + iz*nx*ny`
- `n_elem() = nx * ny * nz`
- `n_nodes() = (nx+1)*(ny+1)*(nz+1)`
- `n_dof() = 3 * n_nodes()`
- `centroid(ix, iy, iz) → (f64, f64, f64)` — element center in metres

```rust
pub struct Material { pub young: f64, pub poisson: f64 }

pub struct SimpConfig {
    pub use_gpu: bool,
    pub volume_fraction: f64,
    pub penal: f64,
    pub filter_radius: f64,        // metres
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub convergence_tol: f64,
    pub compliance_spread_tol: Option<f64>,
    pub density_change_tol: Option<f64>,
    pub move_limit: f64,
    pub damping: f64,              // OC blend factor (0.5)
    pub checkpoint_every: usize,
    pub max_cg_iter: usize,
}
```

`SimpConfig` has two computed methods:
- `spread_tol() → f64`: returns `compliance_spread_tol.unwrap_or(1e-4)`
- `density_tol() → f64`: returns `density_change_tol.unwrap_or(convergence_tol)`

```rust
pub struct Problem {
    pub grid: Grid,
    pub material: Material,
    pub fixed_dofs: Vec<usize>,     // SHARED Dirichlet supports (all load cases)
    pub load_cases: Vec<LoadCase>,  // R1: one or more weighted load scenarios
    pub config: SimpConfig,
    pub nondesign: Vec<bool>,       // len = n_elem
    pub void_mask: Vec<bool>,       // len = n_elem
    pub x_init: Option<Vec<f64>>,   // warm-start density (Stage 2)
}

pub struct LoadCase {
    pub name: String,                    // scenario label (diagnostics / logs)
    pub weight: f64,                     // weight in the aggregated objective
    pub load_dofs: Vec<usize>,
    pub load_vals: Vec<f64>,
    pub fixed_dofs: Option<Vec<usize>>,  // RESERVED: per-case supports (future
                                         // phase). Must be None today — io.rs
                                         // rejects a per-case fixed_dofs_file.
}

pub struct SolveResult {
    pub converged: bool,
    pub n_iterations: usize,
    pub final_compliance: f64,
    pub final_volume_frac: f64,
    pub compliance_history: Vec<f64>,
    pub volume_history: Vec<f64>,
    pub duration_s: f64,
    pub peak_memory_mb: f64,        // currently always 0.0 (not implemented)
    pub final_density: Vec<f64>,    // len = n_elem, NOT serialised to JSON
}
```

`pub const RHO_MIN: f64 = 1e-3;`

**Multi-load aggregation (R1)**: `Problem` carries a SHARED `fixed_dofs` plus a
`Vec<LoadCase>`. All cases share the same supports, so the stiffness matrix K is
assembled and preconditioned (AMGCL hierarchy) ONCE per SIMP iteration; then each
case's RHS is solved and compliance/sensitivities are accumulated as a weighted
sum, C = Σ_i wᵢ·Cᵢ (sensitivities likewise — filtering is linear, so filtering the
summed dc equals summing filtered dc). Single-load is the k=1 special case and is
bit-for-bit the old single-load solve (validated: motor_mount reproduced its
0.03583 baseline through the k=1 path). Per-case supports (`LoadCase.fixed_dofs`)
are reserved for a future phase; today every case shares `Problem.fixed_dofs`.

---

### `main.rs` — Entry Point

**Purpose**: Reads `problem_sN.json` from the path given as argv[1],
deserialises it into `Problem`, calls `run_simp()`, writes `density.bin`
and `result.json` to the same directory.

**Binary invocation** (from NB04 Cell 3):
```
bin/simp_solver outputs/problem/problem_s1.json
```

The binary is at `bin/simp_solver`, NOT `solver/target/release/simp_solver`.
`make build-solver` (or `build_solver.sh`) compiles and copies.

**Module declarations**: All `mod` declarations are in `main.rs`. Every
module that exists must be declared here. When adding a new .rs file,
add `mod new_module;` to main.rs.

Current module list:
```rust
mod types;
mod connectivity;
mod ke_base;
mod filter;
mod assembly;
mod solver;
mod sensitivity;
mod oc_update;
mod simp;
mod multigrid;
mod preconditioner;
mod vcycle_dispatch;
mod gpu_solver;
mod io;
mod amgcl_solver;   // experimental, may be feature-gated
```

---

### `connectivity.rs` — Element-Node Connectivity and DOF Mapping

**Purpose**: Precomputes the element→node connectivity table and the
element→DOF mapping used by assembly and sensitivity.

**Key functions**:
- `precompute_connectivity(grid) → Vec<[usize; 8]>` — for each element,
  returns the 8 corner node indices (hex element ordering)
- `precompute_dof_map(grid) → Vec<[usize; 24]>` — for each element,
  returns the 24 DOF indices (8 nodes × 3 DOFs/node)

Node ordering within an element (counterclockwise bottom face, then top):
```
Bottom face (iz):     Top face (iz+1):
  3 ─── 2              7 ─── 6
  │     │              │     │
  0 ─── 1              4 ─── 5
```
i.e., `[node(ix,iy,iz), node(ix+1,iy,iz), node(ix+1,iy+1,iz), node(ix,iy+1,iz),
        node(ix,iy,iz+1), node(ix+1,iy,iz+1), node(ix+1,iy+1,iz+1), node(ix,iy+1,iz+1)]`

---

### `ke_base.rs` — Element Stiffness Matrix

**Purpose**: Computes the 24×24 element stiffness matrix K_e for a unit
hexahedral element using 2×2×2 Gauss quadrature. Computed ONCE at startup
and reused for every element (valid because all elements are the same size).

**Key function**:
- `compute_ke_base(material, voxel_size) → [[f64; 24]; 24]`

The result is a symmetric matrix. The assembly step uses this matrix scaled
by `ρ_e^p` for each element.

**Performance note**: This is a one-time cost at startup. Even at 120k
elements, computing K_e once is negligible vs the per-iteration assembly.

---

### `filter.rs` — Sensitivity Filter

**Purpose**: Builds the filter weight structure used each iteration to
smooth sensitivities. Built ONCE, reused every iteration.

**Key type**:
```rust
pub struct FilterWeights {
    pub neighbors: Vec<Vec<usize>>,  // [elem_idx] → neighbor element indices
    pub weights:   Vec<Vec<f64>>,    // [elem_idx] → cone kernel weights H_ef
}
```

**Key function**:
- `build_filter(grid, filter_radius_m) → FilterWeights`

Uses a nested loop over elements (no KDTree in Rust — direct distance
computation is fast enough for structured grids). Complexity: O(n_elem × k)
where k = average neighbors per element ≈ (4/3)π(r/h)³.

Filter application (in `sensitivity.rs`, not here):
```
dc_filtered_e = Σ_f H_ef * x_f * dc_f / (Σ_f H_ef * x_f + 1e-16)
```

---

### `assembly.rs` — Stiffness Matrix Assembly

**Purpose**: Assembles the global stiffness matrix K from element matrices
and applies Dirichlet boundary conditions.

**Key functions**:
- `build_csr_pattern(grid, dof_map) → CsrPattern` — builds the sparsity
  pattern (row/col arrays) for K. This pattern is FIXED for all iterations;
  only the values change. Called once.
- `assemble_k(k_vals, x, ke, pattern, void_mask, nondesign, penal)` — fills
  `k_vals` (the CSR value array) by looping over elements and adding
  `ρ_e^p * K_e` contributions. Void and nondesign elements use ρ = ρ_min
  and ρ = 1.0 respectively.
- `apply_dirichlet(k_vals, k_rows, k_cols, fixed_dofs, diag_mean)` — zeroes
  rows and columns for fixed DOFs, sets diagonal to `diag_mean`.

**Why diag_mean instead of 1.0 for Dirichlet diagonal?**
Using the mean diagonal value preserves the scale of K. Using 1.0 when the
rest of the diagonal is O(E * voxel_size) ≈ O(10⁵) would create a badly
scaled matrix with artificial ill-conditioning at the constrained DOFs.

---

### `solver.rs` — Linear Solvers

**Purpose**: Implements the actual linear algebra solvers. This module is
NOT directly called from `simp.rs` — all calls go through `vcycle_dispatch.rs`.

**Key functions**:

`cg_solve_direct(k_rows, k_cols, k_vals, f, u, tol, max_iter) → SolveStats`
- Old entry point. Now only used internally and for tests.
- Dispatches to Cholesky (n_dof < 50k) or Jacobi-CG (n_dof ≥ 50k).

`cg_solve_with_precond(k_rows, k_cols, k_vals, f, u, tol, max_iter, precond) → SolveStats`
- CG loop with arbitrary preconditioner (anything implementing `Preconditioner`).
- Used by VCycle dispatch.

`jacobi_cg_solve(k_rows, k_cols, k_vals, f, u, tol, max_iter) → SolveStats`
- Jacobi-preconditioned CG (diagonal scaling).
- Fallback when VCycle is not dispatched.
- Parallelized with rayon (`into_par_iter()` on the SpMV inner loop).

**faer Cholesky path** (n_dof < 50k):
- Uses the `faer` crate for sparse Cholesky factorization.
- Direct solve — exact, no iteration, fast for small problems.
- Fires on dev grids (e.g., 10×6×4 = 240 elements, 1,029 DOFs).
- Threshold: 50,000 DOFs (matching the VCycle dispatch threshold).

**SolveStats struct**:
```rust
pub struct SolveStats {
    pub iterations: usize,
    pub rel_residual: f64,
    pub converged: bool,
}
```

---

### `multigrid.rs` — VCycle Geometric Multigrid Preconditioner

**Purpose**: Implements the VCycle preconditioner using geometric coarsening
on the structured hex grid.

**STATUS — EXPERIMENTAL, NOT IN DISPATCH.** This module is not called by the
production solver. `VCyclePreconditioner` has no call site in `vcycle_dispatch.rs`
or `simp.rs` — only in this file and its own unit tests. It assumes power-of-2
grid dimensions; production grids (e.g. 70×60×80) violate that and it returns a
wrong operator (C≈1.68e3 at iter 1 vs ~0.1). The primary solver for large CPU
problems is AMGCL (see §3). The components below are individually correct; only
the non-power-of-2 coarsening is broken. Fix-or-retire is roadmap R6.

**Key types**:

```rust
pub struct VCyclePreconditioner {
    levels: Vec<Level>,        // coarse grid hierarchy
    work:   Mutex<VCycleWork>, // scratch buffers (interior mutability for Sync)
}

struct Level {
    n:        usize,           // DOF count at this level
    a_rows:   Vec<usize>,      // CSR: coarse stiffness matrix
    a_cols:   Vec<usize>,
    a_vals:   Vec<f64>,
    p_rows:   Vec<usize>,      // prolongation operator P (fine ← coarse)
    p_cols:   Vec<usize>,
    p_vals:   Vec<f64>,
    r_rows:   Vec<usize>,      // restriction operator R = Pᵀ (coarse ← fine)
    r_cols:   Vec<usize>,
    r_vals:   Vec<f64>,
}

struct VCycleWork {
    residuals: Vec<Vec<f64>>,  // per-level residual buffers
    errors:    Vec<Vec<f64>>,  // per-level error buffers
    temps:     Vec<Vec<f64>>,  // per-level temp buffers
}
```

**Design decisions (locked)**:
- `Mutex<VCycleWork>` for interior mutability — `RefCell` is not `Sync`
  and therefore cannot satisfy the `Preconditioner: Sync` trait bound.
- All work buffers allocated once at construction — zero hot-path allocation.
- Three separate `Vec<Vec<f64>>` fields in VCycleWork to enable NLL
  (non-lexical lifetimes) field-split borrows without `split_at_mut`.
- Iterative V-cycle descent/ascent — recursive implementation conflicts
  with the borrow checker on the work buffer references.

**Smoothing**: Red-black Gauss-Seidel with forward+backward sweep (symmetric).
The symmetric sweep is required for the VCycle to be a symmetric preconditioner,
which is in turn required for VCycle-PCG to be well-defined (PCG requires SPD
preconditioner).

**Coarsening**: Algebraic Galerkin coarsening — A_H = R·A_h·P where P is
the prolongation operator. This produces the correct coarse-grid operator
without requiring knowledge of the coarse-grid geometry.

**Prolongation**: Trilinear interpolation on the structured hex grid.
`p[fine_node] = Σ weights × coarse_node_values`.

**Restriction**: Exact algebraic transpose of prolongation: R = Pᵀ.
This is the Galerkin condition and ensures the coarse-grid residual equation
is consistent with the fine-grid problem.

**Module dependency**: `multigrid.rs` imports from `solver.rs` for the
coarse-grid direct solve. This creates a potential circular dependency
when `solver.rs` needs VCycle — resolved by `vcycle_dispatch.rs`.

---

### `vcycle_dispatch.rs` — Solver Dispatch

**Purpose**: The one module that decides which linear solver runs for each
K·u=f solve; `simp.rs` calls `solve_linear_system` here every iteration. The full
dispatch tree and conditions are in §3. Historically this module also acted as a
shim breaking a `multigrid.rs` ↔ `solver.rs` circular dependency (AD-02); that
role is now vestigial — it no longer imports `multigrid.rs`, and the VCycle path
is not dispatched.

**Module dependency graph** (live call path):
```
main.rs
 ├── simp.rs
 │    └── vcycle_dispatch.rs    ← decides which solver runs
 │         ├── amgcl_solver.rs  ← AMGCL AMG-PCG    [PRIMARY, feature "amgcl"]
 │         ├── solver.rs        ← faer Cholesky, Jacobi-PCG (fallback + corrector)
 │         └── gpu_solver.rs    ← GPU ILU(0)-PCG   [DEPRECATED, feature "gpu" only]
 ├── assembly.rs
 ├── sensitivity.rs
 ├── oc_update.rs
 └── multigrid.rs               ← compiled (mod-declared) but NOT called
                                  (experimental; see §3 and R6)
```

**Dispatch logic** (summary — see §3 for the full tree):
```
n_dof < 50,000                            → faer Cholesky (solver.rs)
n_dof ≥ 50,000, feature "amgcl" (default) → AMGCL AMG-PCG          [PRIMARY]
   └─ per-iteration: rel_residual > 0.1   → Jacobi-PCG for that one iteration
n_dof ≥ 50,000, "gpu" & not "amgcl"       → GPU ILU(0)-PCG         [DEPRECATED]
fallback (amgcl off / gpu off)            → CPU Jacobi-PCG
```

**Key function signature**:
```rust
pub fn solve_linear_system(
    k_rows: &[usize], k_cols: &[usize], k_vals: &[f64],
    f: &[f64], u: &mut [f64],
    tol: f64, max_iter: usize,
    _nx: usize, _ny: usize, _nz: usize,   // reserved for Phase 3 GMG; unused
    gpu_ctx: &mut GpuContext,
) -> CgResult
```

The `_nx, _ny, _nz` parameters (node counts) are underscore-prefixed and
currently unused — reserved for a future V-cycle re-enable (R6). The active AMGCL
path builds its hierarchy algebraically from the matrix and does not need them.
`u` is warm-started on entry (carried from the previous SIMP iteration) and holds
the solution on exit. Return type is `CgResult` (defined in solver.rs).

---

### `preconditioner.rs` — Preconditioner Trait

**Purpose**: Defines the trait that all preconditioners implement.

```rust
pub trait Preconditioner: Sync + Send {
    fn apply(&self, r: &[f64], z: &mut [f64]);
    fn n(&self) -> usize;
}
```

`Sync + Send` bounds are required because VCycleWork uses `Mutex` and the
preconditioner may be used from multiple threads (rayon).

---

### `sensitivity.rs` — Compliance and Sensitivity Computation

**Purpose**: Given the current density field x and displacement u (from
the linear solve), computes compliance and raw sensitivities.

**Key functions**:

`compute_compliance(x, u, ke, dof_map, penal, void_mask, nondesign) → f64`
- C = Σ_e ρ_e^p · u_eᵀ · K_e · u_e
- Returns compliance in Joules (because K_e includes voxel_size scaling)

`compute_sensitivities(x, u, ke, dof_map, fw, penal, void_mask, nondesign) → Vec<f64>`
- dc[e] = -p · ρ_e^(p-1) · u_eᵀ · K_e · u_e   (raw, unfiltered)
- Then applies filter: dc_filtered = (H @ (x * dc)) / (H @ x + 1e-16)
- Returns filtered sensitivities
- void and nondesign elements: dc set to 0.0 (no update signal)

---

### `oc_update.rs` — Optimality Criteria Update

**Purpose**: Given density x and filtered sensitivities dc, performs
bisection to find the volume Lagrange multiplier and computes the
new density field.

**Key function**:
```rust
pub fn oc_update(
    x: &[f64], dc: &[f64], cfg: &SimpConfig,
    void_mask: &[bool], nondesign: &[bool],
) → OcResult

pub struct OcResult {
    pub x_new: Vec<f64>,       // updated density (with damping applied)
    pub vol_frac: f64,         // achieved volume fraction
    pub rho_change: f64,       // max |x_new[e] - x[e]|
}
```

OC damping applied in this function:
```rust
x_new[e] = 0.5 * x_candidate[e] + 0.5 * x[e];
```

Non-design elements: `x_new[e] = 1.0` (forced, not damped)
Void elements: `x_new[e] = 0.0` (forced, excluded from volume)

---

### `simp.rs` — Main SIMP Loop

**Purpose**: Orchestrates one stage of SIMP optimization. Reads from disk
(if checkpoint exists), runs iterations, writes checkpoint periodically,
deletes checkpoint on clean exit.

**Key function**: `run_simp(problem: &Problem, out_dir: &Path) → SolveResult`

Per-iteration sequence:
1. Assemble K from current x
2. Apply Dirichlet BCs
3. Call `vcycle_dispatch::solve_linear_system` → get u
4. `compute_compliance` → compliance
5. `compute_sensitivities` (includes filtering) → dc
6. `oc_update` → x_new, vol_frac, rho_change
7. Update x = x_new
8. Write checkpoint (if iteration % checkpoint_every == 0)
9. Check convergence (spread criterion then density criterion)

**Checkpoint resume priority**:
1. checkpoint.bin + checkpoint_meta.json (interrupted prior run)
2. problem.x_init_file (Stage 2 warm-start from Stage 1)
3. Uniform density at volume_fraction (fresh Stage 1)

Stage 2 checkpoints take precedence over x_init — an interrupted Stage 2
resumes from its own checkpoint, not from x_init.bin.

---

### `gpu_solver.rs` — GPU Linear Solver

**Purpose**: Implements cuSPARSE ILU(0)-preconditioned CG on the RTX 4080.

**Status**: FUNCTIONAL but with one known bug (warm-start issue — see
Section 10, Tech Debt item TD-01).

**Key dependency**: `cudarc 0.19.4` with features `["cuda-12060", "cusparse"]`.

**Key function**: `gpu_cg_solve(k_rows, k_cols, k_vals, f, u, tol, max_iter) → SolveStats`

Zeroes `u` on entry (the warm-start bug). All operations occur on device
(H2D transfer of K and f at the start of each solve, D2H transfer of u
at the end). The K matrix structure is persistent across CG iterations
within one solve but NOT across SIMP iterations (K changes because x changes).

**cudarc API notes** (learned from GPU implementation session):
- Use `cuda-12060` feature for CUDA 12.6 (not `cuda-12040`)
- `csrsv2` triangular solve functions are CUDA 11.x only — unavailable
  with cuda-12060; use `cusparseSpSV` instead
- `cusparseSetStream` requires `std::ptr::null_mut()` not `result::stream::null()`
  due to cross-crate `CUstream_st` type conflict
- Use raw `CUdeviceptr` values from `result::malloc_sync` to avoid lifetime conflicts

**io.rs** — Binary I/O and Checkpointing

**Purpose**: All file I/O for the solver — binary reads/writes of density
arrays, checkpoint serialization, result.json writing.

**Key functions**:
- `write_checkpoint(out_dir, density, meta) → Result<()>`
- `read_checkpoint(out_dir, n_elem) → Option<(Vec<f64>, CheckpointMeta)>`
- `delete_checkpoint(out_dir)`

`CheckpointMeta`:
```rust
pub struct CheckpointMeta {
    pub iter_completed: usize,
    pub compliance_history: Vec<f64>,
    pub volume_history: Vec<f64>,
    pub n_elem: usize,
}
```

`n_elem` is stored in the checkpoint so stale checkpoints (from different grid
sizes) can be detected and rejected. If checkpoint n_elem ≠ current n_elem,
the checkpoint is silently ignored.

density.bin format: raw f32 (NOT f64) values, C order, length n_elem.
Reading in Python: `np.fromfile("density.bin", dtype=np.float32).reshape(nz, ny, nx)`

SolveResult.final_density is decorated with `#[serde(skip)]` — it is NOT
written to result.json. The density is always read from density.bin.
Reason: 120k elements × 8 bytes = ~1MB of f64 text in JSON would be enormous
and is never consumed from JSON (only from binary).


---

## 3. Solver Dispatch Architecture

Full decision tree for `vcycle_dispatch::solve_linear_system`:

```
                       ┌───────────────────────┐
                       │  solve_linear_system  │
                       └───────────┬───────────┘
                                   │
                  n_dof < 50,000  (CHOLESKY_THRESHOLD)?
                      ┌────────────┴────────────┐
                     YES                         NO
                      │                          │
            faer sparse Cholesky       feature "amgcl"  (DEFAULT)?
            (solver.rs)               ┌────────────┴────────────┐
            exact, direct            YES                        NO
                                      │                          │
                           AMGCL AMG-PCG          feature "gpu" + use_gpu?
                           ══ PRIMARY ══          ┌────────────┴────────────┐
                           amgcl_solver.rs +     YES                        NO
                           amgcl_wrapper.cpp      │                          │
                           smoothed aggregation,  GPU ILU(0)-PCG      CPU Jacobi-PCG
                           ILU(0) smoother,       (gpu_solver.rs)     (solver.rs)
                           block_size=3, OpenMP   ~3-4 s/iter         ultimate fallback
                           ~16 s/iter             [LEGACY — broken
                           residual gate @ 0.1:    for high contrast]
                           dirty → Jacobi 1 iter  correct @ iter 1,
                                                  diverges @ iter 6+;
                                                  dead in default build
```
**Dispatch is feature-gated, not just size-gated.** The default build
(`cargo build --release`) compiles the `amgcl` feature, so every solve at or
above the 50k-DOF threshold routes through AMGCL. The GPU ILU(0) leaf is
reachable *only* under `--no-default-features --features gpu`; no default build
ever executes it.

**CPU Jacobi-PCG has two roles.** It is the bottom fallback (AMGCL not compiled,
GPU off) *and* the per-iteration corrector: when an AMGCL solve's relative
residual exceeds `AMGCL_FALLBACK_THRESHOLD` (= 0.1), the dispatcher substitutes
a single Jacobi-PCG solve for that iteration. Rationale (from the code): a ~10%
displacement error corrupts the OC sensitivities and can pull the design into a
wrong local minimum, whereas early AMGCL iterations legitimately show residuals
of 1–10 before ILU(0) has seen a few density fields.

**VCycle geometric multigrid is NOT in this dispatch.** `VCyclePreconditioner`
(multigrid.rs) has no call site in `solve_linear_system` or `simp.rs` — it is
referenced only inside multigrid.rs and its own unit tests. It assumes
power-of-2 grid dimensions; production grids (e.g. 70×60×80) violate this and it
returns a wrong operator (C≈1.68e3 at iter 1 vs the correct ~0.1). It compiles
and its small/power-of-2 unit tests pass, which is why CI reads green despite the
operator being unusable on real meshes. Treat it as experimental, outside the
solver. Fix-or-retire is roadmap item R6.

**50k DOF threshold rationale**: At 50k DOFs, faer Cholesky factorization
takes ~500ms and the resulting direct solve is faster than 200 CG iterations.
Above 50k, Cholesky fill-in becomes too large and CG is preferred. This
threshold was determined empirically during solver development.

**Solver context** (`GpuContext` struct in vcycle_dispatch.rs — the name is a
misnomer kept for back-compat with simp.rs; it is the persistent *solver*
context, not a GPU-only one). Constructed once at the start of `run_simp()` via
`GpuContext::new(use_gpu)` and threaded through `solve_linear_system` on every
iteration, so the context persists for the lifetime of a stage rather than being
recreated per solve. It holds:
- `amgcl_ctx` — the AMGCL AMG-PCG context, the **primary** solver; created
  lazily on the first solve. (`feature = "amgcl"`, default.)
- `inner` — lazy legacy GPU ILU(0) state, `Option<GpuK>`; bypassed entirely when
  the `amgcl` feature is active. (`feature = "gpu"`.)
- `use_gpu` — a forward-looking *intent* flag. In the current default build it is
  **ignored for routing**: AMGCL OpenMP runs regardless of its value. It is
  reserved for Phase B (selecting AMGCL's CUDA vs builtin backend) and does NOT
  switch between CPU and GPU today.

`gpu_ctx.backend_label()` — returns a human-readable string, printed once at SIMP
startup, identifying the active backend: an AMGCL label in the default build, a
`[DEPRECATED]` legacy-GPU label only under `--no-default-features --features gpu`,
or a CPU (Jacobi-PCG / Cholesky) label as the fallback.

---

## 4. Data Flow — End to End

### From User Request to STL

```
User defines geometry
         │
         ▼
scad/{part}.scad + scad/{part}_params.json
         │
         ▼
NB01: openscad_runner.py
  → OpenSCAD renders STL
  → outputs/meshes/{part}.stl
  → writes {part}_stage01.json
         │
         ▼
NB02: gmsh_pipeline.py
  → STL → tetrahedral mesh → XDMF
  → outputs/meshes/{part}.xdmf
  → outputs/meshes/{part}_boundaries.xdmf
  → writes {part}_stage02.json
  [On Rust path: NB02 writes stub stage02 — mesh files not needed]
         │
         ▼
NB03: FEniCSx FEA (validation / displacement field)
  → outputs/meshes/{part}_displacement.xdmf
  → outputs/meshes/{part}_stress.xdmf
  → writes {part}_stage03.json
         │
         ▼
NB04: SIMP Optimization (primary — Rust voxel path)
  Cell 2: voxelize_domain() → void_mask, nondesign (nz×ny×nx bool arrays)
  Cell 2: build_load_case() → fixed_dofs.bin (shared) + per-case
          load_dofs_{i}.bin / load_vals_{i}.bin (one pair per load case)
  Cell 3: write nondesign.bin, void.bin
  Cell 3: write problem_s1.json, run bin/simp_solver → density.bin, result.json
  Cell 3: write problem_s2.json (warm-start), run bin/simp_solver again
  Cell 3: merge histories, save {part}_density.npy
  Cell 4: convergence dashboard PNG
  Cell 5: density render (if RENDER_PLOTS=True)
  Cell 5b: gray fraction + convergence classification → _diag dict
  Cell 5c: orthogonal slice plots → {part}_density_slices.png
  Cell 6: writes {part}_stage04.json + {part}_simp_diagnostic.json
         │
         ▼
NB05: STL Export (marching cubes)
  → reads {part}_density.npy (shape: nz×ny×nx)
  → threshold at 0.5 (configurable)
  → scikit-image marching_cubes
  → island removal (keep largest connected component)
  → trimesh smoothing + decimation
  → outputs/stl/{part}_optimized.stl
  → writes {part}_stage05.json
         │
         ▼
NB06: Part Validation
  → STL watertightness check (trimesh)
  → Compliance-based safety factor (σ_rms method)
  → Overhang angle analysis (printability)
  → Feature size check (min strut width)
  → Dimensional accuracy check
  → writes {part}_validation.json
```

### Binary File Format Summary

| File                | Format         | Shape           | dtype   | Written by  | Read by     |
|---------------------|----------------|-----------------|---------|-------------|-------------|
| fixed_dofs.bin      | raw flat array | (n_fixed,)      | uint32  | NB04 Cell 3 | Rust solver |
| load_dofs_{i}.bin   | raw flat array | (n_load_i,)     | uint32  | NB04 Cell 3 | Rust solver |
| load_vals_{i}.bin   | raw flat array | (n_load_i,)     | float64 | NB04 Cell 3 | Rust solver |
| nondesign.bin       | raw flat array | (n_elem,)       | uint8   | NB04 Cell 3 | Rust solver |
| void.bin            | raw flat array | (n_elem,)       | uint8   | NB04 Cell 3 | Rust solver |
| x_init.bin          | raw flat array | (n_elem,)       | float32 | NB04 Cell 3 | Rust solver |
| density.bin         | raw flat array | (n_elem,)       | float32 | Rust solver | NB04 Cell 3 |
| checkpoint.bin      | raw flat array | (n_elem,)       | float32 | Rust solver | Rust solver |

Note: n_elem = nx × ny × nz (X-fastest layout). When reading density.bin
in Python: reshape to (nz, ny, nx) before any slicing.

Note: DOF index arrays are uint32 little-endian (io.rs reads them with
read_u32_le; the Python writer emits np.uint32). load_vals are float64.

### problem_sN.json — "loading" Block (R1 multi-load schema)

The solver reads its load problem from the `loading` block of problem_sN.json.
Supports are shared; each load case names its own DOF/value files:

```json
"loading": {
  "fixed_dofs_file": "fixed_dofs.bin",
  "load_cases": [
    { "name": "thrust_down", "weight": 1.0,
      "load_dofs_file": "load_dofs_0.bin", "load_vals_file": "load_vals_0.bin" },
    { "name": "side_y", "weight": 1.0,
      "load_dofs_file": "load_dofs_1.bin", "load_vals_file": "load_vals_1.bin" }
  ]
}
```

- `name` (default "lc{i}") and `weight` (default 1.0) are optional per case.
- `load_cases` must be non-empty (io.rs rejects an empty list).
- A per-case `fixed_dofs_file` key is RESERVED — io.rs errors if present
  (per-case supports are a future phase; today all cases share fixed_dofs_file).
- This REPLACED the pre-R1 `load_case` block ({fixed_dofs_file, load_dofs_file,
  load_vals_file}). problem_sN.json is ephemeral (gitignored), so this was a
  clean break, not a back-compat addition.


---

## 5. Python Pipeline Module Map

### `scripts/voxelize.py` — Domain Masking and Load Case Builder

**Purpose**: Converts geometry parameters into voxel-domain boolean masks
and load/boundary condition arrays for the Rust solver.

**Key functions**:

`voxelize_domain(geometry_params, grid_config, nondesign_regions, void_regions,
                 bolt_seats, attachment_regions) → (nondesign, void_mask)`
- Returns (nz, ny, nx) bool arrays (X-fastest within each Z slice)
- Applies all declared region types to build the masks

`build_load_case(geometry_params, load_hints, grid_config,
                 load_case_config=None, attachment_regions=None) → dict`
- Returns {'fixed_dofs': np.array (shared),
           'load_cases': [{'name','weight','load_dofs','load_vals'}, ...],
           'load_dofs', 'load_vals'}   ← top-level 'load_dofs'/'load_vals' are a
  back-compat alias for load_cases[0] (the primary case)
- Builds one load_cases entry per scenario in load_case_config.loads; a legacy
  single 'load' (or the non-config path) yields one case named "primary"
- Dispatches to appropriate selector based on load_case_config

`_fixed_dofs_from_config(geometry_params, grid_config, load_case_config,
                         attachment_regions) → np.array`
- Handles the declarative load case config (center_disk and other selectors)
- Includes `_face_center_m(face, geometry_params, grid_config)` helper

`_load_dofs_from_config(...)` — load DOF equivalent

`center_disk` selector: constrains all nodes within a disk of specified
radius centered on a face. Used by conrod (x_min/x_max) and motor_mount.
The disk radius must produce ≥ 19k fixed DOFs for stable conditioning.

---

### `src/geometry/param_schema.py` — PipelineParams Hierarchy

**Purpose**: All parameter dataclasses and their validation logic.

**Key hierarchy**:
```
PipelineParams
├── GeometryParams            ← physical dimensions in mm
├── LoadHints                 ← primary_face, load_magnitude_n
├── LoadCaseConfig            ← declarative BC spec (fixed + load dicts)
├── List[VoidRegion]          ← regions to exclude from optimization
├── List[NondesignRegion]     ← regions forced to solid
├── List[BoltSeat]            ← bolt attachment geometry
└── List[AttachmentRegion]    ← composite attachment spec (slab + bolts)
```

`PipelineParams.from_json(path)` — primary entry point (reads JSON, validates)
`PipelineParams.from_dict(d)` — used by tests

Papermill parameter injection: scalar values only (int, float, str, bool).
Dict/nested params cannot be injected via `-p`. Use `params.json` simp
block for per-part SIMP overrides.

---

### `src/geometry/region_factory.py` — Geometry Region Resolution

**Purpose**: Resolves high-level geometry descriptions into concrete void
and nondesign region lists that `voxelize_domain` can consume.

`resolve_geometry_regions(pipeline_params) → (void_regions, nondesign_regions)`
- Combines declared regions with auto-generated regions from attachment_regions
- Returns augmented lists (declared + auto)

Auto-generated regions include:
- Bolt void cylinders from `AttachmentRegion.bolt_voids`
- Face slab nondesign regions from `AttachmentRegion.slab_*` geometry

---

### `src/optimization/simp.py` — FEniCSx SIMP (Legacy/Fallback)

**Purpose**: Original Python/FEniCSx SIMP implementation. Active only when
`USE_RUST_SOLVER=False` in NB04. Kept for comparison and fallback.

Key differences from Rust path:
- Operates on tetrahedral mesh (not voxel grid)
- Uses dolfinx DG0 density function space
- LinearProblem built ONCE outside the loop (no JIT recompilation per iter)
- Returns `SIMPResult` dataclass (different schema from Rust `SolveResult`)

The FEniCSx path produces compliance in different units than the Rust voxel
path because the cell volumes differ (tet vs hex). Do NOT compare compliance
values between paths on the same part.


---

## 6. Notebook Pipeline — Cell Summary

### NB04 (04_simp_optimization.ipynb) — CURRENT STATE AFTER DIAGNOSTIC LAYER

| Cell | Index | Purpose                                        | Key outputs                              |
|------|-------|------------------------------------------------|------------------------------------------|
| 0    | 2     | Parameters (Papermill injectable)              | PART_NAME_OVERRIDE, VOXEL_SIZE_MM, etc.  |
| 1    | 4     | Load stage03 handoff, apply params.json overrides | part_name, geometry_params, load_hints |
| 2    | 6     | Build solver config (grid_config, simp_config) | nx, ny, nz, simp_config dict            |
| 2b   | 7     | Corner-disk diagnostic (legacy parts only)     | Fixed node counts per corner             |
| 3    | 9     | Run optimization (two-stage SIMP)              | result dict, void_mask, nondesign, lc    |
| diag | 10    | Namespace diagnostic print                     | (debug only)                             |
| 4    | 12    | 6-panel convergence dashboard                  | {part}_convergence.png                   |
| 5    | 14    | Density render (gated on RENDER_PLOTS)         | {part}_density_slices.png (if enabled)   |
| 5b   | 16    | Density quality diagnostics ← NEW              | _diag dict (gray fraction, conv shape)   |
| 5c   | 18    | Orthogonal slice plots ← NEW                   | {part}_density_slices.png (always)       |
| 6    | 20    | Stage04 handoff ← MODIFIED                     | stage04.json (+ histories + _diag)       |
|      |       |                                                | {part}_simp_diagnostic.json              |

Cell 3 `run_stage()` function: calls the Rust binary as a subprocess,
streams stdout line by line (real-time progress), reads result.json after
completion. The subprocess uses a custom `_env` with LD_LIBRARY_PATH set
to include CUDA libraries.

### pipeline_full.ipynb — Orchestrator

Calls NB01–NB06 via Papermill with per-notebook parameter injection.
SIMP_OVERRIDES dict injection is broken (Papermill `-p` cannot inject dicts).
Workaround: use per-part params.json simp block instead.


---

## 7. Inter-Stage Handoff Schema

### stage04.json — CURRENT SCHEMA (after diagnostic layer addition)

```json
{
  "stage":              "04_simp",
  "solver":             "rust_voxel",
  "part_name":          "motor_mount",
  "density_path":       "outputs/meshes/motor_mount_density.npy",
  "xdmf_path":          "outputs/meshes/opt_domain.xdmf",
  "n_iterations":       347,
  "converged":          true,
  "final_compliance":   0.04093,
  "final_volume_frac":  0.38000,
  "duration_s":         6218.5,
  "config": {
    "volume_fraction":  0.38,
    "penal":            3.0,
    "filter_radius":    0.006,
    "max_iterations":   200,
    "convergence_tol":  0.0005,
    "move_limit":       0.3,
    "damping":          0.5,
    "checkpoint_every": 10,
    "use_gpu":          true,
    "max_cg_iter":      200,
    "filter_radius_mm": 6.0
  },
  "material": { "youngs_modulus_pa": 210e9, "poissons_ratio": 0.3, "name": "steel" },
  "load_hints": { "primary_face": "right", "load_magnitude_n": 5000.0 },
  "grid": { "nx": 70, "ny": 60, "nz": 80, "voxel_size": 0.001 },
  "compliance_history": [...],           ← NEW: full array (was missing before)
  "volume_history": [...],               ← NEW: full array (was missing before)
  "diagnostic": {                        ← NEW: quality summary block
    "gray_fraction_all":    0.031,
    "gray_fraction_design": 0.028,
    "gray_status":          "PASS",
    "solid_fraction":       0.381,
    "void_fraction":        0.619,
    "convergence_shape":    "healthy_monotone",
    "convergence_note":     "...",
    "final_spread":         8.3e-6,
    "stage1_iters":         200,
    "stage2_iters":         147,
    "slice_plots_png":      "outputs/reports/motor_mount_density_slices.png"
  }
}
```

### {part}_simp_diagnostic.json — NEW (written by NB04 Cell 6)

Standalone report in `outputs/reports/`. Contains everything in the diagnostic
block above PLUS the full compliance_history and volume_history arrays.
This is the file to share in a review session.

### validation.json — NB06 Output Schema

```json
{
  "part_name":  "motor_mount",
  "timestamp":  "...",
  "validation_mode": "fast",
  "pipeline_state": { "converged": true, "volume_fraction": 0.45, "final_compliance": 0.039 },
  "stl_quality": { "watertight": true, "open_edges": 0, "euler_number": -26, ... },
  "stress_analysis": { "compliance_based": { "SF": 10.94, "status": "PASS" }, ... },
  "printability": { "overhanging_area_pct": 8.25, "status": "INFO" },
  "dimensional_accuracy": { ... },
  "overall_status": "PASS"
}
```


---

## 8. Key Architectural Decisions

### AD-01: Two-Stage Penalization at Notebook Level, Not Solver Level

The Rust solver always runs a single stage. The notebook calls `run_stage()`
twice with different `penal` values (2.0 → 3.0). The x_init.bin handoff
between stages is the warm-start mechanism.

**Why**: Keeping the solver stateless (one stage, one penal value) makes
it easier to test and reason about. The notebook orchestrates strategy;
the solver implements mechanics.

**Consequence**: If a session modifies the two-stage logic, it must be done
in NB04 Cell 3, not in simp.rs.

### AD-02: vcycle_dispatch.rs Shim Pattern for Circular Dependencies

When module A imports B and B needs A, introduce a shim C that imports both.
Neither A nor B import each other; callers import C.

**Generalisation**: This pattern should be used for any future cross-module
dependency in the Rust solver. Do not restructure core modules to break
cycles; add a shim instead.

**Current role**: With VCycle no longer dispatched, the shim no longer imports
`multigrid.rs`; `vcycle_dispatch.rs` is now primarily the AMGCL / Cholesky /
Jacobi dispatcher (§3). The shim *pattern* and the guidance above still stand for
future cycles — the specific multigrid↔solver cycle it originally resolved is just
no longer live.

### AD-03: mtime-Based Handoff Detection (Not Alphabetical)

All `glob("*_stageXX.json")` auto-detection uses:
```python
sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
```
NOT `sorted(candidates)[-1]` (alphabetical).

**Why**: Alphabetical fails when two runs produce files with the same date
prefix. mtime always picks the most recently written file.

**This was a confirmed bug (fixed in Tier 5 Session 1)**. Do not revert to
alphabetical sorting.

### AD-04: Part-Agnosticism via Declarative params.json

No notebook cell contains hardcoded part names, dimensions, or paths.
Everything flows from `params.json`. The part name comes from stage handoff
files or from `PART_NAME_OVERRIDE` (Papermill injectable).

### AD-05: Papermill Metadata Tags Must Be Cell Metadata, Not Comments

For Papermill to inject parameters into a cell, the cell must have
`{"tags": ["parameters"]}` in its cell metadata JSON, NOT just a
`# parameters` comment in the source.

**This was a confirmed silent failure (fixed in Tier 5 Session 1)**. The
old comment-based approach caused every Papermill run to use the default
parameter values silently. After the fix, parameter injection works correctly.

### AD-06: plt.show() Removed for Headless Execution

All `plt.show()` calls were removed from notebook cells during the Tier 5
audit. In headless Papermill execution, `plt.show()` blocks indefinitely.
Use `plt.savefig()` followed by `plt.close()` instead.

### AD-07: Rust Density Array Orientation

The flat density array from the solver maps as:
```
flat_index = ix + iy*nx + iz*nx*ny     (X-fastest, Z-slowest)
```
Python reshape: `density.reshape(nz, ny, nx)`

Slice conventions (see MATERIALS_SPEC.md Section 3 for full derivation):
```python
density_3d[iz_mid, :, :]   # XY plane (top-down)
density_3d[:, iy_mid, :]   # XZ plane (front view)
density_3d[:, :, ix_mid]   # YZ plane (side view)
```

Getting this wrong produces a silently incorrect array. Always verify with
the domain mask (Cell 3 output) which shows the void_mask in the same
coordinate system.

### AD-08: Non-Design Elements Excluded from Volume Budget

The OC bisection volume constraint operates on DESIGN elements only:
```python
vol = (x_new[design_mask] * cell_volumes[design_mask]).sum()
      / cell_volumes[design_mask].sum()
```
Non-design elements are forced to ρ=1.0 and do not consume volume budget.
The resulting total solid volume is therefore > VF × total_volume when
non-design regions are present.

### AD-09: Marching Cubes Axis Reorder and Z-Flip

The STL export in NB05 applies two transforms after marching cubes:
1. Axis reorder: maps (nz,ny,nx) density array to correct (x,y,z) STL coordinates
2. Z-flip: `vertices[:,2] *= -1` — required so the load face exports facing up

**Both were confirmed bugs (fixed in Tier 5 Session 1)**. The pre-fix STL
was rotated and/or mirrored incorrectly. The current NB05 is correct.


---

## 9. Test Suite

### Rust Unit Tests (cargo test)

**Count**: 141 passing  (+1 over the R0 baseline: two_identical_halfweight_loads_match_single,
which proves k=1 weighted-sum == single-load)
**Run**: `cargo test --release` from `solver/` directory
**Time**: ~0.17s in release mode

Key test modules and what they cover:

| Module        | Tests                                                           |
|---------------|-----------------------------------------------------------------|
| types.rs      | Grid index round-trips, DOF layout, SimpConfig validation       |
| ke_base.rs    | K_e symmetry, positive definiteness, trace matches theory      |
| filter.rs     | Weight sums, corner vs interior neighbor counts, zero outside r |
| assembly.rs   | K symmetry, Dirichlet BC application, reaction force balance   |
| connectivity.rs| node_idx/elem_idx corners, stride verification                 |
| sensitivity.rs | dc ≤ 0 always, compliance = U^T F, finite-diff verification    |
| oc_update.rs  | Volume constraint, void/nondesign forcing, bisection convergence|
| simp.rs       | Compliance decreasing, volume fraction tracking, warm-start     |
| io.rs         | Checkpoint round-trip, stale n_elem rejection                  |
| multigrid.rs  | VCycle reduces residual, symmetry, faster than Jacobi          |

> **Note on multigrid.rs tests**: these pass only because they run on small,
> power-of-2 grids. The VCycle is NOT dispatched in production and produces a
> wrong operator on real (non-power-of-2) grids — see §3 and §2 `multigrid.rs`.
> A green badge here is the false-confidence case flagged in R0; an integration
> test on a non-power-of-2 grid above the 50k threshold would catch it (R2/R6).

**Running specific test modules**:
```bash
cargo test --release sensitivity      # all tests in sensitivity.rs
cargo test --release vcycle           # all tests containing "vcycle"
cargo test --release -- --nocapture   # show println! output
```

### Python Integration Tests (pytest)

**Location**: `tests/` directory
**Run from container**:
```bash
docker exec -it fenics-pipeline bash -c "cd /workspace && python -m pytest tests/ -v"
```

| File                          | Covers                                          |
|-------------------------------|-------------------------------------------------|
| test_solver_interface.py      | Rust binary invocation, result.json schema      |
| test_voxelize_selectors.py    | center_disk, face selectors, DOF counts         |
| test_param_schema.py          | PipelineParams.from_json, validation            |
| test_region_factory.py        | resolve_geometry_regions output correctness     |
| test_bolt_seat.py             | BoltSeat geometry resolution                    |
| test_slabs.py                 | Slab nondesign region voxelization              |
| test_fea_smoke.py             | FEniCSx import and basic solve (slow)           |
| test_mesh_quality.py          | Aspect ratio, Jacobian metrics                  |
| test_skip_meshing_wiring.py   | Stub stage02 path for Rust solver               |


---

## 10. Known Tech Debt

### TD-01: GPU Warm-Start — RESOLVED

**File**: `solver/src/gpu_solver.rs`
**Status**: Implemented. `cg_solve_persistent()` uploads the caller's `u` as the
CG initial guess (correct for both cold- and warm-start, no special-casing), so
the GPU path no longer zeroes `u` on entry.
**Note**: Moot in practice — the GPU ILU(0)-PCG path is deprecated (see §3; it
diverges under SIMP contrast) and is dead code in any default `amgcl` build. The
warm-start plumbing carries over to the Phase B AMGCL-CUDA path (roadmap R4).

### TD-02: GPU Persistent-K / Values-Only Update — RESOLVED

**File**: `solver/src/gpu_solver.rs`
**Status**: Implemented. The CSR structure persists on the device (one
`cusparseSpMatDescr_t` created once, row/col arrays uploaded once); `refactor()`
does a values-only H2D copy of the K and ILU values each iteration and reuses the
SpSV analysis. The CG loop runs device-side via cuBLAS, with only scalars (α, β,
residual) returning to the host. The "K re-uploaded every iteration" concern is
closed.
**Note**: As with TD-01, this is on the deprecated GPU path and does not run in a
default build. It is not the production bottleneck — the live performance lever is
TD-07 (AMGCL rebuild) below. GPU story: §3 and roadmap R4.

### TD-03: SIMP_OVERRIDES Dict Injection via Papermill (LOW PRIORITY)

**File**: `notebooks/pipeline_full.ipynb`
**Issue**: Papermill `-p` cannot inject dicts. Any attempt to pass SIMP
parameter overrides as a dict from pipeline_full fails silently.
**Current workaround**: Per-part params.json simp blocks (fully functional).
**Fix would require**: Serializing override dicts as JSON strings and
deserializing in the receiving notebook. Complex, low value given the
params.json workaround works well.

### TD-04: STL Island Removal Tuning

**File**: `notebooks/05_stl_export.ipynb`
**Issue**: Island removal (keep largest connected component) can discard
legitimately connected sub-structures if the marching cubes threshold is
too aggressive. The current threshold (0.5) is confirmed working but
borderline cases need tuning.
**Impact**: Very low on current parts (motor_mount, conrod). Higher risk
on parts with thin connecting struts.

### TD-05: FEA Remesh Safety Factor (SKIPPED in NB06)

**File**: `notebooks/06_part_validation.ipynb`
**Issue**: The `fea_remesh` safety factor method (STL → tet mesh → FEniCSx
FEA → von Mises) is marked SKIPPED in all validation.json outputs.
**Why skipped**: Requires re-meshing the optimized STL (expensive, fragile
for topology-optimized geometry with thin struts).
**Current alternative**: Compliance-based SF with K_t=3.0 (conservative).
**Impact**: For production use, the remesh path would give more accurate
peak stress. For the current use case (FDM prototyping), compliance-based
SF with generous K_t is sufficient.

### TD-06: peak_memory_mb Always 0.0

**File**: `solver/src/types.rs`, `simp.rs`
**Issue**: SolveResult.peak_memory_mb is always 0.0. Memory tracking was
planned but not implemented.
**Impact**: Cannot monitor VRAM usage during GPU runs from the pipeline.
**Fix**: Add `/proc/self/status` VmRSS read or CUDA memory query.
**Priority**: Very low.

### TD-07: AMGCL Rebuilds the Full Hierarchy Every Iteration (HIGH — roadmap R2)

**File**: `solver/src/amgcl_solver.rs`, `solver/vendor/amgcl_wrapper.cpp`
**Context**: AMGCL is the PRIMARY, default solver for n_dof ≥ 50k (smoothed
aggregation + ILU(0), block_size=3, OpenMP) — not experimental, and it is the
successor to the VCycle path, not an alternative to it. See §3.
**Issue**: On every SIMP iteration after the first, `amgcl_update()` calls
`rebuild()`, which runs `h->solver.reset(new SolverType(A, prm))` — a full
reconstruction: re-aggregation, the full Galerkin coarse-operator hierarchy, and
ILU(0) factorization at every level. The dispatch side calls this a values-only
"refactor," but the AMGCL *setup* phase runs in full each time. The matrix
structure is fixed across SIMP iterations (only values change), so the
aggregation is reusable and most of this setup is redundant.
**Impact**: For AMG on elasticity, setup is comparable to or costlier than a
single solve; at ~1M DOF over 200–350 iterations this is plausibly a large
fraction of the ~16 s/iter. Highest-value performance item on the active path.
**Fix direction**: Investigate AMGCL preconditioner reuse — rebuild operators on
the fixed aggregation rather than re-aggregating, and/or rebuild every N
iterations (density changes slowly late in the run). Instrument setup-vs-solve
timing first; the wrapper currently records neither.
**Effort**: ~3–6 days (roadmap R2).

### TD-08: ILU(0) Zero Pivot on Ill-Conditioned / Lateral Load Cases (HIGH — R2/R4)

**File**: `solver/vendor/amgcl_wrapper.cpp` (relaxation config), `solver/src/amgcl_solver.rs`
**Issue**: AMGCL's ILU(0) smoother hits a zero pivot (`rebuild: Zero pivot in
ILU`) on the near-singular, high-contrast K produced when a load is poorly
constrained by the fixity. The Jacobi-PCG fallback fires but also fails to
converge those systems (res ~1e-2), corrupting the result.
**Confirmed reproducer**: motor_mount + a `side_y` ([0,1,0]) load case at 5000 N
AND 2000 N → zero pivot at iter ~13–16 (Stage 1, p=2). An axial −X case is
well-conditioned and runs clean — it is the load DIRECTION vs fixity, not the
magnitude. See MATERIALS_SPEC FM-09.
**Impact**: lateral / ill-conditioned load cases are currently unusable; only
well-conditioned (axial, bending) loads solve. Multi-load is otherwise correct.
**Fix**: small diagonal shift / regularization on the ILU(0) factorization
(AMGCL relaxation params). **Effort**: ~1–2 days.

### TD-09: NB06 Validation Is Single-Load Only (MEDIUM)

**File**: `notebooks/06_part_validation.ipynb`
**Issue**: The safety-factor calc reads a single `force_vector_N` and certifies
SF against ONE load case. On a multi-load part it validates only the primary
case, not the worst case across `load_cases`.
**Impact**: SF on multi-load parts understates risk — the motor_mount dual-load
SF (10.14) is vs the −Z thrust only, not the −X case.
**Fix**: iterate `loading.load_cases`, compute per-case SF, report worst-case +
a per-case table. **Effort**: ~0.5–1 day.

### TD-10: NB05 / NB06 Part-Detection Mismatch (LOW)

**Files**: `notebooks/05_stl_export.ipynb`, `notebooks/06_part_validation.ipynb`
**Issue**: NB05 auto-detects the part from the most recent `*_stage04.json`;
NB06 requires `PART_NAME_OVERRIDE` or `outputs/pipeline_state.json` and errors
otherwise. Manual Papermill runs of NB06 fail without `-p PART_NAME_OVERRIDE`.
**Fix**: give NB06 the same mtime-based stage-handoff auto-detection as NB05.
**Effort**: ~1 hour.


---

## 11. Git History Summary

Key commits on main (most recent first):

| Commit  | Message                                              | Session            |
|---------|------------------------------------------------------|--------------------|
| (HEAD)  | R1: SOLVER_STATE §2/§7 to multi-load shape; dual-load params | R0/R1 session  |
| dfab2ac | No-slab codification: ring-contract tests, §11 note, ILU(0) label | R0/R1 session |
| 3897714 | Fix test_solver_interface.py to new 'loading' JSON schema | R0/R1 session      |
| 5e93ef7 | R1 Python: loads schema, per-case .bin writer, NB04 block | R0/R1 session      |
| c799d84 | R0 docs reconciled to code + R1 Rust weighted-sum multi-load | R0/R1 session    |
| 5bfec6c | Tier 5 Session 1: part-agnosticism audit complete         | Tier 5 Session 1   |
| 5bfec6c | Tier 4 Phase 6: V-cycle dispatch wired into simp     | Tier 4 Session 5   |
| 6854929 | Tier 4 Phase 5: V-cycle GMG preconditioner           | Tier 4 Session 5   |
| c89f23e | Tier 4 Phase 4: Galerkin coarse operator hierarchy   | Tier 4 Session 4   |
| 79bb5ea | Tier 4 Phase 3: Prolongation and Restriction ops     | Tier 4 Session 3   |
| (...)   | Tier 4 Phase 2: Red-black Gauss-Seidel smoother      | Tier 4 Session 2   |
| (...)   | Tier 4 Phase 1: Multigrid level hierarchy            | Tier 4 Session 1   |
| (...)   | GPU ILU(0)-CG: cuSPARSE triangular solve             | GPU session        |
| (...)   | GPU SpMV via cuSPARSE (Phase 1)                      | GPU session        |
| (...)   | Parallel CG matvec via rayon (3.8× speedup)          | Pre-Tier 4         |
| (...)   | Cholesky dispatch for small problems (faer)          | Pre-Tier 4         |
| (...)   | Rust voxel solver: all modules integrated            | Tier 3             |

Tier naming:
- Tier 1–3: Rust solver module construction (types → assembly → solver → SIMP)
- Tier 4: Preconditioner development (GMG, GPU, dispatch)
- Tier 5: Pipeline part-agnosticism and diagnostic layer


---

## 12. Development Environment

### Container

```
Container name:   fenics-pipeline
Base image:       dolfinx/dolfinx (FEniCSx official)
Compose version:  v1 (do NOT upgrade to v2 without testing — syntax differs)
Jupyter port:     8888
Kernel name:      fenics-pipeline
Working dir:      /workspace (= repo root mounted at /workspace)
```

**Starting the container**:
```bash
cd /mnt/c/Users/dfbar/Documents/Repos/fenics-pipeline
docker-compose up -d
```

**Exec into container**:
```bash
docker exec -it fenics-pipeline bash
```

### Paths

| Location                  | Path in WSL2                                              |
|---------------------------|-----------------------------------------------------------|
| Repo root                 | /mnt/c/Users/dfbar/Documents/Repos/fenics-pipeline        |
| Repo root (in container)  | /workspace                                                |
| Rust solver source        | /mnt/c/.../fenics-pipeline/solver/src/                    |
| Solver binary (deployed)  | /workspace/bin/simp_solver                                |
| Solver binary (compiled)  | /workspace/solver/target/release/simp_solver              |
| Problem I/O dir           | /workspace/outputs/problem/                               |
| Stage handoff files       | /workspace/outputs/meshes/                                |
| Reports/PNGs              | /workspace/outputs/reports/                               |

### Build Commands

```bash
# Build Rust solver (from repo root, in container or WSL2):
bash scripts/build_solver.sh

# Or via Makefile:
make build-solver

# Run Rust tests:
cd solver && cargo test --release

# Run Python tests (in container):
python -m pytest tests/ -v

# Manual solver invocation:
bin/simp_solver outputs/problem/problem_s1.json
```

### GPU Environment

```
GPU:          RTX 4080 (Ada Lovelace, 16GB VRAM)
CUDA version: 12.6
CUDA path:    /usr/local/cuda-12.6 (in container)
cudarc:       0.19.4 with features ["cuda-12060", "cusparse"]
```

LD_LIBRARY_PATH required for GPU binary (set in NB04 Cell 3's `_env`):
```
/usr/lib/wsl/lib:/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib
```

---

## 13. Session Startup Protocol

### Standard Session Start

```bash
cd /mnt/c/Users/dfbar/Documents/Repos/fenics-pipeline
bash scripts/session_start.sh
```

session_start.sh performs, in order:
1. Confirms correct repo directory
2. Shows git branch / HEAD / uncommitted changes
3. Checks / starts Docker container
4. Verifies GPU passthrough (`nvidia-smi` in container)
5. Checks if Rust solver binary is stale vs source — rebuilds if needed
6. Runs Python preflight tests (pytest)
7. Shows last run state (result.json summary if present)
8. Detects stale checkpoints and prompts to clear
9. Prompts to launch the dashboard

### Starting a New Run

After session_start.sh, to launch SIMP:
```bash
bash scripts/simp_dashboard.sh auto <filter_radius> <volume_fraction> <max_iterations>
```

Confirmed working parameters:
```bash
bash scripts/simp_dashboard.sh auto 3.0 0.45 200
```

### Before Starting a New Claude Session on This Project

Load both documents into context:
1. `MATERIALS_SPEC.md` — physics, math, parameter meanings
2. `SOLVER_STATE.md` — (this document) code structure, decisions, tech debt

Then provide: current git HEAD hash, cargo test count, and the specific
task for the session. Do NOT start coding before the audit-first protocol:
confirm HEAD, test count, and exact function signatures of files to be modified.

### Audit-First Protocol (Mandatory Before Any Code Change)

1. Confirm git HEAD: `git log --oneline -3`
2. Confirm test count: `cargo test --release 2>&1 | tail -3`
3. Paste exact function signature and surrounding context of files to change
4. Agree on design decisions in writing before any implementation
5. After changes: `cargo build --release 2>&1 | grep "^error"` then
   `cargo test --release 2>&1 | tail -5`
6. Only commit after green tests


---

## 14. Session Changes (2026-05-22 to 2026-05-24)

### NB04 Diagnostic Layer Added

Three new cells added to `notebooks/04_simp_optimization.ipynb`:

**Cell 5b — Density quality diagnostics**
- Computes gray element fraction against design mask (not total grid)
- Classifies convergence shape: healthy_monotone / mostly_monotone /
  oscillating / early_plateau / insufficient_data
- Stores results in `_diag` dict for downstream cells
- Runtime: <1s (pure numpy, no FEA)

**Cell 5c — Orthogonal slice plots**
- Generates XY, XZ, YZ mid-plane density slices unconditionally
- Always runs regardless of RENDER_PLOTS setting
- Saves to `outputs/reports/{part_name}_density_slices.png`
- Uses correct array orientation: `density[iz, iy, ix]` with physical mm axes

**Modified Cell 6 — Stage04 handoff**
- Now writes `compliance_history` and `volume_history` to stage04.json
  (previously stripped — downstream notebooks couldn't access them)
- Appends `diagnostic` block to stage04.json
- Writes standalone `outputs/reports/{part_name}_simp_diagnostic.json`
- Prints formatted terminal summary at end of run

### NB05 Shell Enforcement Added

New Cell 2b added to `notebooks/05_stl_export.ipynb` between density load
and marching cubes:

```python
SHELL_ENFORCE_FACES = ["x_min", "x_max", "y_min", "y_max"]
SHELL_THICKNESS_MM  = 8.0
```

**Mechanism**: loads `outputs/problem/void.bin`, builds face slice map,
forces outer voxels solid where `void_mask == False`. The modified density
flows into the existing Cell 3 (marching cubes) unchanged.

**Why this works without junction artifacts**: the shell modifies the
density array before marching cubes — it is one continuous field, not two
geometries being merged. The slab approach failed because it created
separate geometry that had to be joined post-marching-cubes.

**Confirmed SHELL_THICKNESS_MM values**:
- 2mm: FAIL — Taubin smoothing erases thin shell between strut anchor points
- 8mm: PASS — survives smoothing, leaves meaningful interior design space
- 15mm: PASS but over-constrains — too much forced material

### Through-Ring Passive Element

**Problem solved**: bolt holes appeared as corner cutouts because the
optimizer had no incentive to build material around the void cylinders
in the interior (middle 40mm) of the part. Entry/exit collar geometry
(seat_depth=15mm) only influenced the first and last 15mm.

**Implementation**:

`src/geometry/param_schema.py` — `BoltSeatRegion` dataclass:
```python
through_ring_radius_m: Optional[float] = None
```

`scripts/voxelize.py` — Phase 4 bolt seat loop, after collar logic:
```python
if hasattr(region, 'through_ring_radius_m') \
        and region.through_ring_radius_m is not None:
    ring_r2   = region.through_ring_radius_m ** 2
    ring_mask = (r2 >= void_r2) & (r2 < ring_r2)
    nondesign[ring_mask] = 1
```

No `axis_coord` constraint — applies full axis length.
Backward compatible — None default means existing params.json unchanged.

`scad/motor_mount_params.json` — both bolt_seat groups:
```json
"through_ring_radius_m": 0.007
```

**Volume budget**: 6.8% of grid at motor mount dimensions.
Previous full-collar approach consumed 40% — optimizer produced 8 solid
columns with minimal bridging.

### Confirmed Working Parameters (motor_mount v3)

```
filter_radius:         3.0mm (changed from 6.0mm)
through_ring_radius_m: 0.007 (new)
SHELL_ENFORCE_FACES:   ["x_min","x_max","y_min","y_max"]
SHELL_THICKNESS_MM:    8.0
```

Run results:
```
iterations:     238 (converged, stage1+stage2)
final_C:        0.03583 J (best yet — 12.5% improvement over 6mm filter run)
gray_fraction:  <5% PASS
SF:             11.09 PASS
watertight:     true
open_edges:     0
```

### Updated Inter-Stage Handoff: stage04.json

Schema additions (all new fields populated by modified Cell 6):

```json
{
  ...existing fields...,
  "compliance_history": [...],
  "volume_history":     [...],
  "diagnostic": {
    "gray_fraction_all":    0.031,
    "gray_fraction_design": 0.028,
    "gray_status":          "PASS",
    "solid_fraction":       0.381,
    "void_fraction":        0.619,
    "convergence_shape":    "mostly_monotone",
    "convergence_note":     "...",
    "final_spread":         8.3e-6,
    "stage1_iters":         91,
    "stage2_iters":         147,
    "slice_plots_png":      "outputs/reports/motor_mount_density_slices.png"
  }
}
```

### Diagnostic Report: {part}_simp_diagnostic.json

New file written to `outputs/reports/` by NB04 Cell 6. Contains:
- Full compliance_history and volume_history arrays
- Convergence block: shape, spread, reduction_pct, avg_iter_s
- Density quality block: gray fractions, solid/void fractions, status
- Volume constraint block: target, achieved, error_pct
- Config block: all SIMP parameters used
- Artifacts block: paths to all output files

This is the file to share in a review session for quantitative evaluation.

### Updated Python Pipeline Module Map: voxelize.py

`BoltSeatRegion` now supports `through_ring_radius_m` (Optional[float]).
When set, applies a thin forced-solid annulus around the void for the full
axis length. When None (default), behaviour is identical to original.

The through-ring ring_mask computation:
```python
ring_mask = (r2 >= void_r2) & (r2 < ring_r2)
```
Lower bound `r2 >= void_r2` ensures the void core is never overwritten.
No axis constraint — full part depth is covered.

### Updated Tech Debt

**TD-08: Dimensional accuracy oversize from shell enforcement (NEW)**
- Parts with SHELL_ENFORCEMENT active report ~2.7% oversize on all axes
- Root cause: 8mm shell + marching cubes interpolation adds ~2-3mm per face
- Impact: dimensional WARN in NB06 for every shell-enforced part
- Fix options: (a) subtract SHELL_THICKNESS_MM/2 from marching cubes
  spacing; (b) post-process STL to clip to bounding box; (c) accept and drill
- Priority: LOW for prototyping, MEDIUM for production fitment

**TD-09: Through-ring holes have organic edges (NEW)**
- At 1mm ring thickness, the marching cubes surface at hole boundaries
  is slightly irregular — not a perfect cylinder
- Fix: increase through_ring_radius_m from 0.007 to 0.0075 or 0.008
- Priority: LOW — holes are functional and usable for FDM prototyping

### Architectural Decisions Added

**AD-10: Shell enforcement must be void-aware**
The shell enforcement in NB05 MUST load void.bin and check the void mask
before forcing voxels solid. Failing to do so fills bolt holes. The
void.bin path is `outputs/problem/void.bin` — regenerated every NB04 run.
If void.bin is stale (from a different grid size), shell enforcement will
silently produce incorrect hole geometry.

**AD-11: Slab approach is permanently retired**
Forced-solid rectangular slabs (via attachment_regions) have been attempted
multiple times and always produce 900+ non-manifold faces at the junction
between slab and optimizer topology. This is an architectural incompatibility
with marching cubes, not a parameter tuning problem. Do not re-propose slabs.
The confirmed replacement is shell enforcement (AD-12) + through-ring (AD-13).

**AD-12: Shell enforcement is correct for connectivity**
Shell enforcement operates on the density array before marching cubes — one
continuous field. This is architecturally correct and produces zero junction
artifacts. The shell thickness must be ≥ bolt_seat seat_depth_m to survive
Taubin smoothing.

**AD-13: Through-ring is correct for bolt hole geometry**
Through-ring passive elements (1mm forced-solid annulus, full axis length)
give the optimizer geometry to anchor around without consuming the volume
budget. The confirmed ring thickness is 1mm (void_radius + 0.001m).
Full-depth thick collars (the previously failed approach) must not be
re-attempted — they consumed 40% of material budget.