// src/main.rs
//
// Smoke harness for types.rs — run with `cargo run`.
// This file will be replaced by the full CLI entry point (reads problem.json,
// calls run_simp(), writes outputs) once the solver modules are built.

mod types;

use types::{Grid, Material, SimpConfig, RHO_MIN};

fn main() {
    // ── Handoff §11 exact assertions ─────────────────────────────────────────
    let grid = Grid { nx: 10, ny: 6, nz: 4, voxel_size: 0.0025 };

    assert_eq!(grid.node_idx(0, 0, 0), 0);
    assert_eq!(grid.node_idx(grid.nx, grid.ny, grid.nz), grid.n_nodes() - 1);

    println!("Elements: {}", grid.n_elem());   // 240
    println!("Nodes:    {}", grid.n_nodes());  // 385
    println!("DOFs:     {}", grid.n_dof());    // 1155

    // ── elem_idx sanity ───────────────────────────────────────────────────────
    assert_eq!(grid.elem_idx(0, 0, 0), 0);
    assert_eq!(grid.elem_idx(grid.nx - 1, grid.ny - 1, grid.nz - 1), grid.n_elem() - 1);

    // ── Stride checks (same as unit tests, but visible in terminal) ───────────
    assert_eq!(grid.node_idx(1, 0, 0) - grid.node_idx(0, 0, 0), 1,        "x stride");
    assert_eq!(grid.node_idx(0, 1, 0) - grid.node_idx(0, 0, 0), 11,       "y stride");
    assert_eq!(grid.node_idx(0, 0, 1) - grid.node_idx(0, 0, 0), 11 * 7,   "z stride");

    // ── Material ──────────────────────────────────────────────────────────────
    let steel = Material { young: 210e9, poisson: 0.3 };
    println!("λ (steel): {:.4e} Pa", steel.lame_lambda());  // ~1.2115e11
    println!("μ (steel): {:.4e} Pa", steel.lame_mu());      // ~8.0769e10

    // ── SimpConfig rho_min ────────────────────────────────────────────────────
    let cfg = SimpConfig {
        volume_fraction: 0.45, penal: 3.0, filter_radius: 0.008,
        max_iterations: 200,   convergence_tol: 0.002,
        move_limit: 0.05,      damping: 0.5, checkpoint_every: 10,
    };
    assert!(cfg.validate().is_ok(), "baseline config should be valid");
    assert_eq!(cfg.rho_min(), RHO_MIN);
    println!("rho_min:  {}", cfg.rho_min());   // 0.001

    println!("\n✓ All types.rs smoke checks passed.");
}