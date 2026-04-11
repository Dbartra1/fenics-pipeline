// src/main.rs
#![allow(dead_code)]

mod types;
mod connectivity;
mod ke_base;

use types::{Grid, Material, SimpConfig, RHO_MIN};
use connectivity::{precompute_connectivity, precompute_dof_map};
use ke_base::compute_ke_base;

fn main() {
    // ── types smoke ───────────────────────────────────────────────────────────
    let grid = Grid { nx: 10, ny: 6, nz: 4, voxel_size: 0.0025 };
    assert_eq!(grid.node_idx(0, 0, 0), 0);
    assert_eq!(grid.node_idx(grid.nx, grid.ny, grid.nz), grid.n_nodes() - 1);
    assert_eq!(grid.elem_idx(0, 0, 0), 0);
    assert_eq!(grid.elem_idx(grid.nx - 1, grid.ny - 1, grid.nz - 1), grid.n_elem() - 1);
    println!("Elements: {}  Nodes: {}  DOFs: {}", grid.n_elem(), grid.n_nodes(), grid.n_dof());

    let steel = Material { young: 210e9, poisson: 0.3 };
    println!("λ={:.4e}  μ={:.4e}", steel.lame_lambda(), steel.lame_mu());

    let cfg = SimpConfig {
        volume_fraction: 0.45, penal: 3.0, filter_radius: 0.008,
        max_iterations: 200, convergence_tol: 0.002,
        move_limit: 0.05, damping: 0.5, checkpoint_every: 10,
    };
    assert!(cfg.validate().is_ok());
    println!("rho_min: {}", RHO_MIN);

    // ── connectivity smoke ────────────────────────────────────────────────────
    let conn    = precompute_connectivity(&grid);
    let dof_map = precompute_dof_map(&grid);
    println!("conn[0]:    {:?}", conn[0]);
    println!("dof_map[0]: {:?}", dof_map[0]);

    // ── ke_base smoke ─────────────────────────────────────────────────────────
    let ke = compute_ke_base(&steel, grid.voxel_size);
    let trace: f64 = (0..24).map(|i| ke[i * 24 + i]).sum();
    let ke_max = ke.iter().cloned().fold(0.0f64, f64::max);
    println!("Ke trace:   {trace:.6e}  max entry: {ke_max:.6e}");
    // Symmetry spot-check
    assert!((ke[1*24+0] - ke[0*24+1]).abs() < 1e-6 * ke_max, "Ke symmetry failed");

    println!("\n✓ All smoke checks passed.");
}