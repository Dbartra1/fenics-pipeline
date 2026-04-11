// src/main.rs
#![allow(dead_code)]

mod types;
mod connectivity;
mod ke_base;
mod filter;
mod assembly;

use types::{Grid, Material, SimpConfig, RHO_MIN};
use connectivity::{precompute_connectivity, precompute_dof_map};
use ke_base::compute_ke_base;
use filter::build_filter;
use assembly::{build_csr_pattern, assemble_k, csr_matvec};

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
    println!("Ke trace: {trace:.6e}  max entry: {ke_max:.6e}");
    assert!((ke[1*24+0] - ke[0*24+1]).abs() < 1e-6 * ke_max);

    // ── filter smoke ──────────────────────────────────────────────────────────
    let fw = build_filter(&grid, cfg.filter_radius);
    println!(
        "Filter: elem 0 has {} neighbours, weight_sum={:.4e}",
        fw.weights[0].len(), fw.weight_sums[0]
    );

    // ── assembly smoke ────────────────────────────────────────────────────────
    let pattern = build_csr_pattern(&grid, &dof_map);
    let nnz = pattern.k_rows[grid.n_dof()];
    println!("K: {} DOFs, {} nonzeros, fill={:.4}%",
        grid.n_dof(), nnz,
        100.0 * nnz as f64 / (grid.n_dof() * grid.n_dof()) as f64
    );

    let n_elem = grid.n_elem();
    let mut k_vals = vec![0.0f64; nnz];
    assemble_k(
        &mut k_vals, &vec![1.0f64; n_elem], &ke, &pattern,
        &vec![false; n_elem], &vec![false; n_elem], cfg.penal,
    );

    // Rigid body x-translation → zero force
    let mut u = vec![0.0f64; grid.n_dof()];
    for i in (0..grid.n_dof()).step_by(3) { u[i] = 1.0; }
    let f = csr_matvec(&pattern.k_rows, &pattern.k_cols, &k_vals, &u);
    let f_max = f.iter().cloned().fold(0.0f64, f64::max.clone());
    println!("K·u (rigid x-translation), max force component: {f_max:.3e}  (should be ~0)");

    println!("\n✓ All smoke checks passed.");
}