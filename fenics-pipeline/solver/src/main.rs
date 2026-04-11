// src/main.rs
#![allow(dead_code)]

mod types;
mod connectivity;
mod ke_base;
mod filter;
mod assembly;
mod solver;
mod sensitivity;

use types::{Grid, Material, SimpConfig, RHO_MIN};
use connectivity::{precompute_connectivity, precompute_dof_map};
use ke_base::compute_ke_base;
use filter::build_filter;
use assembly::{apply_dirichlet, assemble_k, build_csr_pattern};
use solver::cg_solve_direct;
use sensitivity::{compute_compliance, compute_sensitivities};

fn main() {
    // ── types smoke ───────────────────────────────────────────────────────────
    let grid = Grid { nx: 10, ny: 6, nz: 4, voxel_size: 0.0025 };
    assert_eq!(grid.node_idx(0, 0, 0), 0);
    assert_eq!(grid.node_idx(grid.nx, grid.ny, grid.nz), grid.n_nodes() - 1);
    assert_eq!(grid.elem_idx(0, 0, 0), 0);
    assert_eq!(grid.elem_idx(grid.nx - 1, grid.ny - 1, grid.nz - 1), grid.n_elem() - 1);
    println!("Elements: {}  Nodes: {}  DOFs: {}",
             grid.n_elem(), grid.n_nodes(), grid.n_dof());

    let steel = Material { young: 210e9, poisson: 0.3 };
    let cfg = SimpConfig {
        volume_fraction: 0.45, penal: 3.0, filter_radius: 0.008,
        max_iterations: 200, convergence_tol: 0.002,
        move_limit: 0.05, damping: 0.5, checkpoint_every: 10,
    };
    assert!(cfg.validate().is_ok());
    println!("rho_min: {}", RHO_MIN);

    // ── connectivity + ke + filter ────────────────────────────────────────────
    let conn    = precompute_connectivity(&grid);
    let dof_map = precompute_dof_map(&grid);
    let ke      = compute_ke_base(&steel, grid.voxel_size);
    let fw      = build_filter(&grid, cfg.filter_radius);
    println!("conn[0]: {:?}", conn[0]);
    println!("Filter:  elem 0 has {} neighbours", fw.weights[0].len());

    // ── assembly ──────────────────────────────────────────────────────────────
    let pattern = build_csr_pattern(&grid, &dof_map);
    let nnz = pattern.k_rows[grid.n_dof()];
    println!("K: {} DOFs, {} nonzeros", grid.n_dof(), nnz);

    let n_elem = grid.n_elem();
    let x = vec![cfg.volume_fraction; n_elem];
    let mut k_vals = vec![0.0f64; nnz];
    assemble_k(&mut k_vals, &x, &ke, &pattern,
               &vec![false; n_elem], &vec![false; n_elem], cfg.penal);

    // ── boundary conditions ───────────────────────────────────────────────────
    let fixed_dofs: Vec<usize> = {
        let mut v = Vec::new();
        for iy in 0..=grid.ny {
            for ix in 0..=grid.nx {
                let n = grid.node_idx(ix, iy, 0);
                v.extend_from_slice(&[3*n, 3*n+1, 3*n+2]);
            }
        }
        v
    };
    let diag_mean: f64 = (0..grid.n_dof()).map(|i| {
        let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
        let pos = row.binary_search(&i).unwrap();
        k_vals[pattern.k_rows[i] + pos]
    }).sum::<f64>() / grid.n_dof() as f64;
    apply_dirichlet(&mut k_vals, &pattern.k_rows, &pattern.k_cols,
                    &fixed_dofs, diag_mean);

    let mut f_vec = vec![0.0f64; grid.n_dof()];
    let top_count = (grid.nx + 1) * (grid.ny + 1);
    for iy in 0..=grid.ny {
        for ix in 0..=grid.nx {
            let n = grid.node_idx(ix, iy, grid.nz);
            f_vec[3*n + 2] = -10000.0 / top_count as f64;
        }
    }
    for &d in &fixed_dofs { f_vec[d] = 0.0; }

    // ── solve ─────────────────────────────────────────────────────────────────
    let mut u = vec![0.0f64; grid.n_dof()];
    let cg = cg_solve_direct(&pattern.k_rows, &pattern.k_cols, &k_vals,
                              &f_vec, &mut u, 1e-8, grid.n_dof() * 2);
    println!("CG: {} iters, rel_res={:.3e}, converged={}", 
             cg.iterations, cg.rel_residual, cg.converged);

    // ── sensitivity ───────────────────────────────────────────────────────────
    let dc = compute_sensitivities(
        &x, &u, &ke, &dof_map, &fw, cfg.penal,
        &vec![false; n_elem], &vec![false; n_elem],
    );
    let compliance = compute_compliance(
        &x, &u, &ke, &dof_map, cfg.penal,
        &vec![false; n_elem], &vec![false; n_elem],
    );
    let dc_min = dc.iter().cloned().fold(0.0f64, f64::min);
    let dc_max = dc.iter().cloned().fold(0.0f64, f64::max);
    println!("Compliance: {compliance:.6e}");
    println!("dc range:   [{dc_min:.4e}, {dc_max:.4e}]");

    println!("\n✓ All smoke checks passed.");
}