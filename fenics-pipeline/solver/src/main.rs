// src/main.rs
#![allow(dead_code)]

mod types;
mod connectivity;
mod ke_base;
mod filter;
mod assembly;
mod solver;

use types::{Grid, Material, SimpConfig, RHO_MIN};
use connectivity::{precompute_connectivity, precompute_dof_map};
use ke_base::compute_ke_base;
use filter::build_filter;
use assembly::{apply_dirichlet, assemble_k, build_csr_pattern, csr_matvec};
use solver::cg_solve_direct;

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

    // ── solver smoke ──────────────────────────────────────────────────────────
    // Fix bottom face, apply unit z-load on top face, solve.
    let nx = grid.nx; let ny = grid.ny;
    let fixed_dofs: Vec<usize> = (0..=(nx)*(ny+1)-(ny+1))
        .flat_map(|n| [3*n, 3*n+1, 3*n+2])
        .collect();

    // Bottom face nodes: iz=0, all ix,iy
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

    let diag_mean: f64 = (0..grid.n_dof())
        .map(|i| {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
            let pos = row.binary_search(&i).unwrap();
            k_vals[pattern.k_rows[i] + pos]
        })
        .sum::<f64>() / grid.n_dof() as f64;

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

    let mut u = vec![0.0f64; grid.n_dof()];
    let result = cg_solve_direct(
        &pattern.k_rows, &pattern.k_cols, &k_vals, &f_vec, &mut u, 1e-8, grid.n_dof() * 2
    );
    println!("CG solve: {} iters, rel_res={:.3e}, converged={}", 
             result.iterations, result.rel_residual, result.converged);

    println!("\n✓ All smoke checks passed.");
}