// src/main.rs
#![allow(dead_code)]

mod types;
mod connectivity;
mod ke_base;
mod filter;
mod assembly;
mod solver;
mod sensitivity;
mod oc_update;

use types::{Grid, Material, SimpConfig, RHO_MIN};
use connectivity::{precompute_connectivity, precompute_dof_map};
use ke_base::compute_ke_base;
use filter::build_filter;
use assembly::{apply_dirichlet, assemble_k, build_csr_pattern};
use solver::cg_solve_direct;
use sensitivity::{compute_compliance, compute_sensitivities};
use oc_update::oc_update;

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

    // ── precompute ────────────────────────────────────────────────────────────
    let conn    = precompute_connectivity(&grid);
    let dof_map = precompute_dof_map(&grid);
    let ke      = compute_ke_base(&steel, grid.voxel_size);
    let fw      = build_filter(&grid, cfg.filter_radius);
    println!("conn[0]: {:?}", conn[0]);
    println!("Filter:  elem 0 has {} neighbours", fw.weights[0].len());

    let pattern = build_csr_pattern(&grid, &dof_map);
    let nnz = pattern.k_rows[grid.n_dof()];
    println!("K: {} DOFs, {} nonzeros", grid.n_dof(), nnz);

    let n_elem = grid.n_elem();
    let void_mask = vec![false; n_elem];
    let nondesign = vec![false; n_elem];
    let mut x = vec![cfg.volume_fraction; n_elem];

    // ── fixed BCs and force vector ────────────────────────────────────────────
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
    let mut f_vec = vec![0.0f64; grid.n_dof()];
    let top_count = (grid.nx + 1) * (grid.ny + 1);
    for iy in 0..=grid.ny {
        for ix in 0..=grid.nx {
            let n = grid.node_idx(ix, iy, grid.nz);
            f_vec[3*n + 2] = -10000.0 / top_count as f64;
        }
    }
    for &d in &fixed_dofs { f_vec[d] = 0.0; }

    // ── 3-iteration SIMP mini-loop ────────────────────────────────────────────
    for iter in 0..3 {
        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &x, &ke, &pattern, &void_mask, &nondesign, cfg.penal);

        let diag_mean: f64 = (0..grid.n_dof()).map(|i| {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
            let pos = row.binary_search(&i).unwrap();
            k_vals[pattern.k_rows[i] + pos]
        }).sum::<f64>() / grid.n_dof() as f64;
        apply_dirichlet(&mut k_vals, &pattern.k_rows, &pattern.k_cols,
                        &fixed_dofs, diag_mean);

        let mut u = vec![0.0f64; grid.n_dof()];
        let cg = cg_solve_direct(&pattern.k_rows, &pattern.k_cols, &k_vals,
                                  &f_vec, &mut u, 1e-8, grid.n_dof() * 2);

        let compliance = compute_compliance(&x, &u, &ke, &dof_map, cfg.penal,
                                            &void_mask, &nondesign);
        let dc = compute_sensitivities(&x, &u, &ke, &dof_map, &fw, cfg.penal,
                                       &void_mask, &nondesign);
        let oc = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);

        println!("Iter {:2} | C={:.4e} | Vol={:.3} | Δρ={:.4e} | CG={}iters",
                 iter + 1, compliance, oc.vol_frac, oc.rho_change, cg.iterations);
        x = oc.x_new;
    }

    println!("\n✓ All smoke checks passed.");
}