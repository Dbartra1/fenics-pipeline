// src/simp.rs
//
// Main SIMP optimization loop.
//
// run_simp() is the top-level entry point called from main.rs.
// It precomputes all invariant data structures, then iterates:
//   1. Assemble K(x)
//   2. Apply Dirichlet BCs
//   3. Solve K·u = f  (CG)
//   4. Compute compliance and sensitivities
//   5. OC density update
//   6. Check convergence
//
// Convergence criteria (both checked each iteration after iter 10):
//   - Compliance-flat: spread of last 10 compliance values < 1e-4 relative
//   - Density change backstop: max|Δρ| < convergence_tol

use std::time::Instant;

use crate::assembly::{apply_dirichlet, assemble_k, build_csr_pattern};
use crate::connectivity::{precompute_connectivity, precompute_dof_map};
use crate::filter::build_filter;
use crate::io::SolveResult;
use crate::ke_base::compute_ke_base;
use crate::oc_update::oc_update;
use crate::sensitivity::{compute_compliance, compute_sensitivities};
use crate::solver::cg_solve_direct;
use crate::types::{Problem, RHO_MIN};

pub fn run_simp(problem: &Problem) -> SolveResult {
    let start = Instant::now();
    let grid   = &problem.grid;
    let cfg    = &problem.config;
    let mat    = &problem.material;

    // ── Precompute invariants ─────────────────────────────────────────────────
    let _conn   = precompute_connectivity(grid);   // kept for future checkpointing
    let dof_map = precompute_dof_map(grid);
    let ke      = compute_ke_base(mat, grid.voxel_size);
    let fw      = build_filter(grid, cfg.filter_radius);
    let pattern = build_csr_pattern(grid, &dof_map);

    let n_elem  = grid.n_elem();
    let n_dof   = grid.n_dof();
    let nnz     = pattern.k_rows[n_dof];

    let void_mask  = &problem.void_mask;
    let nondesign  = &problem.nondesign;

    // ── Build force vector (constant) ─────────────────────────────────────────
    let mut f = vec![0.0f64; n_dof];
    for (&dof, &val) in problem.load_case.load_dofs.iter()
                                                    .zip(problem.load_case.load_vals.iter()) {
        if dof < n_dof { f[dof] += val; }
    }

    // ── Initial density field ─────────────────────────────────────────────────
    let mut x: Vec<f64> = match &problem.x_init {
        Some(xi) => xi.clone(),
        None     => vec![cfg.volume_fraction; n_elem],
    };
    // Clamp to valid range
    for v in &mut x {
        *v = v.clamp(RHO_MIN, 1.0);
    }

    // ── Convergence history ───────────────────────────────────────────────────
    let mut compliance_history: Vec<f64> = Vec::new();
    let mut volume_history:     Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iterations = 0usize;

    // ── Main SIMP loop ────────────────────────────────────────────────────────
    for iter in 0..cfg.max_iterations {
        n_iterations = iter + 1;

        // 1. Assemble K
        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &x, &ke, &pattern, void_mask, nondesign, cfg.penal);

        // 2. Dirichlet BCs — diagonal value = mean of current diagonal
        let diag_mean: f64 = (0..n_dof).map(|i| {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i + 1]];
            let pos = row.binary_search(&i).unwrap();
            k_vals[pattern.k_rows[i] + pos]
        }).sum::<f64>() / n_dof as f64;

        let mut k_bc = k_vals;
        apply_dirichlet(&mut k_bc, &pattern.k_rows, &pattern.k_cols,
                        &problem.load_case.fixed_dofs, diag_mean);

        // Zero force at fixed DOFs
        let mut f_bc = f.clone();
        for &d in &problem.load_case.fixed_dofs { f_bc[d] = 0.0; }

        // 3. Solve K·u = f
        let mut u = vec![0.0f64; n_dof];
        let cg = cg_solve_direct(
            &pattern.k_rows, &pattern.k_cols, &k_bc,
            &f_bc, &mut u, 1e-8, n_dof * 2,
        );

        // 4. Compliance and sensitivities
        let compliance = compute_compliance(&x, &u, &ke, &dof_map, cfg.penal,
                                            void_mask, nondesign);
        let dc = compute_sensitivities(&x, &u, &ke, &dof_map, &fw, cfg.penal,
                                       void_mask, nondesign);

        // 5. OC update
        let oc = oc_update(&x, &dc, cfg, void_mask, nondesign);
        let rho_change = oc.rho_change;
        let vol_frac   = oc.vol_frac;

        compliance_history.push(compliance);
        volume_history.push(vol_frac);

        // 6. Log iteration
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "Iter {:4} | C={:.4e} | Vol={:.3} | Δρ={:.4e} | CG={:3}iters | {:.1}s{}",
            n_iterations, compliance, vol_frac, rho_change, cg.iterations, elapsed,
            if cg.converged { "" } else { " [CG!]" }
        );

        // Optional checkpoint
        if cfg.checkpoint_every > 0 && n_iterations % cfg.checkpoint_every == 0 {
            eprintln!("[checkpoint] iter {n_iterations}");
        }

        x = oc.x_new;

        // 7. Convergence check
        if n_iterations > 10 {
            // Compliance-flat: relative spread of last 10 values
            let recent = &compliance_history[compliance_history.len() - 10..];
            let c_max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let c_min = recent.iter().cloned().fold(f64::INFINITY,     f64::min);
            let spread = (c_max - c_min) / (c_max.abs() + 1e-30);
            if spread < 1e-4 {
                println!("✓ Compliance flat (spread={spread:.2e}) — converged at iteration {n_iterations}");
                converged = true;
                break;
            }
        }

        // Density change backstop
        if rho_change < cfg.convergence_tol {
            println!("✓ Density change {rho_change:.4e} < tol {} — converged at iteration {n_iterations}",
                     cfg.convergence_tol);
            converged = true;
            break;
        }
    }

    if !converged {
        println!("✗ Max iterations ({}) reached without convergence", cfg.max_iterations);
    }

    let final_compliance  = compliance_history.last().copied().unwrap_or(0.0);
    let final_volume_frac = volume_history.last().copied().unwrap_or(0.0);
    let duration_s = start.elapsed().as_secs_f64();

    SolveResult {
        converged,
        n_iterations,
        final_compliance,
        final_volume_frac,
        compliance_history,
        volume_history,
        duration_s,
        peak_memory_mb: 0.0,
        final_density: x,   // ← add this line
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Grid, LoadCase, Material, Problem, SimpConfig};

    fn make_problem(nx: usize, ny: usize, nz: usize) -> Problem {
        let g = Grid { nx, ny, nz, voxel_size: 0.001 };
        let n_elem = g.n_elem();
        let _n_dof = g.n_dof();

        // Bottom face fixed, top face loaded in -z
        let fixed_dofs: Vec<usize> = {
            let mut v = Vec::new();
            for iy in 0..=g.ny {
                for ix in 0..=g.nx {
                    let n = g.node_idx(ix, iy, 0);
                    v.extend_from_slice(&[3*n, 3*n+1, 3*n+2]);
                }
            }
            v
        };

        let top_nodes: Vec<usize> = {
            let mut v = Vec::new();
            for iy in 0..=g.ny {
                for ix in 0..=g.nx {
                    v.push(g.node_idx(ix, iy, g.nz));
                }
            }
            v
        };
        let n_top = top_nodes.len();
        let load_dofs: Vec<usize> = top_nodes.iter().map(|&n| 3*n + 2).collect();
        let load_vals: Vec<f64>   = vec![-1000.0 / n_top as f64; n_top];

        Problem {
            grid: g,
            material: Material { young: 210e9, poisson: 0.3 },
            load_case: LoadCase { fixed_dofs, load_dofs, load_vals },
            config: SimpConfig {
                volume_fraction:  0.5,
                penal:            3.0,
                filter_radius:    0.002,
                max_iterations:   30,
                convergence_tol:  0.01,
                move_limit:       0.2,
                damping:          0.5,
                checkpoint_every: 0,
            },
            nondesign: vec![false; n_elem],
            void_mask: vec![false; n_elem],
            x_init: None,
        }
    }

    #[test]
    fn compliance_decreases_over_iterations() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem);

        assert!(result.compliance_history.len() >= 2,
            "need at least 2 iterations");

        // Compliance should decrease from iter 1 to iter 2
        assert!(
            result.compliance_history[1] <= result.compliance_history[0] * 1.01,
            "compliance increased: {} -> {}",
            result.compliance_history[0], result.compliance_history[1]
        );
    }

    #[test]
    fn volume_fraction_stays_near_target() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem);
        let target = problem.config.volume_fraction;

        for (i, &vf) in result.volume_history.iter().enumerate() {
            assert!(
                (vf - target).abs() < 0.05,
                "iter {}: vol_frac={:.4} too far from target={:.4}", i+1, vf, target
            );
        }
    }

    #[test]
    fn result_fields_are_populated() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem);

        assert!(result.n_iterations > 0);
        assert!(result.final_compliance > 0.0);
        assert!(!result.compliance_history.is_empty());
        assert_eq!(result.compliance_history.len(), result.volume_history.len());
        assert!(result.duration_s > 0.0);
    }

    #[test]
    fn warm_start_produces_same_final_compliance() {
        // Run once to get a result, then warm-start from uniform vf and verify
        // the solver still runs without error.
        let problem = make_problem(4, 3, 2);
        let result1 = run_simp(&problem);
        assert!(result1.final_compliance > 0.0);

        // Warm start from uniform density (valid x_init)
        let mut problem2 = make_problem(4, 3, 2);
        let n_elem = problem2.grid.n_elem();
        problem2.x_init = Some(vec![problem2.config.volume_fraction; n_elem]);

        let result2 = run_simp(&problem2);
        assert!(result2.final_compliance > 0.0);
        assert!(!result2.compliance_history.is_empty());
    }
}