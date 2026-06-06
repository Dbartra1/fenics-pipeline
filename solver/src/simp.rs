// src/simp.rs
//
// Main SIMP optimization loop.
//
// run_simp(problem, out_dir) is called from main.rs.
// Two-stage penal continuation (penal=2 → penal=3) is handled externally
// by the notebook (04_simp_optimization.ipynb Cell 3: run_stage() twice).
// run_simp() is always single-stage, using cfg.penal throughout.
//
// ── Checkpoint / resume ───────────────────────────────────────────────────────
//
// Every cfg.checkpoint_every iterations, two files are written to out_dir:
//   checkpoint.bin       — f32 density field (identical format to density.bin)
//   checkpoint_meta.json — {iter_completed, n_elem, compliance_history,
//                           volume_history}
//
// On startup, if both files exist and n_elem matches, the run resumes from
// iter_completed rather than restarting.  The x_init_file field in problem.json
// is IGNORED when a checkpoint is found — the checkpoint takes precedence.
//
// On clean exit (converged or max_iterations), checkpoint files are deleted.
// On interrupted exit (kill, WSL close), checkpoint files remain on disk and
// will be picked up on the next invocation with the same out_dir.
//
// Stage 1 checkpoints live in outputs/problem/ (alongside problem_s1.json).
// Stage 2 checkpoints live in the same directory (alongside problem_s2.json).
// They never collide because each run_stage() call targets a different json_path,
// and out_dir is derived from json_path.parent() in main.rs.

use std::path::Path;
use std::time::Instant;

use crate::assembly::{apply_dirichlet, assemble_k, build_csr_pattern};
use crate::connectivity::{precompute_connectivity, precompute_dof_map};
use crate::filter::build_filter;
use crate::io::{write_checkpoint, read_checkpoint, delete_checkpoint,
                CheckpointMeta, SolveResult};
use crate::ke_base::compute_ke_base;
use crate::oc_update::oc_update;
use crate::sensitivity::{compute_compliance, compute_sensitivities};
use crate::vcycle_dispatch::{solve_linear_system, GpuContext};
use crate::types::{Problem, RHO_MIN};

pub fn run_simp(problem: &Problem, out_dir: &Path) -> SolveResult {
    let start  = Instant::now();
    let grid   = &problem.grid;
    let cfg    = &problem.config;
    let mat    = &problem.material;

    let _conn   = precompute_connectivity(grid);
    let dof_map = precompute_dof_map(grid);
    let ke      = compute_ke_base(mat, grid.voxel_size);
    let fw      = build_filter(grid, cfg.filter_radius);
    let pattern = build_csr_pattern(grid, &dof_map);

    let n_elem = grid.n_elem();
    let n_dof  = grid.n_dof();
    let nnz    = pattern.k_rows[n_dof];

    let void_mask = &problem.void_mask;
    let nondesign = &problem.nondesign;

    // One RHS vector per load case (loads are constant across SIMP iterations;
    // only the density / K changes between iterations).
    let n_lc = problem.load_cases.len();
    let f_per_case: Vec<Vec<f64>> = problem.load_cases.iter().map(|lc| {
        let mut f = vec![0.0f64; n_dof];
        for (&dof, &val) in lc.load_dofs.iter().zip(lc.load_vals.iter()) {
            if dof < n_dof { f[dof] += val; }
        }
        f
    }).collect();

    // ── Resume or fresh start ─────────────────────────────────────────────────
    //
    // Priority order:
    //   1. checkpoint.bin + checkpoint_meta.json  (interrupted prior run)
    //   2. problem.x_init_file                   (Stage 2 warm-start from Stage 1)
    //   3. uniform density at volume_fraction     (fresh Stage 1)
    //
    // This means an interrupted Stage 2 resumes from its own checkpoint, not
    // from x_init.bin.  A fresh Stage 2 uses x_init.bin as normal.

    let (mut x, mut compliance_history, mut volume_history, iter_start) =
        if cfg.checkpoint_every > 0 {
            match read_checkpoint(out_dir, n_elem) {
                Some((dens, meta)) => {
                    eprintln!(
                        "[resume] Resuming from checkpoint at iter {} \
                         (compliance={:.4e})",
                        meta.iter_completed,
                        meta.compliance_history.last().copied().unwrap_or(0.0)
                    );
                    (dens, meta.compliance_history, meta.volume_history, meta.iter_completed)
                }
                None => {
                    let x0 = match &problem.x_init {
                        Some(xi) => xi.clone(),
                        None     => vec![cfg.volume_fraction; n_elem],
                    };
                    (x0, Vec::new(), Vec::new(), 0)
                }
            }
        } else {
            let x0 = match &problem.x_init {
                Some(xi) => xi.clone(),
                None     => vec![cfg.volume_fraction; n_elem],
            };
            (x0, Vec::new(), Vec::new(), 0)
        };

    for v in &mut x { *v = v.clamp(RHO_MIN, 1.0); }

    let mut converged    = false;
    let mut n_iterations = iter_start;

    // One warm-start displacement vector per load case, persisted across iterations.
    let mut us: Vec<Vec<f64>> = vec![vec![0.0f64; n_dof]; n_lc];

    // ── Solver backend ────────────────────────────────────────────────────────
    let mut gpu_ctx = GpuContext::new(cfg.use_gpu);
    if iter_start == 0 {
        println!("Solver backend: {}", gpu_ctx.backend_label());
    } else {
        println!("Solver backend: {} [resuming from iter {}]",
                 gpu_ctx.backend_label(), iter_start);
    }

    // ── Main SIMP loop ────────────────────────────────────────────────────────
    for iter in iter_start..cfg.max_iterations {
        n_iterations = iter + 1;

        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &x, &ke, &pattern, void_mask, nondesign, cfg.penal);

        let diag_mean: f64 = (0..n_dof).map(|i| {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i + 1]];
            let pos = row.binary_search(&i).unwrap();
            k_vals[pattern.k_rows[i] + pos]
        }).sum::<f64>() / n_dof as f64;

        let mut k_bc = k_vals;
        apply_dirichlet(&mut k_bc, &pattern.k_rows, &pattern.k_cols,
                        &problem.fixed_dofs, diag_mean);

        // ── Solve every load case against the shared, BC-applied K, then
        //    aggregate (Phase 1: weighted sum of compliances/sensitivities).
        //    K is assembled and preconditioned once per iteration; each load
        //    case is a separate RHS with its own warm-started u in `us`.
        //    Filtering is linear in dc with x shared across cases, so summing
        //    filtered per-case sensitivities equals filtering the summed raw.
        let mut compliance     = 0.0f64;
        let mut dc             = vec![0.0f64; n_elem];
        let mut cg_iters_total = 0usize;
        let mut worst_residual = 0.0f64;
        let mut all_converged  = true;

        for (lc_i, lc) in problem.load_cases.iter().enumerate() {
            let mut f_bc = f_per_case[lc_i].clone();
            for &d in &problem.fixed_dofs { f_bc[d] = 0.0; }

            let solve = solve_linear_system(
                &pattern.k_rows, &pattern.k_cols, &k_bc,
                &f_bc, &mut us[lc_i],
                1e-6,
                cfg.max_cg_iter,
                grid.nx + 1, grid.ny + 1, grid.nz + 1,
                &mut gpu_ctx,
            );

            let c_i  = compute_compliance(&x, &us[lc_i], &ke, &dof_map, cfg.penal,
                                          void_mask, nondesign);
            let dc_i = compute_sensitivities(&x, &us[lc_i], &ke, &dof_map, &fw, cfg.penal,
                                             void_mask, nondesign);

            compliance += lc.weight * c_i;
            for (acc, &v) in dc.iter_mut().zip(dc_i.iter()) { *acc += lc.weight * v; }

            cg_iters_total += solve.iterations;
            if solve.rel_residual > worst_residual { worst_residual = solve.rel_residual; }
            if !solve.converged { all_converged = false; }
        }

        let oc        = oc_update(&x, &dc, cfg, void_mask, nondesign);
        let rho_change = oc.rho_change;
        let vol_frac   = oc.vol_frac;
        let elapsed    = start.elapsed().as_secs_f64();

        compliance_history.push(compliance);
        volume_history.push(vol_frac);

        println!(
            "Iter {:4} | C={:.4e} | Vol={:.3} | Δρ={:.4e} | p={} | LC={} | CG={:4} | res={:.1e} | {:.1}s{}",
            n_iterations, compliance, vol_frac, rho_change, cfg.penal,
            n_lc, cg_iters_total, worst_residual, elapsed,
            if all_converged { "" } else { " [SOLVE!]" }
        );

        // ── Checkpoint write ──────────────────────────────────────────────────
        // Written AFTER x is updated (oc.x_new) so the checkpoint contains the
        // density that will be the warm-start for the NEXT iteration, not the
        // one just solved.  This means on resume we correctly start iter N+1.
        x = oc.x_new;

        if cfg.checkpoint_every > 0 && n_iterations % cfg.checkpoint_every == 0 {
            let meta = CheckpointMeta {
                iter_completed:     n_iterations,
                compliance_history: compliance_history.clone(),
                volume_history:     volume_history.clone(),
                n_elem,
            };
            match write_checkpoint(out_dir, &x, &meta) {
                Ok(()) => eprintln!("[checkpoint] iter {} written to {:?}",
                                    n_iterations, out_dir),
                Err(e) => eprintln!("[checkpoint] WARNING: write failed: {e}"),
            }
        }

        // ── Convergence checks ────────────────────────────────────────────────
        if n_iterations > cfg.min_iterations {
            let hist_len = compliance_history.len();
            if hist_len >= 10 {
                let recent = &compliance_history[hist_len - 10..];
                let c_max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let c_min = recent.iter().cloned().fold(f64::INFINITY,     f64::min);
                let spread = (c_max - c_min) / (c_max.abs() + 1e-30);
                let s_tol = cfg.spread_tol();
                if spread < s_tol {
                    println!(
                        "✓ Compliance flat (spread={spread:.2e} < tol={s_tol:.2e}) \
                         — converged at iteration {n_iterations} (min_iter={})",
                        cfg.min_iterations
                    );
                    converged = true;
                    break;
                }
            }
        }

        if n_iterations > cfg.min_iterations {
            let d_tol = cfg.density_tol();
            if rho_change < d_tol {
                println!(
                    "✓ Density change {rho_change:.4e} < tol {d_tol:.2e} \
                     — converged at iteration {n_iterations} (min_iter={})",
                    cfg.min_iterations
                );
                converged = true;
                break;
            }
        }
    }

    if !converged {
        println!("✗ Max iterations ({}) reached without convergence", cfg.max_iterations);
    }

    // ── Clean up checkpoint on successful exit ────────────────────────────────
    // Checkpoint is only useful for resuming an interrupted run.  Once the
    // solver exits cleanly (converged or max_iterations), the checkpoint is
    // stale — delete it so a fresh re-run doesn't accidentally resume.
    if cfg.checkpoint_every > 0 {
        delete_checkpoint(out_dir);
    }

    SolveResult {
        converged,
        n_iterations,
        final_compliance:  compliance_history.last().copied().unwrap_or(0.0),
        final_volume_frac: volume_history.last().copied().unwrap_or(0.0),
        compliance_history,
        volume_history,
        duration_s:        start.elapsed().as_secs_f64(),
        peak_memory_mb:    0.0,
        final_density:     x,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Grid, LoadCase, Material, Problem, SimpConfig};

    fn make_problem(nx: usize, ny: usize, nz: usize) -> Problem {
        let g = Grid { nx, ny, nz, voxel_size: 0.001 };
        let n_elem = g.n_elem();

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
            fixed_dofs,
            load_cases: vec![LoadCase {
                name: "test".to_string(),
                weight: 1.0,
                load_dofs,
                load_vals,
                fixed_dofs: None,
            }],
            config: SimpConfig {
                use_gpu:               false,
                volume_fraction:       0.5,
                penal:                 3.0,
                filter_radius:         0.002,
                max_iterations:        30,
                min_iterations:        10,
                convergence_tol:       0.01,
                compliance_spread_tol: None,
                density_change_tol:    None,
                move_limit:            0.2,
                damping:               0.5,
                checkpoint_every:      0,   // no checkpoint I/O in tests
                max_cg_iter:           2000,
            },
            nondesign: vec![false; n_elem],
            void_mask: vec![false; n_elem],
            x_init: None,
        }
    }

    // Tests pass Path::new("/tmp") as out_dir.  Checkpoint writes are disabled
    // (checkpoint_every=0) so no files are created there.

    #[test]
    fn compliance_decreases_over_iterations() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem, Path::new("/tmp"));
        assert!(result.compliance_history.len() >= 2);
        assert!(
            result.compliance_history[1] <= result.compliance_history[0] * 1.01,
            "compliance increased: {} -> {}",
            result.compliance_history[0], result.compliance_history[1]
        );
    }

    #[test]
    fn volume_fraction_stays_near_target() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem, Path::new("/tmp"));
        let target = problem.config.volume_fraction;
        for (i, &vf) in result.volume_history.iter().enumerate() {
            assert!((vf - target).abs() < 0.05,
                "iter {}: vol_frac={:.4} too far from target={:.4}", i+1, vf, target);
        }
    }

    #[test]
    fn two_identical_halfweight_loads_match_single() {
        // Aggregation invariant: one load case at weight 1.0 must produce exactly
        // the same optimisation as two identical load cases at weight 0.5 each
        // (0.5*C + 0.5*C == C; 0.5*dc + 0.5*dc == dc). Also guards the k=1
        // single-load path against regression from the multi-load refactor.
        let single = make_problem(4, 3, 2);
        let r1 = run_simp(&single, Path::new("/tmp"));

        let mut dual = make_problem(4, 3, 2);
        let ld = dual.load_cases[0].load_dofs.clone();
        let lv = dual.load_cases[0].load_vals.clone();
        dual.load_cases[0].weight = 0.5;
        dual.load_cases.push(LoadCase {
            name: "dup".to_string(),
            weight: 0.5,
            load_dofs: ld,
            load_vals: lv,
            fixed_dofs: None,
        });
        let r2 = run_simp(&dual, Path::new("/tmp"));

        assert_eq!(r1.compliance_history.len(), r2.compliance_history.len());
        let c1 = *r1.compliance_history.last().unwrap();
        let c2 = *r2.compliance_history.last().unwrap();
        assert!((c1 - c2).abs() <= 1e-9 * c1.abs().max(1e-30),
            "two half-weight loads diverged from single: {c1} vs {c2}");
        assert_eq!(r1.final_density.len(), r2.final_density.len());
        for (a, b) in r1.final_density.iter().zip(r2.final_density.iter()) {
            assert!((a - b).abs() <= 1e-9, "final density diverged: {a} vs {b}");
        }
    }

    #[test]
    fn result_fields_are_populated() {
        let problem = make_problem(4, 3, 2);
        let result = run_simp(&problem, Path::new("/tmp"));
        assert!(result.n_iterations > 0);
        assert!(result.final_compliance > 0.0);
        assert!(!result.compliance_history.is_empty());
        assert_eq!(result.compliance_history.len(), result.volume_history.len());
        assert!(result.duration_s > 0.0);
    }

    #[test]
    fn warm_start_produces_same_final_compliance() {
        let problem = make_problem(4, 3, 2);
        let result1 = run_simp(&problem, Path::new("/tmp"));
        assert!(result1.final_compliance > 0.0);

        let mut problem2 = make_problem(4, 3, 2);
        let n_elem = problem2.grid.n_elem();
        problem2.x_init = Some(vec![problem2.config.volume_fraction; n_elem]);

        let result2 = run_simp(&problem2, Path::new("/tmp"));
        assert!(result2.final_compliance > 0.0);
        assert!(!result2.compliance_history.is_empty());
    }

    #[test]
    fn checkpoint_round_trip() {
        use crate::io::{write_checkpoint, read_checkpoint, delete_checkpoint, CheckpointMeta};
        let tmp = std::env::temp_dir().join(format!("simp_ckpt_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let n = 12usize;
        let density = vec![0.5f64; n];
        let meta = CheckpointMeta {
            iter_completed:     10,
            compliance_history: vec![0.1, 0.09, 0.08],
            volume_history:     vec![0.35, 0.35, 0.35],
            n_elem:             n,
        };

        write_checkpoint(&tmp, &density, &meta).expect("checkpoint write failed");
        let (dens2, meta2) = read_checkpoint(&tmp, n).expect("checkpoint read failed");

        assert_eq!(meta2.iter_completed, 10);
        assert_eq!(meta2.n_elem, n);
        assert_eq!(dens2.len(), n);
        for (&a, &b) in density.iter().zip(dens2.iter()) {
            assert!((a as f32 - b as f32).abs() < 1e-6, "density mismatch: {a} vs {b}");
        }

        delete_checkpoint(&tmp);
        assert!(!tmp.join("checkpoint.bin").exists());
        assert!(!tmp.join("checkpoint_meta.json").exists());

        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn checkpoint_stale_n_elem_rejected() {
        use crate::io::{write_checkpoint, read_checkpoint, CheckpointMeta};
        let tmp = std::env::temp_dir().join(format!("simp_ckpt_stale_{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        let meta = CheckpointMeta {
            iter_completed: 5, compliance_history: vec![0.1],
            volume_history: vec![0.35], n_elem: 100,
        };
        write_checkpoint(&tmp, &vec![0.5f64; 100], &meta).unwrap();

        // Ask for n_elem=200 — should be rejected
        let result = read_checkpoint(&tmp, 200);
        assert!(result.is_none(), "stale checkpoint should be rejected");

        std::fs::remove_dir_all(&tmp).ok();
    }
}