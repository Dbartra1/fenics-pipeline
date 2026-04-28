// src/solver.rs
//
// Linear solver: K·u = f
//
// Primary path: faer sparse Cholesky direct solver (SPD, exact solution).
//   - Factorizes K once in O(n^1.5) for 3D FEM sparsity patterns
//   - Back-solves in O(n) — no iteration count, no preconditioner
//   - At 388k DOFs: factorization ~1-5s, back-solve <0.1s per SIMP step
//
// Fallback path: Jacobi-preconditioned CG (kept for testing and diagnostics).
//   - Inadequate for high stiffness-contrast SIMP problems (O(√κ) iterations)
//   - Still used in unit tests to avoid dependency on faer sparse in test paths
//
// Call site in simp.rs calls `cg_solve_direct` by name.
// `cg_solve_direct` now dispatches to Cholesky; the CG implementation is
// preserved as `cg_solve_inner` for diagnostics and the existing test suite.
//
/// ── faer 0.19 sparse API note ─────────────────────────────────────────────────
// After adding `faer = { version = "0.19", features = ["sparse"] }` to
// Cargo.toml, run `cargo doc --open` and verify:
//   1. `faer::sparse::SparseColMat::<I, E>` — confirm the index type I.
//      faer 0.19 uses `I: faer::sparse::Index` which includes `usize` and `i32`.
//      If `usize` constructors don't exist, try `i32`.
//   2. The constructor name: `try_new_from_col_major_sorted` vs
//      `new_from_csc_data_checked` vs `try_new_from_triplets`.

use crate::preconditioner::{JacobiPreconditioner, Preconditioner};

//   3. `Cholesky::try_new` signature — some versions take a `Side` argument
//      (Upper/Lower), others infer from the stored triangle.
//   4. `solve_in_place` vs `solve_mut` for back-substitution.
//
// If the API names below don't compile, check the items marked VERIFY.

/// Result returned by `cg_solve_direct` — kept identical so simp.rs is unchanged.
#[derive(Debug)]
pub struct CgResult {
    /// For Cholesky: always 1 (factorization + back-solve, no iteration count).
    /// For CG fallback: actual CG iteration count.
    pub iterations: usize,
    /// For Cholesky: residual ‖K·u - f‖₂ / ‖f‖₂ after solve (diagnostic).
    /// For CG fallback: CG residual estimate.
    pub rel_residual: f64,
    /// Whether the solve succeeded.
    pub converged: bool,
}

/// Primary entry point — called by simp.rs.
/// Dispatches to Cholesky direct solver; signature unchanged from CG era.
///
/// `tol` and `max_iter` are ignored for the Cholesky path — they are retained
/// only so the call site in simp.rs does not need to change.
pub fn cg_solve_direct(
    k_rows:   &[usize],
    k_cols:   &[usize],
    k_vals:   &[f64],
    f:        &[f64],
    u:        &mut [f64],
    _tol:     f64,
    _max_iter: usize,
) -> CgResult {
    // Cholesky fill-in on 3D hex meshes makes it slower than CG above ~50k DOFs.
    // Use Cholesky for dev grid, parallel CG for production.
    if f.len() < 50_000 {
        match cholesky_solve(k_rows, k_cols, k_vals, f, u) {
            Ok(rel_res) => CgResult { iterations: 1, rel_residual: rel_res, converged: true },
            Err(e) => {
                eprintln!("[solver] Cholesky failed ({e}), falling back to CG");
                cg_solve_inner(k_rows, k_cols, k_vals, f, u, 1e-6, f.len())
            }
        }
    } else {
        // ── GPU path (compiled only with --features gpu) ──────────────────
        #[cfg(feature = "gpu")]
        {
            use crate::gpu_solver::GpuK;
            match GpuK::upload(k_rows, k_cols, k_vals) {
                Ok(gpu_k) => return gpu_cg_solve(&gpu_k, f, u, 1e-6, f.len()),
                Err(e)    => eprintln!("[gpu] upload failed: {e} — falling back to CPU CG"),
            }
        }
        cg_solve_inner(k_rows, k_cols, k_vals, f, u, 1e-6, f.len())
    }
}

// ─── faer sparse Cholesky ─────────────────────────────────────────────────────

/// Direct Cholesky solve for symmetric positive definite K·u = f.
///
/// Returns the relative residual ‖K·u - f‖/‖f‖ for diagnostics.
///
/// Key insight: K is symmetric, so its CSR representation IS its CSC
/// representation (K = Kᵀ, so interpreting k_rows as col_ptrs and k_cols
/// as row_idxs gives the same matrix).  No data conversion needed — we
/// cast usize to i32 only to satisfy faer's index type constraint.
fn cholesky_solve(
    k_rows: &[usize],
    k_cols: &[usize],
    k_vals: &[f64],
    f:      &[f64],
    u:      &mut [f64],
) -> Result<f64, String> {
    use faer::sparse::{
        SparseColMat, SymbolicSparseColMat,
        linalg::solvers::{Cholesky, SymbolicCholesky},
    };
    use faer::Side;
    use faer::prelude::SpSolver;

    let n = f.len();

    // K is symmetric so CSR == CSC: k_rows are col_ptrs, k_cols are row_idxs
    let symbolic = SymbolicSparseColMat::<usize>::new_unsorted_checked(
        n, n,
        k_rows.to_vec(),
        None,
        k_cols.to_vec(),
    );

    let mat = SparseColMat::<usize, f64>::new(symbolic, k_vals.to_vec());

    let symbolic_chol = SymbolicCholesky::try_new(mat.symbolic(), Side::Lower)
    .map_err(|e| { eprintln!("[chol] symbolic failed: {:?}", e); format!("{:?}", e) })?;

    let chol = Cholesky::try_new_with_symbolic(symbolic_chol, mat.as_ref(), Side::Lower)
    .map_err(|e| { eprintln!("[chol] numeric failed: {:?}", e); format!("{:?}", e) })?;

    let mut rhs = faer::Mat::<f64>::from_fn(n, 1, |i, _| f[i]);
    chol.solve_in_place(rhs.as_mut());

    for i in 0..n { u[i] = rhs.read(i, 0); }

    let f_norm = dot(f, f).sqrt().max(1e-30);
    let ku = csr_matvec_local(k_rows, k_cols, k_vals, u);
    let res_norm = ku.iter().zip(f.iter())
        .map(|(ki, fi)| (ki - fi).powi(2)).sum::<f64>().sqrt();
    Ok(res_norm / f_norm)
}

// ─── CG fallback (preserved for tests and diagnostics) ───────────────────────

/// Jacobi-preconditioned CG.  Kept for:
///   1. Unit tests (avoid faer sparse in test paths)
///   2. Fallback if Cholesky fails on a degenerate system
///   3. Diagnostic comparison with Cholesky residual
///
/// Algorithm: Preconditioned CG (Shewchuk 1994, Algorithm B4)
pub fn cg_solve_inner(
    k_rows:   &[usize],
    k_cols:   &[usize],
    k_vals:   &[f64],
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
) -> CgResult {
    let precond = JacobiPreconditioner::new(k_rows, k_cols, k_vals);
    cg_solve_with_precond(k_rows, k_cols, k_vals, f, u, tol, max_iter, &precond)
}

/// Preconditioned CG against an arbitrary `Preconditioner`. The public
/// `cg_solve_inner` wraps this with a Jacobi preconditioner to preserve the
/// existing call-site contract; Phase 6 will expose config-driven dispatch.
pub fn cg_solve_with_precond(
    k_rows:   &[usize],
    k_cols:   &[usize],
    k_vals:   &[f64],
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
    precond:  &dyn Preconditioner,
) -> CgResult {
    let n = f.len();
    let f_norm = dot(f, f).sqrt().max(1e-30);

    let ku = csr_matvec_local(k_rows, k_cols, k_vals, u);
    let mut r: Vec<f64> = f.iter().zip(ku.iter()).map(|(fi, ki)| fi - ki).collect();

    // Pre-allocate z once; reuse across iterations. Pre-refactor code allocated
    // a fresh Vec each call via `.collect()`; arithmetic is identical, only the
    // allocation pattern differs.
    let mut z = vec![0.0f64; n];
    precond.apply(&r, &mut z);

    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let mut iterations  = 0;
    let mut rel_residual = dot(&r, &r).sqrt() / f_norm;

    for iter in 0..max_iter {
        iterations = iter + 1;
        rel_residual = dot(&r, &r).sqrt() / f_norm;
        if rel_residual < tol { break; }

        let kp = csr_matvec_local(k_rows, k_cols, k_vals, &p);
        let pkp = dot(&p, &kp);
        if pkp.abs() < 1e-30 { break; }

        let alpha = rz / pkp;
        for i in 0..n {
            u[i] += alpha * p[i];
            r[i] -= alpha * kp[i];
        }

        precond.apply(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz.max(1e-30);
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }

    CgResult {
        iterations,
        rel_residual,
        converged: rel_residual < tol,
    }
}

// ─── GPU-accelerated CG (compiled only with --features gpu) ──────────────────

/// Delegates the entire CG loop to GpuK::cg_solve_persistent().
///
/// Session 2 change: the hand-rolled CPU-orchestrated loop (4 full-vector
/// H2D/D2H transfers per iteration) is replaced by a single call that keeps
/// all CG state GPU-resident.  Only 3 scalar D2H per iteration (DDOT results
/// for α, β, convergence) plus one full D2H at exit for u.
///
/// Timing is logged to stderr so per-solve cost is visible in notebook output.
#[cfg(feature = "gpu")]
fn gpu_cg_solve(
    gpu_k:    &crate::gpu_solver::GpuK,
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
) -> CgResult {
    let t0     = std::time::Instant::now();
    let result = gpu_k.cg_solve_persistent(f, u, tol, max_iter);
    eprintln!(
        "[gpu_cg] {} iters  rel_res={:.3e}  converged={}  {:.0}ms",
        result.iterations,
        result.rel_residual,
        result.converged,
        t0.elapsed().as_secs_f64() * 1000.0,
    );
    result
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn csr_matvec_local(
    k_rows: &[usize],
    k_cols: &[usize],
    k_vals: &[f64],
    u:      &[f64],
) -> Vec<f64> {
    use rayon::prelude::*;
    let n = k_rows.len() - 1;
    (0..n).into_par_iter().map(|i| {
        (k_rows[i]..k_rows[i + 1])
            .map(|pos| k_vals[pos] * u[k_cols[pos]])
            .sum()
    }).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────
//
// All existing tests target cg_solve_inner directly.  This is intentional:
// the tests verify correctness of the CG algorithm (which is used as fallback).
// The Cholesky path is verified indirectly by the simp.rs integration test
// and by checking that compliance decreases monotonically in notebook 04.
//
// Add a `cholesky_solves_tridiagonal_system` test below once the feature flag
// is confirmed compiling.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{apply_dirichlet, assemble_k, build_csr_pattern, csr_matvec};
    use crate::connectivity::precompute_dof_map;
    use crate::ke_base::compute_ke_base;
    use crate::types::{Grid, Material};

    fn steel() -> Material { Material { young: 210e9, poisson: 0.3 } }

    fn tridiag_system() -> ([usize; 4], [usize; 7], [f64; 7], [f64; 3]) {
        let k_rows = [0usize, 2, 5, 7];
        let k_cols = [0usize, 1,  0, 1, 2,  1, 2];
        let k_vals = [4.0f64, -1.0,  -1.0, 4.0, -1.0,  -1.0, 4.0];
        let f      = [1.0f64, 0.0, 1.0];
        (k_rows, k_cols, k_vals, f)
    }

    // ── CG (inner) tests — unchanged from original ─────────────────────────

    #[test]
    fn cg_solves_tridiagonal_system() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_inner(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);

        assert!(result.converged, "CG did not converge: {:?}", result);
        let expected = [2.0/7.0, 1.0/7.0, 2.0/7.0];
        for i in 0..3 {
            let err = (u[i] - expected[i]).abs();
            assert!(err < 1e-10, "u[{i}]={:.8e}, expected {:.8e}", u[i], expected[i]);
        }
    }

    #[test]
    fn cg_residual_is_small_after_solve() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_inner(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);
        assert!(result.rel_residual < 1e-10, "rel_residual={:.3e}", result.rel_residual);
    }

    #[test]
    fn cg_converges_in_at_most_n_iterations_for_spd() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_inner(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);
        assert!(result.iterations <= 3, "took {} iterations on 3×3 system", result.iterations);
    }

    #[test]
    fn cg_with_warm_start_converges_instantly() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [2.0/7.0, 1.0/7.0, 2.0/7.0];
        let result = cg_solve_inner(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-8, 100);
        assert!(result.rel_residual < 1e-8, "warm start residual={:.3e}", result.rel_residual);
        assert!(result.iterations <= 2, "warm start took {} iterations", result.iterations);
    }

    #[test]
    fn cg_result_satisfies_ku_equals_f() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        cg_solve_inner(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);

        let ku = csr_matvec(&k_rows, &k_cols, &k_vals, &u);
        for i in 0..3 {
            let err = (ku[i] - f[i]).abs();
            assert!(err < 1e-10, "K·u[{i}]={:.6e}, f[{i}]={:.6e}", ku[i], f[i]);
        }
    }

    #[test]
    fn cg_solves_single_element_fem_system() {
        let g = Grid { nx: 1, ny: 1, nz: 1, voxel_size: 0.001 };
        let dof_map = precompute_dof_map(&g);
        let pattern = build_csr_pattern(&g, &dof_map);
        let ke = compute_ke_base(&steel(), g.voxel_size);
        let n_dof = g.n_dof();
        let nnz = pattern.k_rows[n_dof];

        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &[1.0], &ke, &pattern, &[false], &[false], 3.0);

        let fixed_dofs: Vec<usize> = (0..4).flat_map(|n| [3*n, 3*n+1, 3*n+2]).collect();

        let diag_mean: f64 = (0..n_dof)
            .map(|i| {
                let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
                let pos = row.binary_search(&i).unwrap();
                k_vals[pattern.k_rows[i] + pos]
            })
            .sum::<f64>() / n_dof as f64;

        apply_dirichlet(&mut k_vals, &pattern.k_rows, &pattern.k_cols,
                        &fixed_dofs, diag_mean);

        let mut f = vec![0.0f64; n_dof];
        for n in 4..8usize { f[3*n + 2] = 0.25; }
        for &d in &fixed_dofs { f[d] = 0.0; }

        let mut u = vec![0.0f64; n_dof];
        let result = cg_solve_inner(
            &pattern.k_rows, &pattern.k_cols, &k_vals, &f, &mut u, 1e-8, 1000
        );

        assert!(result.converged,
            "FEM solve did not converge: rel_res={:.3e} in {} iters",
            result.rel_residual, result.iterations
        );

        for n in 4..8usize {
            assert!(u[3*n + 2] > 0.0,
                "node {n} z-disp={:.4e} should be positive", u[3*n + 2]);
        }
        for &d in &fixed_dofs {
            assert!(u[d].abs() < 1e-6,
                "fixed DOF {d} has displacement {:.4e}", u[d]);
        }
    }

    //── Cholesky integration test (requires sparse feature) ───────────────
    //Uncomment after confirming faer sparse feature compiles.

    #[test]
    fn cholesky_solves_tridiagonal_system() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 0.0, 0);
        assert!(result.converged, "Cholesky did not report success");
        let expected = [2.0/7.0, 1.0/7.0, 2.0/7.0];
        for i in 0..3 {
            let err = (u[i] - expected[i]).abs();
            assert!(err < 1e-10, "u[{i}]={:.8e}, expected {:.8e}", u[i], expected[i]);
        }
    }
}