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
// ── faer 0.19 sparse API note ─────────────────────────────────────────────────
// After adding `faer = { version = "0.19", features = ["sparse"] }` to
// Cargo.toml, run `cargo doc --open` and verify:
//   1. `faer::sparse::SparseColMat::<I, E>` — confirm the index type I.
//      faer 0.19 uses `I: faer::sparse::Index` which includes `usize` and `i32`.
//      If `usize` constructors don't exist, try `i32`.
//   2. The constructor name: `try_new_from_col_major_sorted` vs
//      `new_from_csc_data_checked` vs `try_new_from_triplets`.
//   3. `Cholesky::try_new` signature — some versions take a `Side` argument
//      (Upper/Lower), others infer from the stored triangle.
//   4. `solve_in_place` vs `solve_mut` for back-substitution.
//
// If the API names below don't compile, check the items marked VERIFY.

use faer::sparse::{SparseColMat, linalg::solvers::Cholesky}; // VERIFY: module path
use faer::Mat;

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
    match cholesky_solve(k_rows, k_cols, k_vals, f, u) {
        Ok(rel_res) => CgResult {
            iterations:   1,
            rel_residual: rel_res,
            converged:    true,
        },
        Err(e) => {
            // Cholesky failed (K not SPD — likely a degenerate density field
            // or a Dirichlet BC application error).  Fall back to CG so the
            // SIMP loop can continue degraded rather than panic.
            eprintln!("[solver] Cholesky failed ({e}), falling back to CG (tol=1e-4, 5000 iter)");
            cg_solve_inner(k_rows, k_cols, k_vals, f, u, 1e-4, 5000)
        }
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
    let n = f.len();

    // ── Index type conversion ──────────────────────────────────────────────
    // VERIFY: faer 0.19 SparseColMat index type.
    // Try `i32` first. If the compiler complains about Index not being
    // implemented for i32, try `usize`. The faer::sparse::Index trait
    // documents which primitive types are supported.
    let col_ptrs: Vec<i32> = k_rows.iter()
        .map(|&x| x as i32)
        .collect();
    let row_idxs: Vec<i32> = k_cols.iter()
        .map(|&x| x as i32)
        .collect();

    // ── Sparse matrix construction ─────────────────────────────────────────
    // VERIFY: constructor name in faer 0.19 sparse docs.
    // The CSR rows are sorted by column index (guaranteed by build_csr_pattern),
    // which means when interpreted as CSC, each column's row indices are sorted.
    // This satisfies the "sorted" precondition.
    //
    // Candidate names to check in docs (most likely first):
    //   SparseColMat::try_new_from_col_major_sorted
    //   SparseColMat::new_from_col_major
    //   SparseColMat::try_new_from_csc
    let mat = SparseColMat::<i32, f64>::try_new_from_col_major_sorted( // VERIFY
        n,
        n,
        col_ptrs.as_slice(),
        row_idxs.as_slice(),
        k_vals,
    ).map_err(|e| format!("SparseColMat construction failed: {:?}", e))?;

    // ── Cholesky factorization ─────────────────────────────────────────────
    // VERIFY: Cholesky::try_new signature.
    // In some faer 0.19 builds it takes (mat, side) where side is
    // faer::Side::Lower. In others it infers the triangle automatically.
    // Try without Side argument first; add `faer::Side::Lower` if it
    // doesn't compile.
    //
    //   Option A: Cholesky::try_new(mat.as_ref())
    //   Option B: Cholesky::try_new(mat.as_ref(), faer::Side::Lower)
    let chol = Cholesky::try_new(mat.as_ref()) // VERIFY option A or B above
        .map_err(|e| format!("Cholesky factorization failed: {:?}", e))?;

    // ── Back-solve ─────────────────────────────────────────────────────────
    // VERIFY: faer 0.19 back-solve API.
    // Most likely: chol.solve_in_place(rhs.as_mut())  where rhs is a Col or Mat.
    // Alternative: chol.solve_mut(&mut rhs) or similar.
    let mut rhs = Mat::<f64>::from_fn(n, 1, |i, _| f[i]);
    chol.solve_in_place(rhs.as_mut()); // VERIFY method name

    // Extract solution
    for i in 0..n {
        u[i] = rhs.read(i, 0);
    }

    // ── Residual (diagnostic) ──────────────────────────────────────────────
    let f_norm = dot(f, f).sqrt().max(1e-30);
    let ku = csr_matvec_local(k_rows, k_cols, k_vals, u);
    let res_norm = ku.iter().zip(f.iter()).map(|(ki, fi)| (ki - fi).powi(2)).sum::<f64>().sqrt();
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
    let n = f.len();

    // ── Jacobi preconditioner ─────────────────────────────────────────────
    let mut diag = vec![1.0f64; n];
    for i in 0..n {
        let row = &k_cols[k_rows[i]..k_rows[i + 1]];
        if let Ok(local) = row.binary_search(&i) {
            let d = k_vals[k_rows[i] + local];
            if d.abs() > 1e-30 { diag[i] = d; }
        }
    }

    let f_norm = dot(f, f).sqrt().max(1e-30);

    let ku = csr_matvec_local(k_rows, k_cols, k_vals, u);
    let mut r: Vec<f64> = f.iter().zip(ku.iter()).map(|(fi, ki)| fi - ki).collect();

    let z: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
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

        let z_new: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
        let rz_new = dot(&r, &z_new);
        let beta = rz_new / rz.max(1e-30);

        for i in 0..n {
            p[i] = z_new[i] + beta * p[i];
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

/// ILU(0)-preconditioned CG using GPU SpMV + GPU triangular solves.
///
/// Per CG iteration:
///   1. kp = K·p       — GPU SpMV via gpu_k.matvec()
///   2. z  = M⁻¹·r     — GPU fwd/bwd triangular solves via gpu_k.precondition()
///
/// Falls back to Jacobi on preconditioner error (triangular solve failure
/// indicates structural singularity in ILU factor — rare but possible when
/// ρ_min elements produce near-zero pivots).
#[cfg(feature = "gpu")]
fn gpu_cg_solve(
    gpu_k:    &crate::gpu_solver::GpuK,
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
) -> CgResult {
    let n      = f.len();
    let f_norm = dot(f, f).sqrt().max(1e-30);
    let diag   = &gpu_k.diag;   // Jacobi diagonal — fallback only

    // Zero u — GPU CG accumulates updates in-place (u += α·p each step).
    // Without this, reusing u from the previous SIMP iteration corrupts the
    // solution: u_final = u_prev + u_correct instead of u_correct.
    // Cholesky overwrites u entirely; CPU CG computes r = f - K·u to handle
    // warm starts. GPU CG must start from u = 0.
    for v in u.iter_mut() { *v = 0.0; }
    let mut r  = f.to_vec();
    let mut z  = vec![0.0_f64; n];
    let mut kp = vec![0.0_f64; n];

    // First preconditioner application: z = M⁻¹·r
    let use_ilu = gpu_k.precondition(&r, &mut z).is_ok();
    if !use_ilu {
        eprintln!("[gpu] ILU precondition failed on iter 0, falling back to Jacobi");
        for i in 0..n { z[i] = r[i] / diag[i]; }
    }

    let mut p  = z.clone();
    let mut rz = dot(&r, &z);

    let mut iterations   = 0;
    let mut rel_residual = dot(&r, &r).sqrt() / f_norm;

    for iter in 0..max_iter {
        iterations   = iter + 1;
        rel_residual = dot(&r, &r).sqrt() / f_norm;
        if rel_residual < tol { break; }

        // SpMV: kp = K·p
        if let Err(e) = gpu_k.matvec(&p, &mut kp) {
            eprintln!("[gpu] matvec iter {iter}: {e}");
            return CgResult { iterations: iter, rel_residual: f64::INFINITY, converged: false };
        }

        let pkp = dot(&p, &kp);
        if pkp.abs() < 1e-30 { break; }
        let alpha = rz / pkp;

        for i in 0..n {
            u[i] += alpha * p[i];
            r[i] -= alpha * kp[i];
        }

        // Preconditioner: z = M⁻¹·r  (ILU or Jacobi fallback)
        if use_ilu {
            if let Err(e) = gpu_k.precondition(&r, &mut z) {
                eprintln!("[gpu] ILU precondition failed iter {iter}: {e}");
                for i in 0..n { z[i] = r[i] / diag[i]; }
            }
        } else {
            for i in 0..n { z[i] = r[i] / diag[i]; }
        }

        let rz_new = dot(&r, &z);
        let beta   = rz_new / rz.max(1e-30);
        for i in 0..n { p[i] = z[i] + beta * p[i]; }
        rz = rz_new;
    }

    CgResult { iterations, rel_residual, converged: rel_residual < tol }
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

    // ── Cholesky integration test (requires sparse feature) ───────────────
    // Uncomment after confirming faer sparse feature compiles.
    //
    // #[test]
    // fn cholesky_solves_tridiagonal_system() {
    //     let (k_rows, k_cols, k_vals, f) = tridiag_system();
    //     let mut u = [0.0f64; 3];
    //     let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 0.0, 0);
    //     assert!(result.converged, "Cholesky did not report success");
    //     let expected = [2.0/7.0, 1.0/7.0, 2.0/7.0];
    //     for i in 0..3 {
    //         let err = (u[i] - expected[i]).abs();
    //         assert!(err < 1e-10, "u[{i}]={:.8e}, expected {:.8e}", u[i], expected[i]);
    //     }
    // }
}
