// src/solver.rs
//
// Conjugate Gradient linear solver: K·u = f
//
// Jacobi (diagonal) preconditioner: M = diag(K).
// For structured hex grids at the scales we target (up to ~750k DOFs),
// preconditioned CG converges in O(√κ) iterations. With Dirichlet BCs
// applied and a well-penalized density field, κ is well-behaved.

/// Result returned by `cg_solve_direct`.
#[derive(Debug)]
pub struct CgResult {
    /// Number of CG iterations taken.
    pub iterations: usize,
    /// Final residual ‖r‖₂ / ‖f‖₂  (relative).
    pub rel_residual: f64,
    /// Whether the solver converged within `max_iter`.
    pub converged: bool,
}


/// Direct CG implementation using our CSR matvec.
/// Jacobi (diagonal) preconditioner: M = diag(K).
///
/// Algorithm: Preconditioned CG (Shewchuk 1994, Algorithm B4)
///   r = f - K·u
///   z = M⁻¹·r
///   p = z
///   loop:
///     α = (r·z) / (p·K·p)
///     u += α·p
///     r -= α·K·p
///     z_new = M⁻¹·r_new
///     β = (r_new·z_new) / (r·z)
///     p = z_new + β·p
pub fn cg_solve_direct(
    k_rows:   &[usize],
    k_cols:   &[usize],
    k_vals:   &[f64],
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
) -> CgResult {
    let n = f.len();

    // ── Extract diagonal for Jacobi preconditioner ────────────────────────────
    let mut diag = vec![1.0f64; n];
    for i in 0..n {
        let row = &k_cols[k_rows[i]..k_rows[i + 1]];
        if let Ok(local) = row.binary_search(&i) {
            let d = k_vals[k_rows[i] + local];
            if d.abs() > 1e-30 { diag[i] = d; }
        }
    }

    // ── f norm for relative residual ──────────────────────────────────────────
    let f_norm = dot(f, f).sqrt().max(1e-30);

    // ── r = f - K·u (initial residual) ───────────────────────────────────────
    let ku = csr_matvec_local(k_rows, k_cols, k_vals, u);
    let mut r: Vec<f64> = f.iter().zip(ku.iter()).map(|(fi, ki)| fi - ki).collect();

    // ── z = M⁻¹·r ────────────────────────────────────────────────────────────
    let z: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let mut iterations  = 0;
    let mut rel_residual = dot(&r, &r).sqrt() / f_norm;

    for iter in 0..max_iter {
        iterations = iter + 1;
        rel_residual = dot(&r, &r).sqrt() / f_norm;
        if rel_residual < tol { break; }

        // α = (r·z) / (p·K·p)
        let kp = csr_matvec_local(k_rows, k_cols, k_vals, &p);
        let pkp = dot(&p, &kp);
        if pkp.abs() < 1e-30 { break; }   // breakdown
        let alpha = rz / pkp;

        // u += α·p,  r -= α·K·p
        for i in 0..n {
            u[i] += alpha * p[i];
            r[i] -= alpha * kp[i];
        }

        // z_new = M⁻¹·r_new
        let z_new: Vec<f64> = r.iter().zip(diag.iter()).map(|(ri, di)| ri / di).collect();
        let rz_new = dot(&r, &z_new);

        // β = (r_new·z_new) / (r·z)
        let beta = rz_new / rz.max(1e-30);

        // p = z_new + β·p
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
    let n = k_rows.len() - 1;
    let mut f = vec![0.0f64; n];
    for i in 0..n {
        for pos in k_rows[i]..k_rows[i + 1] {
            f[i] += k_vals[pos] * u[k_cols[pos]];
        }
    }
    f
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{apply_dirichlet, assemble_k, build_csr_pattern, csr_matvec};
    use crate::connectivity::precompute_dof_map;
    use crate::ke_base::compute_ke_base;
    use crate::types::{Grid, Material};

    fn steel() -> Material { Material { young: 210e9, poisson: 0.3 } }

    // ── Analytic 3-DOF system ─────────────────────────────────────────────────
    // K = [[4, -1, 0], [-1, 4, -1], [0, -1, 4]]   (tridiagonal SPD)
    // f = [1, 0, 1]
    // Exact solution: u = [5/14, 2/14, 5/14]  (solve by hand)

    fn tridiag_system() -> ([usize; 4], [usize; 7], [f64; 7], [f64; 3]) {
        let k_rows = [0usize, 2, 5, 7];
        let k_cols = [0usize, 1,  0, 1, 2,  1, 2];
        let k_vals = [4.0f64, -1.0,  -1.0, 4.0, -1.0,  -1.0, 4.0];
        let f      = [1.0f64, 0.0, 1.0];
        (k_rows, k_cols, k_vals, f)
    }

    #[test]
    fn cg_solves_tridiagonal_system() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);

        assert!(result.converged, "CG did not converge: {:?}", result);

        // Exact: [2/7, 1/7, 2/7]
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
        let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);
        assert!(result.rel_residual < 1e-10, "rel_residual={:.3e}", result.rel_residual);
    }

    #[test]
    fn cg_converges_in_at_most_n_iterations_for_spd() {
        // Exact CG on n×n SPD system converges in at most n iterations (theory).
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);
        assert!(result.iterations <= 3, "took {} iterations on 3×3 system", result.iterations);
    }

    #[test]
    fn cg_with_warm_start_converges_instantly() {
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [2.0/7.0, 1.0/7.0, 2.0/7.0];   // exact solution
        let result = cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-8, 100);
        assert!(result.rel_residual < 1e-8, "warm start residual={:.3e}", result.rel_residual);
        assert!(result.iterations <= 2, "warm start took {} iterations", result.iterations);
    }

    // ── FEM system: 1-element patch with Dirichlet BCs ───────────────────────

    #[test]
    fn cg_solves_single_element_fem_system() {
        // 1×1×1 element, bottom face fixed, unit z-load on top face.
        let g = Grid { nx: 1, ny: 1, nz: 1, voxel_size: 0.001 };
        let dof_map = precompute_dof_map(&g);
        let pattern = build_csr_pattern(&g, &dof_map);
        let ke = compute_ke_base(&steel(), g.voxel_size);
        let n_dof = g.n_dof();
        let nnz = pattern.k_rows[n_dof];

        // Assemble at unit density
        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &[1.0], &ke, &pattern, &[false], &[false], 3.0);

        // Fix bottom face (z=0): nodes 0,1,2,3 → DOFs 0..11
        let fixed_dofs: Vec<usize> = (0..4).flat_map(|n| [3*n, 3*n+1, 3*n+2]).collect();

        // Find mean diagonal for Dirichlet penalty value
        let diag_mean: f64 = (0..n_dof)
            .map(|i| {
                let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
                let pos = row.binary_search(&i).unwrap();
                k_vals[pattern.k_rows[i] + pos]
            })
            .sum::<f64>() / n_dof as f64;

        apply_dirichlet(&mut k_vals, &pattern.k_rows, &pattern.k_cols,
                        &fixed_dofs, diag_mean);

        // Unit z-load on top face: nodes 4,5,6,7 → z-DOFs 14,17,20,23
        let mut f = vec![0.0f64; n_dof];
        for n in 4..8usize { f[3*n + 2] = 0.25; }   // 1N total, split over 4 nodes
        for &d in &fixed_dofs { f[d] = 0.0; }

        let mut u = vec![0.0f64; n_dof];
        let result = cg_solve_direct(
            &pattern.k_rows, &pattern.k_cols, &k_vals, &f, &mut u, 1e-8, 1000
        );

        assert!(result.converged,
            "FEM solve did not converge: rel_res={:.3e} in {} iters",
            result.rel_residual, result.iterations
        );

        // Top nodes should have positive z-displacement (load is in +z)
        for n in 4..8usize {
            assert!(u[3*n + 2] > 0.0,
                "node {n} z-disp={:.4e} should be positive", u[3*n + 2]);
        }

        // Fixed nodes should have ~zero displacement
        for &d in &fixed_dofs {
            assert!(u[d].abs() < 1e-6,
                "fixed DOF {d} has displacement {:.4e}", u[d]);
        }

        println!("FEM solve: {} iters, rel_res={:.3e}", result.iterations, result.rel_residual);
        println!("Top node z-disp: {:.6e}", u[3*4 + 2]);
    }

    #[test]
    fn cg_result_satisfies_ku_equals_f() {
        // After solving, verify K·u ≈ f directly.
        let (k_rows, k_cols, k_vals, f) = tridiag_system();
        let mut u = [0.0f64; 3];
        cg_solve_direct(&k_rows, &k_cols, &k_vals, &f, &mut u, 1e-12, 100);

        let ku = csr_matvec(&k_rows, &k_cols, &k_vals, &u);
        for i in 0..3 {
            let err = (ku[i] - f[i]).abs();
            assert!(err < 1e-10, "K·u[{i}]={:.6e}, f[{i}]={:.6e}", ku[i], f[i]);
        }
    }
}