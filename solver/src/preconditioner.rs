//! Preconditioner trait and implementations for the CPU CG path.
//!
//! Tier 4 Phase 1 — trait scaffold. This module introduces the abstraction
//! that subsequent phases will populate with a geometric-multigrid (GMG)
//! preconditioner. The Jacobi implementation here is a mechanical lift of
//! the diagonal-inverse logic previously inlined in `solver::cg_solve_inner`
//! and produces bit-identical arithmetic to the pre-refactor code.
//!
//! Scope note: this trait is CPU-host-slice only. The GPU path in
//! `gpu_solver.rs` has its own device-memory preconditioner abstraction
//! (`GpuK::precondition`) and does not go behind this trait — the host/device
//! slice type mismatch would force a lossy unification that loses more than
//! it gains. Keep the boundary at the CPU/GPU dispatch in `cg_solve_direct`.

/// Applies M⁻¹ to a residual vector. Implementations must be stateless with
/// respect to the input vectors — they may hold precomputed data from
/// construction (e.g. diagonal entries, multigrid hierarchy) but must not
/// mutate that data during `apply`.
///
/// `Send + Sync` so rayon-parallelized CG can hold a `&dyn Preconditioner`.
/// `apply` writes into a caller-supplied buffer to avoid per-iter allocation;
/// the buffer is assumed to be the same length as `r`.
pub trait Preconditioner: Send + Sync {
    /// Compute `z = M⁻¹ · r`. Must not allocate.
    ///
    /// Panics if `r.len() != z.len()`.
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// Diagonal (Jacobi) preconditioner. M = diag(A); M⁻¹ r = r ⊘ diag(A).
///
/// Extracts the diagonal from a CSR matrix at construction. Rows with no
/// diagonal entry or a near-zero diagonal fall back to 1.0 — matches the
/// pre-refactor guard at `if d.abs() > 1e-30`.
pub struct JacobiPreconditioner {
    diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Construct from CSR arrays. The diagonal is extracted via binary search
    /// within each row, matching the `cg_solve_inner` setup verbatim.
    pub fn new(k_rows: &[usize], k_cols: &[usize], k_vals: &[f64]) -> Self {
        let n = k_rows.len().saturating_sub(1);
        let mut diag = vec![1.0f64; n];
        for i in 0..n {
            let row = &k_cols[k_rows[i]..k_rows[i + 1]];
            if let Ok(local) = row.binary_search(&i) {
                let d = k_vals[k_rows[i] + local];
                if d.abs() > 1e-30 {
                    diag[i] = d;
                }
            }
        }
        Self { diag }
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        debug_assert_eq!(r.len(), z.len());
        debug_assert_eq!(r.len(), self.diag.len());
        for i in 0..r.len() {
            z[i] = r[i] / self.diag[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CSR representation of diag(2, 4, 8) — trivial SPD 3×3.
    fn diag_csr_3x3() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let k_rows = vec![0, 1, 2, 3];
        let k_cols = vec![0, 1, 2];
        let k_vals = vec![2.0, 4.0, 8.0];
        (k_rows, k_cols, k_vals)
    }

    #[test]
    fn jacobi_apply_diagonal_system() {
        let (k_rows, k_cols, k_vals) = diag_csr_3x3();
        let pc = JacobiPreconditioner::new(&k_rows, &k_cols, &k_vals);

        let r = vec![4.0, 12.0, 32.0];
        let mut z = vec![0.0; 3];
        pc.apply(&r, &mut z);

        // z = r ⊘ diag = [4/2, 12/4, 32/8] = [2, 3, 4]
        assert_eq!(z, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn jacobi_falls_back_to_one_for_missing_diagonal() {
        // 2×2 with no diagonal entry in row 1 — only off-diagonal.
        let k_rows = vec![0, 1, 2];
        let k_cols = vec![0, 0];
        let k_vals = vec![5.0, 7.0];
        let pc = JacobiPreconditioner::new(&k_rows, &k_cols, &k_vals);

        let r = vec![10.0, 3.0];
        let mut z = vec![0.0; 2];
        pc.apply(&r, &mut z);

        // Row 0: diag = 5, z = 10/5 = 2.
        // Row 1: no diagonal, falls back to 1, z = 3/1 = 3.
        assert_eq!(z, vec![2.0, 3.0]);
    }

    #[test]
    fn jacobi_falls_back_to_one_for_near_zero_diagonal() {
        // Diagonal entry present but below 1e-30 threshold.
        let k_rows = vec![0, 1, 2];
        let k_cols = vec![0, 1];
        let k_vals = vec![3.0, 1e-40];
        let pc = JacobiPreconditioner::new(&k_rows, &k_cols, &k_vals);

        let r = vec![9.0, 5.0];
        let mut z = vec![0.0; 2];
        pc.apply(&r, &mut z);

        assert_eq!(z, vec![3.0, 5.0]);
    }

    #[test]
    fn jacobi_bit_identical_to_inline_division() {
        // Construct an arbitrary SPD diagonal and confirm the trait method
        // produces bit-identical output to the idiom currently inlined in
        // cg_solve_inner: `r.iter().zip(diag.iter()).map(|(ri, di)| ri / di)`.
        let k_rows = vec![0, 1, 2, 3, 4];
        let k_cols = vec![0, 1, 2, 3];
        let k_vals = vec![1.7, 2.3, 0.9, 4.1];
        let pc = JacobiPreconditioner::new(&k_rows, &k_cols, &k_vals);

        let r = vec![0.314159, -2.71828, 1.41421, -0.57721];
        let mut z_trait = vec![0.0; 4];
        pc.apply(&r, &mut z_trait);

        let z_inline: Vec<f64> =
            r.iter().zip(k_vals.iter()).map(|(ri, di)| ri / di).collect();

        // Bit-identical: same operands, same order, same op.
        assert_eq!(z_trait, z_inline);
    }
}