// src/assembly.rs
//
// Global stiffness matrix K assembly for the structured hex grid.
//
// K is stored in CSR (Compressed Sparse Row) format:
//   k_rows[i]            — index into k_cols/k_vals where row i starts
//   k_cols[k_rows[i]..k_rows[i+1]] — column indices for row i (sorted)
//   k_vals[k_rows[i]..k_rows[i+1]] — values for row i (zeroed and rebuilt each iter)
//
// Key design: the sparsity pattern (k_rows, k_cols) never changes between
// iterations — only k_vals is updated. We precompute elem_to_csr[e][i*24+j]
// which maps each of the 576 local (i,j) entry positions of element e to its
// flat index in k_vals. This makes assembly O(n_elem * 576) with no searching.
//
// DOF elimination for Dirichlet BCs is handled by zeroing the row/column and
// placing 1.0 on the diagonal (the standard "penalty diagonal" approach that
// preserves matrix structure without reordering).

use crate::types::Grid;

// ─── CSR pattern ─────────────────────────────────────────────────────────────

/// The immutable sparsity structure of K. Built once, reused every iteration.
pub struct CsrPattern {
    /// Length n_dof + 1. k_rows[i+1] - k_rows[i] = number of nonzeros in row i.
    pub k_rows: Vec<usize>,
    /// Length nnz. Column index for each stored entry.
    pub k_cols: Vec<usize>,
    /// For each element e: flat index into k_vals for each of its 576 (i,j) pairs.
    /// elem_to_csr[e][i*24+j] = position in k_vals of the entry K[dof_i][dof_j].
    pub elem_to_csr: Vec<[usize; 576]>,
}

/// Build the CSR sparsity pattern from the DOF map.
///
/// Algorithm:
///   1. For each element, mark all (dof_i, dof_j) pairs as coupled.
///   2. Sort and deduplicate column indices per row.
///   3. For each element, binary-search each (dof_i, dof_j) to find its
///      position in k_cols — stored in elem_to_csr for O(1) assembly later.
///
/// This binary search happens once here, not once per iteration. That's the
/// entire point of elem_to_csr.
pub fn build_csr_pattern(grid: &Grid, dof_map: &[[usize; 24]]) -> CsrPattern {
    let n_dof = grid.n_dof();
    let n_elem = grid.n_elem();

    // ── Step 1: collect all (row, col) pairs ─────────────────────────────────
    // Use a Vec<Vec<usize>> (row → list of col indices, with duplicates).
    let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); n_dof];
    for e in 0..n_elem {
        for &di in &dof_map[e] {
            for &dj in &dof_map[e] {
                row_cols[di].push(dj);
            }
        }
    }

    // ── Step 2: sort and deduplicate each row ─────────────────────────────────
    for row in &mut row_cols {
        row.sort_unstable();
        row.dedup();
    }

    // ── Step 3: build k_rows and k_cols ──────────────────────────────────────
    let mut k_rows = vec![0usize; n_dof + 1];
    for i in 0..n_dof {
        k_rows[i + 1] = k_rows[i] + row_cols[i].len();
    }
    let nnz = k_rows[n_dof];
    let mut k_cols = vec![0usize; nnz];
    for i in 0..n_dof {
        k_cols[k_rows[i]..k_rows[i + 1]].copy_from_slice(&row_cols[i]);
    }

    // ── Step 4: build elem_to_csr ─────────────────────────────────────────────
    // For each element e and each (i,j) local pair, binary-search the sorted
    // k_cols slice for row dof_i to find where dof_j sits.
    let mut elem_to_csr: Vec<[usize; 576]> = vec![[0usize; 576]; n_elem];
    for e in 0..n_elem {
        let dofs = &dof_map[e];
        for i in 0..24 {
            let row = dofs[i];
            let row_start = k_rows[row];
            let row_slice = &k_cols[k_rows[row]..k_rows[row + 1]];
            for j in 0..24 {
                let col = dofs[j];
                // Binary search within the sorted row slice
                let local_pos = row_slice
                    .binary_search(&col)
                    .expect("DOF pair missing from pattern — this is a bug in step 1");
                elem_to_csr[e][i * 24 + j] = row_start + local_pos;
            }
        }
    }

    CsrPattern { k_rows, k_cols, elem_to_csr }
}

// ─── K assembly ──────────────────────────────────────────────────────────────

/// Assemble the global stiffness matrix values into k_vals.
///
/// k_vals must be pre-allocated to nnz = k_rows[n_dof] entries.
/// It is zeroed at the start of this function — call once per iteration.
///
/// void_mask[e]=true  → element contributes nothing (always void)
/// nondesign[e]=true  → element uses ρ=1 regardless of x[e]
pub fn assemble_k(
    k_vals:     &mut Vec<f64>,
    x:          &[f64],
    ke_base:    &[f64; 576],
    pattern:    &CsrPattern,
    void_mask:  &[bool],
    nondesign:  &[bool],
    penal:      f64,
) {
    // Zero k_vals — cheaper than re-allocating
    k_vals.iter_mut().for_each(|v| *v = 0.0);

    for e in 0..x.len() {
        if void_mask[e] { continue; }
        let rho   = if nondesign[e] { 1.0 } else { x[e] };
        let scale = rho.powf(penal);
        for i in 0..24 {
            for j in 0..24 {
                let pos = pattern.elem_to_csr[e][i * 24 + j];
                k_vals[pos] += scale * ke_base[i * 24 + j];
            }
        }
    }
}

/// Apply Dirichlet boundary conditions by zeroing rows/columns and setting
/// the diagonal to the mean diagonal value (preserves conditioning).
///
/// After this call, K·u = f with u[fixed_dofs] = 0 is solvable without
/// reordering. The force vector f must also have f[fixed_dofs] zeroed by
/// the caller (done once in solver setup).
pub fn apply_dirichlet(
    k_vals:     &mut [f64],
    k_rows:     &[usize],
    k_cols:     &[usize],
    fixed_dofs: &[usize],
    diag_value: f64,
) {
    let n_dof = k_rows.len() - 1;

    // Zero all entries in fixed rows
    for &dof in fixed_dofs {
        for pos in k_rows[dof]..k_rows[dof + 1] {
            k_vals[pos] = 0.0;
        }
    }

    // Zero all entries in fixed columns (requires a scan — unavoidable in CSR)
    // and set diagonal to diag_value
    for row in 0..n_dof {
        for pos in k_rows[row]..k_rows[row + 1] {
            let col = k_cols[pos];
            if fixed_dofs.contains(&col) {
                k_vals[pos] = 0.0;
            }
        }
    }

    // Place diag_value on the diagonal of every fixed DOF
    for &dof in fixed_dofs {
        let row_slice = &k_cols[k_rows[dof]..k_rows[dof + 1]];
        if let Ok(local) = row_slice.binary_search(&dof) {
            k_vals[k_rows[dof] + local] = diag_value;
        }
    }
}

/// Compute K·u as a dense vector. Used in tests and for residual checks.
pub fn csr_matvec(
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
    use crate::connectivity::precompute_dof_map;
    use crate::ke_base::compute_ke_base;
    use crate::types::{Grid, Material};

    fn small_grid() -> Grid {
        Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 }
    }

    fn steel() -> Material {
        Material { young: 210e9, poisson: 0.3 }
    }

    fn setup(grid: &Grid) -> (Vec<[usize; 24]>, CsrPattern, [f64; 576]) {
        let dof_map = precompute_dof_map(grid);
        let pattern = build_csr_pattern(grid, &dof_map);
        let ke      = compute_ke_base(&steel(), grid.voxel_size);
        (dof_map, pattern, ke)
    }

    fn full_k(grid: &Grid) -> (CsrPattern, Vec<f64>) {
        let (_, pattern, ke) = setup(grid);
        let nnz = pattern.k_rows[grid.n_dof()];
        let mut k_vals = vec![0.0f64; nnz];
        let n = grid.n_elem();
        assemble_k(
            &mut k_vals, &vec![1.0f64; n], &ke, &pattern,
            &vec![false; n], &vec![false; n], 3.0,
        );
        (pattern, k_vals)
    }

    // ── CSR structure ─────────────────────────────────────────────────────────

    #[test]
    fn k_rows_length_is_n_dof_plus_one() {
        let g = small_grid();
        let (_, pattern, _) = setup(&g);
        assert_eq!(pattern.k_rows.len(), g.n_dof() + 1);
    }

    #[test]
    fn k_rows_is_monotone_nondecreasing() {
        let g = small_grid();
        let (_, pattern, _) = setup(&g);
        for i in 0..g.n_dof() {
            assert!(pattern.k_rows[i] <= pattern.k_rows[i + 1]);
        }
    }

    #[test]
    fn k_cols_all_in_valid_range() {
        let g = small_grid();
        let (_, pattern, _) = setup(&g);
        for &c in &pattern.k_cols {
            assert!(c < g.n_dof(), "col {c} >= n_dof {}", g.n_dof());
        }
    }

    #[test]
    fn each_row_cols_sorted_and_unique() {
        let g = small_grid();
        let (_, pattern, _) = setup(&g);
        for i in 0..g.n_dof() {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i + 1]];
            for w in row.windows(2) {
                assert!(w[0] < w[1], "row {i}: cols not strictly sorted: {w:?}");
            }
        }
    }

    #[test]
    fn diagonal_entries_exist_in_every_row() {
        // Every DOF must have a diagonal entry (required for CG convergence).
        let g = small_grid();
        let (_, pattern, _) = setup(&g);
        for i in 0..g.n_dof() {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i + 1]];
            assert!(
                row.binary_search(&i).is_ok(),
                "row {i} has no diagonal entry"
            );
        }
    }

    #[test]
    fn elem_to_csr_maps_to_correct_column() {
        // For each element, verify that k_cols[elem_to_csr[e][i*24+j]] == dof_map[e][j].
        let g = small_grid();
        let dof_map = precompute_dof_map(&g);
        let pattern = build_csr_pattern(&g, &dof_map);
        for e in 0..g.n_elem() {
            for i in 0..24 {
                for j in 0..24 {
                    let pos = pattern.elem_to_csr[e][i * 24 + j];
                    let expected_col = dof_map[e][j];
                    assert_eq!(
                        pattern.k_cols[pos], expected_col,
                        "elem {e} ({i},{j}): k_cols[{pos}]={} != dof_j={}",
                        pattern.k_cols[pos], expected_col
                    );
                }
            }
        }
    }

    // ── K assembly ────────────────────────────────────────────────────────────

    #[test]
    fn k_is_symmetric_at_unit_density() {
        let g = small_grid();
        let (pattern, k_vals) = full_k(&g);
        let n = g.n_dof();

        // Build dense copy for symmetric check
        let mut dense = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for pos in pattern.k_rows[i]..pattern.k_rows[i + 1] {
                dense[i][pattern.k_cols[pos]] = k_vals[pos];
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (dense[i][j] - dense[j][i]).abs();
                let scale = dense[i][i].abs().max(dense[j][j].abs()).max(1.0);
                assert!(
                    diff / scale < 1e-10,
                    "K[{i}][{j}]={:.4e} != K[{j}][{i}]={:.4e}",
                    dense[i][j], dense[j][i]
                );
            }
        }
    }

    #[test]
    fn k_diagonal_all_positive_at_unit_density() {
        let g = small_grid();
        let (pattern, k_vals) = full_k(&g);
        for i in 0..g.n_dof() {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i + 1]];
            let diag_pos = pattern.k_rows[i] + row.binary_search(&i).unwrap();
            assert!(k_vals[diag_pos] > 0.0, "K[{i}][{i}] = {} is not positive", k_vals[diag_pos]);
        }
    }

    #[test]
    fn rigid_body_translation_gives_zero_force() {
        // K · [1,0,0, 1,0,0, ...] should be zero for a free (unconstrained) mesh.
        let g = small_grid();
        let (pattern, k_vals) = full_k(&g);
        let n = g.n_dof();

        let mut u = vec![0.0f64; n];
        for i in (0..n).step_by(3) { u[i] = 1.0; }   // unit x-displacement everywhere

        let f = csr_matvec(&pattern.k_rows, &pattern.k_cols, &k_vals, &u);
        let k_max = k_vals.iter().cloned().fold(0.0f64, f64::max);
        for (i, &fi) in f.iter().enumerate() {
            assert!(
                fi.abs() < k_max * 1e-9,
                "f[{i}]={fi:.3e} for rigid x-translation (k_max={k_max:.3e})"
            );
        }
    }

    #[test]
    fn void_elements_contribute_nothing() {
        let g = small_grid();
        let (_, pattern, ke) = setup(&g);
        let n_elem = g.n_elem();
        let nnz = pattern.k_rows[g.n_dof()];

        // All solid
        let mut k_solid = vec![0.0f64; nnz];
        assemble_k(&mut k_solid, &vec![1.0; n_elem], &ke, &pattern,
                   &vec![false; n_elem], &vec![false; n_elem], 3.0);

        // Element 0 void
        let mut void_mask = vec![false; n_elem];
        void_mask[0] = true;
        let mut k_void = vec![0.0f64; nnz];
        assemble_k(&mut k_void, &vec![1.0; n_elem], &ke, &pattern,
                   &void_mask, &vec![false; n_elem], 3.0);

        // The two should differ — voiding element 0 changes K
        let diff: f64 = k_solid.iter().zip(k_void.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "voiding element 0 had no effect on K");
    }

    #[test]
    fn nondesign_elements_use_unit_density() {
        // Nondesign elements use ρ=1 regardless of x[e].
        // K with all x=0.5 + nondesign[0]=true should equal
        // K with x=0.5 everywhere + manually setting x[0]=1.
        let g = small_grid();
        let (_, pattern, ke) = setup(&g);
        let n_elem = g.n_elem();
        let nnz = pattern.k_rows[g.n_dof()];

        let mut nondesign = vec![false; n_elem];
        nondesign[0] = true;
        let x = vec![0.5f64; n_elem];

        let mut k_nondesign = vec![0.0f64; nnz];
        assemble_k(&mut k_nondesign, &x, &ke, &pattern,
                   &vec![false; n_elem], &nondesign, 3.0);

        // Reference: x[0]=1.0, rest=0.5, no nondesign
        let mut x_ref = x.clone();
        x_ref[0] = 1.0;
        let mut k_ref = vec![0.0f64; nnz];
        assemble_k(&mut k_ref, &x_ref, &ke, &pattern,
                   &vec![false; n_elem], &vec![false; n_elem], 3.0);

        let diff: f64 = k_nondesign.iter().zip(k_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-6, "nondesign should force ρ=1, diff={diff:.3e}");
    }

    #[test]
    fn penal_scaling_is_correct() {
        // At ρ=0.5, penal=3: scale = 0.5³ = 0.125.
        // K(ρ=0.5) should equal 0.125 * K(ρ=1.0) on a single-element grid.
        let g = Grid { nx: 1, ny: 1, nz: 1, voxel_size: 0.001 };
        let (_, pattern, ke) = setup(&g);
        let nnz = pattern.k_rows[g.n_dof()];

        let mut k1 = vec![0.0f64; nnz];
        assemble_k(&mut k1, &[1.0], &ke, &pattern,
                   &[false], &[false], 3.0);

        let mut k_half = vec![0.0f64; nnz];
        assemble_k(&mut k_half, &[0.5], &ke, &pattern,
                   &[false], &[false], 3.0);

        for i in 0..nnz {
            let expected = 0.125 * k1[i];
            let diff = (k_half[i] - expected).abs();
            let scale = k1[i].abs().max(1e-30);
            assert!(diff / scale < 1e-12, "nnz[{i}]: {:.6e} != 0.125*{:.6e}", k_half[i], k1[i]);
        }
    }
}