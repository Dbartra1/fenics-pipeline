## Cargo.toml patch
## Change:  faer = "0.19"
## To:      faer = { version = "0.19", features = ["sparse"] }

[package]
name    = "simp_solver"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "simp_solver"
path = "src/main.rs"

[dependencies]
serde      = { version = "1",    features = ["derive"] }
serde_json = "1"
faer       = { version = "0.19", features = ["sparse"] }   # CHANGED: added features
rayon      = "1.10"

[profile.release]
opt-level = 3
lto       = true

---

## apply_dirichlet O(n_fixed) → O(1) lookup fix in assembly.rs
##
## Replace the column-zeroing scan in apply_dirichlet with a HashSet lookup.
## Not a correctness bug — just slow at production scale with a large fixed DOF set.
## Paste this replacement for apply_dirichlet() into assembly.rs.

pub fn apply_dirichlet(
    k_vals:     &mut [f64],
    k_rows:     &[usize],
    k_cols:     &[usize],
    fixed_dofs: &[usize],
    diag_value: f64,
) {
    use std::collections::HashSet;
    // O(n_fixed) build — done once per call
    let fixed_set: HashSet<usize> = fixed_dofs.iter().copied().collect();
    let n_dof = k_rows.len() - 1;

    // Zero all entries in fixed rows
    for &dof in fixed_dofs {
        for pos in k_rows[dof]..k_rows[dof + 1] {
            k_vals[pos] = 0.0;
        }
    }

    // Zero all entries in fixed columns — O(nnz) with O(1) lookup per entry
    for row in 0..n_dof {
        for pos in k_rows[row]..k_rows[row + 1] {
            if fixed_set.contains(&k_cols[pos]) {
                k_vals[pos] = 0.0;
            }
        }
    }

    // Place diag_value on diagonal of every fixed DOF
    for &dof in fixed_dofs {
        let row_slice = &k_cols[k_rows[dof]..k_rows[dof + 1]];
        if let Ok(local) = row_slice.binary_search(&dof) {
            k_vals[k_rows[dof] + local] = diag_value;
        }
    }
}

---

## faer 0.19 API verification checklist
## Run these in order after adding the sparse feature:

## 1. cargo build --features sparse 2>&1 | head -40
##    If SparseColMat not found → check module path:
##      use faer::sparse::SparseColMat;          (most likely)
##      use faer_sparse::SparseColMat;           (if faer-sparse is a separate crate)

## 2. Find constructor name:
##    cargo doc --open
##    Search for "SparseColMat" → look for:
##      try_new_from_col_major_sorted             (sorted CSC input)
##      new_from_csc_data_checked
##      try_new_from_triplets                    (slower, use if sorted API not available)

## 3. Find Cholesky path:
##    Search "Cholesky" in faer sparse docs → look for:
##      faer::sparse::linalg::solvers::Cholesky
##      faer::sparse::linalg::cholesky::LdltRef  (if LDLT not full Cholesky)

## 4. If index type i32 fails to implement faer::sparse::Index:
##    Change Vec<i32> → Vec<usize> in cholesky_solve()
##    And change SparseColMat::<i32, f64> → SparseColMat::<usize, f64>

## 5. solve_in_place signature:
##    May need: chol.solve_in_place(rhs.as_mut())
##    Or:       chol.solve(rhs.as_ref(), dest.as_mut())
##    Check the Cholesky impl's method list in docs.
