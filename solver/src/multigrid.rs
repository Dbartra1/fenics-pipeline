// Tier 4 phases 2-4 plant infrastructure that Phase 5 (V-cycle) and Phase 6
// (trait integration) wire into the solver. Until then, all items in this
// module are dead code from the compiler's point of view. Remove this
// attribute once the GMG preconditioner is called from cg_solve_direct.
#![allow(dead_code)]

//! Multigrid components for the GMG preconditioner (Tier 4).
//!
//! Phase 2 lands the block-symmetric red-black Gauss-Seidel smoother.
//! Phase 3 adds restriction/prolongation transfer operators.
//! Phase 4 adds the Galerkin coarse operator hierarchy (this phase).
//! Later phases will add the V-cycle preconditioner (Phases 5-6).
//!
//! # Smoother design — why this shape
//!
//! CG requires a symmetric positive-definite preconditioner. A single GS
//! sweep is not symmetric as a linear operator, so we compose a forward
//! sweep (red then black) with a backward sweep (black then red) within
//! each `smooth` call. The V-cycle calls `smooth` ν times per level.
//!
//! Red-black coloring partitions nodes by spatial parity: a node at grid
//! index (ix, iy, iz) is red if (ix+iy+iz) is even, black otherwise. On a
//! structured hex grid the first-neighbor coupling means every red node's
//! spatial neighbors are all black and vice versa, so all red nodes can be
//! updated in parallel (rayon), then all black nodes in parallel.
//!
//! For elasticity, each node carries three DOFs (ux, uy, uz) that couple
//! through the element stiffness. Pointwise GS on DOFs gives a poor
//! smoothing rate because it ignores the 3×3 node-local coupling. We use
//! **block GS**: each node update solves the 3×3 local system
//!     K_ii · dx_i = r_i     where r_i = b_i - Σ_{j≠i} K_ij · x_j
//! using a pre-inverted K_ii stored at construction.
//!
//! # Contrast robustness
//!
//! This smoother is intended for SIMP problems where adjacent elements can
//! differ in stiffness by 10^6:1 or more. RB-GS propagates contrast
//! information one half-stencil per sweep (using updated neighbor values
//! within the sweep), versus Jacobi's one full stencil per sweep with
//! stale values. In combination with Galerkin coarse operators (Phase 4),
//! this gives near-mesh-independent convergence across the contrast range.

use rayon::prelude::*;

/// Red-black block Gauss-Seidel smoother for 3D elasticity on a structured
/// hex grid.
///
/// Holds references-to-operator-data via cloned CSR arrays. Must be
/// reconstructed when the operator changes (i.e., every SIMP iteration).
pub struct RedBlackGSSmoother {
    // CSR arrays of K (owned copy — the smoother's lifetime is bounded by
    // one SIMP iter, so avoiding a lifetime parameter simplifies the API).
    k_rows: Vec<usize>,
    k_cols: Vec<usize>,
    k_vals: Vec<f64>,

    // Pre-inverted 3×3 diagonal blocks, one per spatial node (not per DOF).
    // Flattened row-major: block for node n is at indices [9*n .. 9*n+9].
    // Rows singular enough to fail inversion get identity as a fallback.
    k_ii_inv: Vec<f64>,

    // Spatial node index lists by color. Each entry is a node index in
    // [0, n_nodes), NOT a DOF index.
    red_nodes:   Vec<usize>,
    black_nodes: Vec<usize>,

    // Dimensions — node counts in each axis. Each DOF index is 3*node + d.
    n_nodes: usize,
    n_dof:   usize,

    // Count of nodes whose 3×3 block was singular. Nonzero is a red flag;
    // checked in tests and logged by the caller in production.
    pub singular_blocks: usize,
}

impl RedBlackGSSmoother {
    /// Build the smoother for a given CSR operator on a structured grid of
    /// (nx+1) × (ny+1) × (nz+1) *nodes* (the element grid is nx × ny × nz,
    /// which matches the convention used elsewhere in the solver).
    pub fn new(
        k_rows: &[usize],
        k_cols: &[usize],
        k_vals: &[f64],
        nx_nodes: usize,
        ny_nodes: usize,
        nz_nodes: usize,
    ) -> Self {
        let n_nodes = nx_nodes * ny_nodes * nz_nodes;
        let n_dof   = 3 * n_nodes;
        assert_eq!(k_rows.len(), n_dof + 1,
            "k_rows length {} incompatible with n_dof {}", k_rows.len(), n_dof);

        // ── Color nodes by spatial parity ─────────────────────────────────
        let mut red_nodes   = Vec::with_capacity(n_nodes / 2 + 1);
        let mut black_nodes = Vec::with_capacity(n_nodes / 2 + 1);
        for iz in 0..nz_nodes {
            for iy in 0..ny_nodes {
                for ix in 0..nx_nodes {
                    let node = iz * (ny_nodes * nx_nodes) + iy * nx_nodes + ix;
                    if (ix + iy + iz) % 2 == 0 {
                        red_nodes.push(node);
                    } else {
                        black_nodes.push(node);
                    }
                }
            }
        }

        // ── Extract and invert 3×3 diagonal blocks ────────────────────────
        let mut k_ii_inv = vec![0.0_f64; 9 * n_nodes];
        let mut singular_blocks = 0_usize;
        for node in 0..n_nodes {
            let mut block = [[0.0_f64; 3]; 3];
            for d_row in 0..3 {
                let dof_row = 3 * node + d_row;
                for k in k_rows[dof_row]..k_rows[dof_row + 1] {
                    let dof_col = k_cols[k];
                    if dof_col / 3 == node {
                        let d_col = dof_col % 3;
                        block[d_row][d_col] = k_vals[k];
                    }
                }
            }

            let block_inv = match invert_3x3(&block) {
                Some(inv) => inv,
                None => {
                    singular_blocks += 1;
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                }
            };

            for d_row in 0..3 {
                for d_col in 0..3 {
                    k_ii_inv[9 * node + 3 * d_row + d_col] = block_inv[d_row][d_col];
                }
            }
        }

        Self {
            k_rows: k_rows.to_vec(),
            k_cols: k_cols.to_vec(),
            k_vals: k_vals.to_vec(),
            k_ii_inv,
            red_nodes,
            black_nodes,
            n_nodes,
            n_dof,
            singular_blocks,
        }
    }

    /// One forward sweep: update all red nodes, then all black nodes.
    /// Within a color, node updates are independent and run in parallel.
    fn forward_sweep(&self, x: &mut [f64], b: &[f64]) {
        self.sweep_color(&self.red_nodes, x, b);
        self.sweep_color(&self.black_nodes, x, b);
    }

    /// One backward sweep: update all black nodes, then all red nodes.
    /// This is the transpose of forward_sweep in the bilinear-form sense,
    /// making the composed operator (forward ∘ backward) symmetric.
    fn backward_sweep(&self, x: &mut [f64], b: &[f64]) {
        self.sweep_color(&self.black_nodes, x, b);
        self.sweep_color(&self.red_nodes, x, b);
    }

    /// One symmetric block-RB-GS smoothing iteration:
    ///   1. forward sweep (R then B)
    ///   2. backward sweep (B then R)
    /// Together these form a symmetric linear operator on (x, b).
    pub fn smooth(&self, x: &mut [f64], b: &[f64]) {
        assert_eq!(x.len(), self.n_dof);
        assert_eq!(b.len(), self.n_dof);
        self.forward_sweep(x, b);
        self.backward_sweep(x, b);
    }

    /// Sweep a single color. Nodes within a color have no RAW dependencies
    /// on each other (their spatial neighbors are all the other color), so
    /// we can compute all new values from a shared snapshot of x, then
    /// write them back.
    ///
    /// Reading x during the sweep while other threads mutate it would be a
    /// data race. We snapshot the reads we need, compute new node blocks
    /// in parallel, then apply the updates serially. The serial write-back
    /// is cache-friendly and negligible compared to the CSR walks.
    fn sweep_color(&self, nodes: &[usize], x: &mut [f64], b: &[f64]) {
        // Collect (node_idx, new_block_values) from parallel computation.
        // `x` is read-only during this phase; we pass &*x to make that
        // explicit to the borrow checker.
        let updates: Vec<(usize, [f64; 3])> = nodes
            .par_iter()
            .map(|&node| {
                let new_x = self.compute_node_update(node, x, b);
                (node, new_x)
            })
            .collect();

        // Serial write-back. Could be parallelized with par_chunks_mut if
        // this ever becomes a bottleneck, but within-color writes target
        // disjoint node indices so the ordering is irrelevant.
        for (node, new_x) in updates {
            x[3 * node]     = new_x[0];
            x[3 * node + 1] = new_x[1];
            x[3 * node + 2] = new_x[2];
        }
    }

    /// Compute the new 3-vector at `node` given the current global `x` and
    /// the RHS `b`. Pure function of inputs — no `x` mutation.
    ///
    ///   r_i  = b_i - Σ_{j≠i} K_ij · x_j      (off-diagonal coupling)
    ///   x_i' = K_ii⁻¹ · r_i                  (local solve)
    fn compute_node_update(&self, node: usize, x: &[f64], b: &[f64]) -> [f64; 3] {
        let mut r = [0.0_f64; 3];
        for d in 0..3 {
            r[d] = b[3 * node + d];
        }

        // Subtract off-diagonal contribution: walk CSR for each of the 3
        // rows of this node, skip entries that land inside our own 3×3
        // block (those belong to the diagonal block, not off-diagonal).
        for d_row in 0..3 {
            let dof_row = 3 * node + d_row;
            for k in self.k_rows[dof_row]..self.k_rows[dof_row + 1] {
                let dof_col = self.k_cols[k];
                if dof_col / 3 != node {
                    r[d_row] -= self.k_vals[k] * x[dof_col];
                }
            }
        }

        // Apply pre-inverted 3×3 diagonal block: x_new = K_ii⁻¹ · r
        let inv = &self.k_ii_inv[9 * node..9 * node + 9];
        [
            inv[0] * r[0] + inv[1] * r[1] + inv[2] * r[2],
            inv[3] * r[0] + inv[4] * r[1] + inv[5] * r[2],
            inv[6] * r[0] + inv[7] * r[1] + inv[8] * r[2],
        ]
    }

    /// Accessor used by tests to verify coloring.
    #[cfg(test)]
    pub fn node_color_counts(&self) -> (usize, usize) {
        (self.red_nodes.len(), self.black_nodes.len())
    }

    /// Accessor used by tests to verify coloring is conflict-free.
    #[cfg(test)]
    pub fn is_red(&self, node: usize) -> bool {
        self.red_nodes.binary_search(&node).is_ok()
    }
}

/// Invert a 3×3 matrix via cofactor expansion. Returns None if the
/// determinant is below a safe threshold (near-singular).
///
/// Threshold 1e-20 is aggressive — in elasticity the smallest valid
/// block determinants for physically reasonable E and grid sizes are
/// around 1e-5 * (E·h)³. Even void elements at ρ_min^p = 10^-9 produce
/// determinants well above 1e-20. A trip here means something is actually
/// broken (e.g., a node with zero elements), not a numerical edge case.
fn invert_3x3(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-20 {
        return None;
    }

    let inv_det = 1.0 / det;
    let mut inv = [[0.0_f64; 3]; 3];

    inv[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * inv_det;
    inv[0][2] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * inv_det;
    inv[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * inv_det;
    inv[2][0] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * inv_det;
    inv[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    Some(inv)
}

// ── Prolongation operator ─────────────────────────────────────────────────

/// Prolongation operator P: coarse grid → fine grid via trilinear interpolation.
///
/// Grid convention:
///   Fine grid:   nx × ny × nz nodes (0-based), DOFs interleaved [ux,uy,uz] per node.
///   Coarse grid: cx × cy × cz nodes, cx = (nx+1)/2, same for y and z.
///   Coarse node (ci,cj,ck) maps exactly to fine node (2·ci, 2·cj, 2·ck).
///
/// Weight per fine node (tensor product of 1D stencil [1/2, 1, 1/2]):
///   0 odd axes → 1 parent,  weight 1.0   (coincide)
///   1 odd axis → 2 parents, weight 0.5   (edge centre)
///   2 odd axes → 4 parents, weight 0.25  (face centre)
///   3 odd axes → 8 parents, weight 0.125 (body centre)
///
/// Boundary: off-grid coarse parents are zero-extended (skipped).
/// Use odd fine-grid dimensions so every fine boundary node coincides with a
/// coarse node — this keeps partition-of-unity intact at all fine nodes and
/// avoids one-sided interpolation at the grid edge.
pub struct Prolongation {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub cx: usize,
    pub cy: usize,
    pub cz: usize,
}

impl Prolongation {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz, cx: (nx + 1) / 2, cy: (ny + 1) / 2, cz: (nz + 1) / 2 }
    }

    /// u_fine ← P · u_coarse  (overwrites u_fine).
    ///
    /// u_coarse: length 3·cx·cy·cz
    /// u_fine:   length 3·nx·ny·nz
    pub fn apply(&self, u_coarse: &[f64], u_fine: &mut [f64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (cx, cy, cz) = (self.cx, self.cy, self.cz);
        assert_eq!(u_coarse.len(), 3 * cx * cy * cz);
        assert_eq!(u_fine.len(),   3 * nx * ny * nz);
        u_fine.iter_mut().for_each(|v| *v = 0.0);

        for fz in 0..nz {
            let (z0, zc) = if fz % 2 == 0 { (fz / 2, 1usize) } else { (fz / 2, 2) };
            let zw = if fz % 2 == 0 { 1.0_f64 } else { 0.5 };
            for fy in 0..ny {
                let (y0, yc) = if fy % 2 == 0 { (fy / 2, 1usize) } else { (fy / 2, 2) };
                let yw = if fy % 2 == 0 { 1.0_f64 } else { 0.5 };
                for fx in 0..nx {
                    let (x0, xc) = if fx % 2 == 0 { (fx / 2, 1usize) } else { (fx / 2, 2) };
                    let xw = if fx % 2 == 0 { 1.0_f64 } else { 0.5 };
                    let fine_node = fz * ny * nx + fy * nx + fx;
                    for dk in 0..zc {
                        let ck = z0 + dk; if ck >= cz { continue; }
                        for dj in 0..yc {
                            let cj = y0 + dj; if cj >= cy { continue; }
                            for di in 0..xc {
                                let ci = x0 + di; if ci >= cx { continue; }
                                let coarse_node = ck * cy * cx + cj * cx + ci;
                                let w = xw * yw * zw;
                                for d in 0..3 {
                                    u_fine[3 * fine_node + d] +=
                                        w * u_coarse[3 * coarse_node + d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Restriction operator ──────────────────────────────────────────────────

/// Restriction operator R = Pᵀ: fine grid → coarse grid.
///
/// Implemented as the exact algebraic transpose of Prolongation: the identical
/// stencil loop with source and destination swapped. This guarantees
///
///   ⟨R·x, y⟩  ==  ⟨x, P·y⟩   (to machine precision)
///
/// which is necessary for the Phase 4 Galerkin coarse operator
/// A_H = Pᵀ·A_h·P to be SPD when A_h is SPD.
///
/// Row sums of R:
///   Interior coarse nodes: 8  (column of P sums to 8 over all contributing fine nodes)
///   Boundary coarse nodes: < 8 (fewer fine neighbours within the domain)
///
/// This is NOT the "full-weighting" restriction with row sums of 1 that some
/// texts describe. Those texts scale by 1/8, breaking the exact transpose
/// property. We keep R = Pᵀ unscaled; the V-cycle is scaled in Phase 5.
pub struct Restriction {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub cx: usize,
    pub cy: usize,
    pub cz: usize,
}

impl Restriction {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self { nx, ny, nz, cx: (nx + 1) / 2, cy: (ny + 1) / 2, cz: (nz + 1) / 2 }
    }

    /// u_coarse ← R · u_fine  (overwrites u_coarse).
    ///
    /// u_fine:   length 3·nx·ny·nz
    /// u_coarse: length 3·cx·cy·cz
    pub fn apply(&self, u_fine: &[f64], u_coarse: &mut [f64]) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let (cx, cy, cz) = (self.cx, self.cy, self.cz);
        assert_eq!(u_fine.len(),   3 * nx * ny * nz);
        assert_eq!(u_coarse.len(), 3 * cx * cy * cz);
        u_coarse.iter_mut().for_each(|v| *v = 0.0);

        for fz in 0..nz {
            let (z0, zc) = if fz % 2 == 0 { (fz / 2, 1usize) } else { (fz / 2, 2) };
            let zw = if fz % 2 == 0 { 1.0_f64 } else { 0.5 };
            for fy in 0..ny {
                let (y0, yc) = if fy % 2 == 0 { (fy / 2, 1usize) } else { (fy / 2, 2) };
                let yw = if fy % 2 == 0 { 1.0_f64 } else { 0.5 };
                for fx in 0..nx {
                    let (x0, xc) = if fx % 2 == 0 { (fx / 2, 1usize) } else { (fx / 2, 2) };
                    let xw = if fx % 2 == 0 { 1.0_f64 } else { 0.5 };
                    let fine_node = fz * ny * nx + fy * nx + fx;
                    for dk in 0..zc {
                        let ck = z0 + dk; if ck >= cz { continue; }
                        for dj in 0..yc {
                            let cj = y0 + dj; if cj >= cy { continue; }
                            for di in 0..xc {
                                let ci = x0 + di; if ci >= cx { continue; }
                                let coarse_node = ck * cy * cx + cj * cx + ci;
                                let w = xw * yw * zw;
                                for d in 0..3 {
                                    u_coarse[3 * coarse_node + d] +=
                                        w * u_fine[3 * fine_node + d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// ── Galerkin coarse operator (Phase 4) ───────────────────────────────────

/// Owned CSR matrix. Introduced in Phase 4 so each multigrid level can
/// carry its stiffness operator without lifetime parameters.
///
/// Invariants upheld by every constructor in this module:
///   row_ptr.len() == n_rows + 1
///   col_idx and values have the same length (== nnz)
///   each row's col_idx slice is strictly sorted, values in [0, n_cols)
pub struct OwnedCsr {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values:  Vec<f64>,
    /// Number of columns. Not inferrable from row_ptr alone for non-square matrices.
    pub n_cols:  usize,
}

impl OwnedCsr {
    /// Number of rows.
    pub fn n_rows(&self) -> usize {
        self.row_ptr.len().saturating_sub(1)
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// y ← self · x. Panics if lengths are inconsistent.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.n_rows();
        assert_eq!(x.len(), self.n_cols,
            "matvec: x length {} != n_cols {}", x.len(), self.n_cols);
        assert_eq!(y.len(), n,
            "matvec: y length {} != n_rows {}", y.len(), n);
        for i in 0..n {
            let mut s = 0.0_f64;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                s += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = s;
        }
    }

    /// Extract the main diagonal. Entries missing from the sparsity pattern
    /// return 0.0. Length = min(n_rows, n_cols).
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.n_rows().min(self.n_cols);
        let mut d = vec![0.0_f64; n];
        for i in 0..n {
            let row = &self.col_idx[self.row_ptr[i]..self.row_ptr[i + 1]];
            if let Ok(local) = row.binary_search(&i) {
                d[i] = self.values[self.row_ptr[i] + local];
            }
        }
        d
    }

    /// Look up a single entry. Returns 0.0 if structurally zero.
    /// Binary-searches the sorted row slice — O(log nnz_per_row).
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let slice = &self.col_idx[self.row_ptr[row]..self.row_ptr[row + 1]];
        match slice.binary_search(&col) {
            Ok(local) => self.values[self.row_ptr[row] + local],
            Err(_)    => 0.0,
        }
    }
}

/// Materialize the Prolongation stencil as an explicit CSR matrix.
///
/// Dimensions: N_fine × N_coarse  where N_fine = 3·nx·ny·nz,
///   N_coarse = 3·cx·cy·cz.
///
/// Each fine DOF row has at most 8 nonzeros (one per trilinear coarse parent).
/// DOFs are interleaved: row 3·fn+d is nonzero only in columns 3·cn+d,
/// so the stencil is block-diagonal per component (no DOF mixing in P).
///
/// Columns within each row are sorted because we sort parents by coarse
/// node index before writing, and the +d offset preserves that order.
fn materialize_p_csr(p: &Prolongation) -> OwnedCsr {
    let (nx, ny, nz) = (p.nx, p.ny, p.nz);
    let (cx, cy, cz) = (p.cx, p.cy, p.cz);
    let n_fine   = 3 * nx * ny * nz;
    let n_coarse = 3 * cx * cy * cz;

    let mut row_ptr = vec![0usize; n_fine + 1];
    // Upper bound: 8 coarse parents × 3 DOFs per fine DOF.
    let mut col_idx: Vec<usize> = Vec::with_capacity(8 * n_fine);
    let mut values:  Vec<f64>   = Vec::with_capacity(8 * n_fine);

    for fz in 0..nz {
        let (z0, zc) = if fz % 2 == 0 { (fz / 2, 1usize) } else { (fz / 2, 2) };
        let zw: f64  = if fz % 2 == 0 { 1.0 } else { 0.5 };
        for fy in 0..ny {
            let (y0, yc) = if fy % 2 == 0 { (fy / 2, 1usize) } else { (fy / 2, 2) };
            let yw: f64  = if fy % 2 == 0 { 1.0 } else { 0.5 };
            for fx in 0..nx {
                let (x0, xc) = if fx % 2 == 0 { (fx / 2, 1usize) } else { (fx / 2, 2) };
                let xw: f64  = if fx % 2 == 0 { 1.0 } else { 0.5 };
                let fine_node = fz * ny * nx + fy * nx + fx;

                // Collect (coarse_node, weight) pairs, sorted by coarse_node.
                // Sorting here means col_idx within each DOF row is also sorted
                // (the +d offset shifts all columns uniformly, preserving order).
                let mut parents: Vec<(usize, f64)> = Vec::with_capacity(8);
                for dk in 0..zc {
                    let ck = z0 + dk; if ck >= cz { continue; }
                    for dj in 0..yc {
                        let cj = y0 + dj; if cj >= cy { continue; }
                        for di in 0..xc {
                            let ci = x0 + di; if ci >= cx { continue; }
                            let cn = ck * cy * cx + cj * cx + ci;
                            parents.push((cn, xw * yw * zw));
                        }
                    }
                }
                parents.sort_unstable_by_key(|&(cn, _)| cn);

                // Write one CSR row per DOF component d ∈ {0,1,2}.
                // Rows are filled in strictly increasing fine_dof order
                // (outer loops are fx, fy, fz in ascending order).
                for d in 0..3 {
                    for &(cn, w) in &parents {
                        col_idx.push(3 * cn + d);
                        values.push(w);
                    }
                    row_ptr[3 * fine_node + d + 1] = col_idx.len();
                }
            }
        }
    }

    OwnedCsr { row_ptr, col_idx, values, n_cols: n_coarse }
}

/// Compute C = A · B for two sparse matrices in CSR format.
///
/// Algorithm: dense-accumulator scatter-gather.
///   For each output row i:
///     Scatter: for each (i,k) in A, add A[i,k]·B[k,:] into a dense acc[].
///     Gather:  collect and sort occupied columns, write to CSR.
///
/// The accumulator `acc` has length B.n_cols and is allocated once, reused
/// across rows. A boolean marker `in_acc` tracks which columns are occupied
/// so only those entries need to be reset each row — O(nnz_per_output_row),
/// not O(B.n_cols).
///
/// Output column indices within each row are sorted ascending.
fn spmatmat(
    a_row_ptr: &[usize], a_col_idx: &[usize], a_values: &[f64],
    b_row_ptr: &[usize], b_col_idx: &[usize], b_values: &[f64],
    b_n_cols:  usize,
) -> OwnedCsr {
    let n_a = a_row_ptr.len() - 1;
    let n_b = b_row_ptr.len() - 1;

    let mut c_row_ptr = vec![0usize; n_a + 1];
    let mut c_col_idx: Vec<usize> = Vec::new();
    let mut c_values:  Vec<f64>   = Vec::new();

    let mut acc      = vec![0.0_f64; b_n_cols];
    let mut in_acc   = vec![false;   b_n_cols];
    let mut occupied: Vec<usize> = Vec::new();

    for i in 0..n_a {
        // Scatter phase: accumulate contributions from A[i,:] × B.
        for pa in a_row_ptr[i]..a_row_ptr[i + 1] {
            let k    = a_col_idx[pa];
            let a_ik = a_values[pa];
            debug_assert!(k < n_b,
                "A col index {k} out of range for B which has {n_b} rows");
            for pb in b_row_ptr[k]..b_row_ptr[k + 1] {
                let j = b_col_idx[pb];
                if !in_acc[j] {
                    in_acc[j] = true;
                    occupied.push(j);
                }
                acc[j] += a_ik * b_values[pb];
            }
        }

        // Gather phase: sort occupied columns, write to output CSR, reset acc.
        occupied.sort_unstable();
        for &j in &occupied {
            c_col_idx.push(j);
            c_values.push(acc[j]);
            // Reset here — not in a separate pass — so we only touch occupied entries.
            acc[j]    = 0.0;
            in_acc[j] = false;
        }
        occupied.clear();

        c_row_ptr[i + 1] = c_col_idx.len();
    }

    OwnedCsr { row_ptr: c_row_ptr, col_idx: c_col_idx, values: c_values, n_cols: b_n_cols }
}

/// Compute B = Aᵀ in CSR format.
///
/// Standard two-pass algorithm:
///   Pass 1 — count nnz per output row (= per input column).
///   Pass 2 — scatter (row i, value v) pairs into output row j.
///
/// The scatter order is determined by input row iteration, which is generally
/// not sorted for output columns. Each output row is sorted after the scatter.
fn transpose_csr(
    row_ptr: &[usize],
    col_idx: &[usize],
    values:  &[f64],
    n_rows:  usize,   // rows of A  (= cols of Aᵀ)
    n_cols:  usize,   // cols of A  (= rows of Aᵀ)
) -> OwnedCsr {
    // Pass 1: count entries per output row.
    let mut counts = vec![0usize; n_cols];
    for &c in col_idx { counts[c] += 1; }

    // Build output row_ptr.
    let mut t_row_ptr = vec![0usize; n_cols + 1];
    for c in 0..n_cols {
        t_row_ptr[c + 1] = t_row_ptr[c] + counts[c];
    }
    let nnz = t_row_ptr[n_cols];

    let mut t_col_idx = vec![0usize; nnz];
    let mut t_values  = vec![0.0_f64; nnz];
    // One write cursor per output row, initialised from row_ptr.
    let mut write_pos: Vec<usize> = t_row_ptr[..n_cols].to_vec();

    // Pass 2: scatter.
    for i in 0..n_rows {
        for k in row_ptr[i]..row_ptr[i + 1] {
            let j = col_idx[k];
            let p = write_pos[j];
            t_col_idx[p] = i;
            t_values[p]  = values[k];
            write_pos[j] += 1;
        }
    }

    // Sort each output row by column index.
    // For P, fine-node indices arrive in monotone order so rows of Pᵀ are
    // already sorted. We sort anyway for robustness with arbitrary input.
    for c in 0..n_cols {
        let s = t_row_ptr[c];
        let e = t_row_ptr[c + 1];
        if e - s <= 1 { continue; }
        let mut pairs: Vec<(usize, f64)> = t_col_idx[s..e]
            .iter().zip(t_values[s..e].iter())
            .map(|(&ci, &vi)| (ci, vi))
            .collect();
        pairs.sort_unstable_by_key(|&(ci, _)| ci);
        for (k, (ci, vi)) in pairs.into_iter().enumerate() {
            t_col_idx[s + k] = ci;
            t_values[s + k]  = vi;
        }
    }

    OwnedCsr { row_ptr: t_row_ptr, col_idx: t_col_idx, values: t_values, n_cols: n_rows }
}

/// Compute the Galerkin coarse operator A_H = Pᵀ · A_h · P via sequential SpMM.
///
/// Steps:
///   1. Materialize P as an explicit CSR matrix  (N_fine × N_coarse).
///   2. T   = A_h · P                            (N_fine × N_coarse).
///   3. Pᵀ  = transpose(P)                       (N_coarse × N_fine).
///   4. A_H = Pᵀ · T                             (N_coarse × N_coarse).
///
/// A_H is SPD whenever A_h is SPD because R = Pᵀ (unscaled) guarantees
/// ⟨A_H·v, v⟩ = ⟨A_h·Pv, Pv⟩ ≥ 0 with equality only at v = 0
/// (P has full column rank for any reasonable grid).
///
/// Panics if a_h dimensions are inconsistent with p.
pub fn galerkin_triple_product(a_h: &OwnedCsr, p: &Prolongation) -> OwnedCsr {
    let n_fine   = 3 * p.nx * p.ny * p.nz;
    let n_coarse = 3 * p.cx * p.cy * p.cz;
    assert_eq!(a_h.n_rows(), n_fine,
        "A_h has {} rows but prolongation expects {} fine DOFs",
        a_h.n_rows(), n_fine);
    assert_eq!(a_h.n_cols, n_fine,
        "A_h must be square: {} rows but n_cols={}",
        a_h.n_rows(), a_h.n_cols);

    // Step 1 — explicit P in CSR.
    let p_csr = materialize_p_csr(p);

    // Step 2 — T = A_h · P.
    let t = spmatmat(
        &a_h.row_ptr, &a_h.col_idx, &a_h.values,
        &p_csr.row_ptr, &p_csr.col_idx, &p_csr.values,
        n_coarse,
    );

    // Step 3 — Pᵀ.
    let pt = transpose_csr(
        &p_csr.row_ptr, &p_csr.col_idx, &p_csr.values,
        n_fine, n_coarse,
    );

    // Step 4 — A_H = Pᵀ · T.
    spmatmat(
        &pt.row_ptr, &pt.col_idx, &pt.values,
        &t.row_ptr,  &t.col_idx,  &t.values,
        n_coarse,
    )
}

// ── Level hierarchy ───────────────────────────────────────────────────────

/// DOF count at which the hierarchy terminates and Phase 6 applies a direct
/// (or heavily-iterated CG) coarse-grid solve. 512 ≈ 5×5×5 nodes × 3 DOFs.
/// Adjust upward if the direct solve in Phase 6 is too slow at this size.
pub const COARSEST_DOFS: usize = 512;

/// One level in the Galerkin multigrid hierarchy.
///
/// `p` and `r` are the transfer operators to/from the **next coarser** level.
/// At the coarsest level they are still present (cx/cy/cz are nonzero) but
/// are never called by the V-cycle — the V-cycle detects the coarsest level
/// by index, not by the absence of p/r.
pub struct GridLevel {
    /// Stiffness operator at this level.
    pub a:  OwnedCsr,
    /// Prolongation to next-finer level (used by V-cycle post-correction).
    pub p:  Prolongation,
    /// Restriction to next-coarser level (used by V-cycle pre-residual transfer).
    pub r:  Restriction,
    /// Node counts at this level.
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

/// Full Galerkin multigrid level hierarchy.
///
/// `levels[0]` = finest, `levels[L-1]` = coarsest (≤ COARSEST_DOFS DOFs).
/// Built once per SIMP iteration (A_h changes every iteration).
/// The V-cycle (Phase 5) indexes into `levels[k]` for smoothing and uses
/// `levels[k].p` / `levels[k].r` for inter-level transfer.
pub struct GalerkinHierarchy {
    pub levels: Vec<GridLevel>,
}

impl GalerkinHierarchy {
    /// Build the complete level hierarchy from the fine-grid stiffness matrix.
    ///
    /// `nx`, `ny`, `nz`  — **node** counts on the fine grid (n_elem_x + 1, etc.).
    /// `max_levels`       — hard cap on depth; COARSEST_DOFS terminates earlier
    ///                      on typical grids.
    ///
    /// The fine matrix is accepted as raw slices matching `cg_solve_direct`'s
    /// signature so the caller does not need to construct OwnedCsr.
    pub fn build(
        a_rows:     &[usize],
        a_cols:     &[usize],
        a_vals:     &[f64],
        nx:         usize,
        ny:         usize,
        nz:         usize,
        max_levels: usize,
    ) -> Self {
        assert!(max_levels >= 1);
        let n_fine = a_rows.len() - 1;
        assert_eq!(n_fine, 3 * nx * ny * nz,
            "matrix has {n_fine} rows but 3·{nx}·{ny}·{nz}={} expected",
            3 * nx * ny * nz);

        let a_fine = OwnedCsr {
            row_ptr: a_rows.to_vec(),
            col_idx: a_cols.to_vec(),
            values:  a_vals.to_vec(),
            n_cols:  n_fine,
        };

        let mut levels: Vec<GridLevel> = Vec::with_capacity(max_levels);
        levels.push(GridLevel {
            p:  Prolongation::new(nx, ny, nz),
            r:  Restriction::new(nx, ny, nz),
            a:  a_fine,
            nx, ny, nz,
        });

        for _ in 1..max_levels {
            // Extract coarse dimensions from the current-finest level's P.
            // Done in a scoped block so the immutable borrow ends before push().
            let (cx, cy, cz) = {
                let last = levels.last().unwrap();
                (last.p.cx, last.p.cy, last.p.cz)
            };

            // Build A_H = Pᵀ A_h P. Borrow ends when this block exits.
            let a_coarse = {
                let last = levels.last().unwrap();
                galerkin_triple_product(&last.a, &last.p)
            };

            levels.push(GridLevel {
                p:  Prolongation::new(cx, cy, cz),
                r:  Restriction::new(cx, cy, cz),
                a:  a_coarse,
                nx: cx,
                ny: cy,
                nz: cz,
            });

            // Terminate after pushing: the level just added is the coarsest.
            // Checking after push (not before) ensures we always descend into
            // a level ≤ COARSEST_DOFS rather than stopping one level above it.
            if 3 * cx * cy * cz <= COARSEST_DOFS {
                break;
            }
        }

        Self { levels }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // ── Test harness helpers ──────────────────────────────────────────────

    /// Build a trivial block-diagonal CSR: each 3×3 block is 2*I. Used for
    /// basic plumbing tests where we need a valid CSR but don't care about
    /// off-diagonal coupling.
    fn trivial_csr_block_diagonal(n_nodes: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let n_dof = 3 * n_nodes;
        let mut k_rows = Vec::with_capacity(n_dof + 1);
        let mut k_cols = Vec::with_capacity(3 * n_dof);
        let mut k_vals = Vec::with_capacity(3 * n_dof);

        k_rows.push(0);
        for node in 0..n_nodes {
            for d_row in 0..3 {
                for d_col in 0..3 {
                    k_cols.push(3 * node + d_col);
                    k_vals.push(if d_row == d_col { 2.0 } else { 0.0 });
                }
                k_rows.push(k_cols.len());
            }
        }

        (k_rows, k_cols, k_vals)
    }

    /// Scalar Laplacian on a 1D grid of n_nodes, replicated across 3 DOFs
    /// per node. Produces a proper sparse structure with off-diagonal
    /// coupling — enough to exercise symmetry and smoothing tests without
    /// pulling in the full elasticity assembly.
    fn scalar_laplacian_3dof_1d(n_nodes: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let n_dof = 3 * n_nodes;
        let mut k_rows = Vec::with_capacity(n_dof + 1);
        let mut k_cols = Vec::new();
        let mut k_vals = Vec::new();

        k_rows.push(0);
        for node in 0..n_nodes {
            for d in 0..3 {
                // Left neighbor
                if node > 0 {
                    k_cols.push(3 * (node - 1) + d);
                    k_vals.push(-1.0);
                }
                // Diagonal: 2.0 interior, 1.0 on the 1D boundary to make K SPD
                k_cols.push(3 * node + d);
                let diag = if node == 0 || node == n_nodes - 1 { 1.0 } else { 2.0 };
                k_vals.push(diag);
                // Right neighbor
                if node < n_nodes - 1 {
                    k_cols.push(3 * (node + 1) + d);
                    k_vals.push(-1.0);
                }
                k_rows.push(k_cols.len());
            }
        }

        (k_rows, k_cols, k_vals)
    }

    /// 3D scalar Laplacian with 3 DOFs per node, 7-point stencil. Each of
    /// the 3 DOFs at a node is a decoupled scalar Laplacian (no inter-DOF
    /// coupling), which gives us a valid SPD 3-DOF-per-node system that
    /// exercises the block-GS code path without requiring full elasticity.
    ///
    /// Boundary handling: Dirichlet-like row (diag = (# of interior neighbors) + 1)
    /// so boundary rows are strictly diagonally dominant — SPD guaranteed.
    fn scalar_laplacian_3dof_3d(nx: usize, ny: usize, nz: usize)
        -> (Vec<usize>, Vec<usize>, Vec<f64>)
    {
        let n_nodes = nx * ny * nz;
        let n_dof   = 3 * n_nodes;
        let mut k_rows = Vec::with_capacity(n_dof + 1);
        let mut k_cols = Vec::new();
        let mut k_vals = Vec::new();

        let node_idx = |ix: usize, iy: usize, iz: usize| -> usize {
            iz * (ny * nx) + iy * nx + ix
        };

        k_rows.push(0);
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let mut n_neighbors = 0_usize;
                    if ix > 0      { n_neighbors += 1; }
                    if ix + 1 < nx { n_neighbors += 1; }
                    if iy > 0      { n_neighbors += 1; }
                    if iy + 1 < ny { n_neighbors += 1; }
                    if iz > 0      { n_neighbors += 1; }
                    if iz + 1 < nz { n_neighbors += 1; }
                    let diag = n_neighbors as f64 + 1.0;

                    let me = node_idx(ix, iy, iz);

                    for d in 0..3 {
                        let mut row: Vec<(usize, f64)> = Vec::with_capacity(7);
                        if ix > 0      { row.push((3 * node_idx(ix - 1, iy, iz) + d, -1.0)); }
                        if ix + 1 < nx { row.push((3 * node_idx(ix + 1, iy, iz) + d, -1.0)); }
                        if iy > 0      { row.push((3 * node_idx(ix, iy - 1, iz) + d, -1.0)); }
                        if iy + 1 < ny { row.push((3 * node_idx(ix, iy + 1, iz) + d, -1.0)); }
                        if iz > 0      { row.push((3 * node_idx(ix, iy, iz - 1) + d, -1.0)); }
                        if iz + 1 < nz { row.push((3 * node_idx(ix, iy, iz + 1) + d, -1.0)); }
                        row.push((3 * me + d, diag));
                        row.sort_by_key(|e| e.0);

                        for (c, v) in row {
                            k_cols.push(c);
                            k_vals.push(v);
                        }
                        k_rows.push(k_cols.len());
                    }
                }
            }
        }

        (k_rows, k_cols, k_vals)
    }

    /// CSR mat-vec for testing residual monotonicity.
    fn csr_matvec(k_rows: &[usize], k_cols: &[usize], k_vals: &[f64], x: &[f64]) -> Vec<f64> {
        let n = k_rows.len() - 1;
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let mut acc = 0.0_f64;
            for k in k_rows[i]..k_rows[i + 1] {
                acc += k_vals[k] * x[k_cols[k]];
            }
            y[i] = acc;
        }
        y
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Wrap a raw CSR triple into OwnedCsr for test use.
    /// Production code passes raw slices directly to GalerkinHierarchy::build.
    fn owned_csr_from_triple(rows: &[usize], cols: &[usize], vals: &[f64]) -> OwnedCsr {
        let n = rows.len() - 1;
        OwnedCsr {
            row_ptr: rows.to_vec(),
            col_idx: cols.to_vec(),
            values:  vals.to_vec(),
            n_cols:  n,
        }
    }

    // ── Phase 2 tests — smoother ──────────────────────────────────────────

    #[test]
    fn invert_3x3_identity() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = invert_3x3(&id).expect("identity is invertible");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn invert_3x3_rejects_singular() {
        // Rank-1 matrix: [[1,2,3],[2,4,6],[3,6,9]]. Determinant = 0.
        let singular = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert!(invert_3x3(&singular).is_none());
    }

    #[test]
    fn rb_coloring_is_conflict_free_on_10x10x10() {
        let nx = 10; let ny = 10; let nz = 10;
        let n_nodes = nx * ny * nz;
        let (kr, kc, kv) = trivial_csr_block_diagonal(n_nodes);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, nx, ny, nz);

        let (nr, nb) = sm.node_color_counts();
        assert_eq!(nr + nb, n_nodes, "every node gets a color");

        // Verify: no red node has a red face-neighbor.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let node = iz * (ny * nx) + iy * nx + ix;
                    let node_red = sm.is_red(node);
                    let neighbors = [
                        (ix as isize - 1, iy as isize,     iz as isize),
                        (ix as isize + 1, iy as isize,     iz as isize),
                        (ix as isize,     iy as isize - 1, iz as isize),
                        (ix as isize,     iy as isize + 1, iz as isize),
                        (ix as isize,     iy as isize,     iz as isize - 1),
                        (ix as isize,     iy as isize,     iz as isize + 1),
                    ];
                    for (nx_i, ny_i, nz_i) in neighbors {
                        if nx_i < 0 || nx_i >= nx as isize { continue; }
                        if ny_i < 0 || ny_i >= ny as isize { continue; }
                        if nz_i < 0 || nz_i >= nz as isize { continue; }
                        let n_idx = (nz_i as usize) * (ny * nx)
                                  + (ny_i as usize) * nx
                                  + (nx_i as usize);
                        assert_ne!(node_red, sm.is_red(n_idx),
                            "node {} and neighbor {} have the same color", node, n_idx);
                    }
                }
            }
        }
    }

    #[test]
    fn smoother_produces_no_singular_blocks_on_well_posed_operator() {
        let nx = 8; let ny = 8; let nz = 8;
        let (kr, kc, kv) = trivial_csr_block_diagonal(nx * ny * nz);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, nx, ny, nz);
        assert_eq!(sm.singular_blocks, 0);
    }

    #[test]
    fn smoother_monotonic_residual_reduction() {
        let n_nodes = 20;
        let (kr, kc, kv) = scalar_laplacian_3dof_1d(n_nodes);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, n_nodes, 1, 1);

        let n_dof = 3 * n_nodes;
        let b: Vec<f64> = (0..n_dof).map(|i| ((i as f64) * 0.3).sin()).collect();
        let mut x = vec![0.0_f64; n_dof];

        let residual_norm = |x: &[f64]| -> f64 {
            let kx = csr_matvec(&kr, &kc, &kv, x);
            kx.iter().zip(b.iter()).map(|(ki, bi)| (bi - ki).powi(2)).sum::<f64>().sqrt()
        };

        let mut prev = residual_norm(&x);
        for iter in 0..5 {
            sm.smooth(&mut x, &b);
            let curr = residual_norm(&x);
            assert!(curr < prev,
                "iter {}: residual did not decrease (prev={:.3e}, curr={:.3e})",
                iter, prev, curr);
            prev = curr;
        }
    }

    #[test]
    fn smoother_damps_high_frequency_mode() {
        use std::f64::consts::PI;

        let nx = 8; let ny = 8; let nz = 8;
        let n_nodes = nx * ny * nz;
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, nx, ny, nz);

        let n_dof = 3 * n_nodes;
        let mut x = vec![0.0_f64; n_dof];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let node = iz * (ny * nx) + iy * nx + ix;
                    let val = (3.0 * PI * (ix as f64 + 0.5) / nx as f64).sin()
                            * (3.0 * PI * (iy as f64 + 0.5) / ny as f64).sin()
                            * (3.0 * PI * (iz as f64 + 0.5) / nz as f64).sin();
                    x[3 * node] = val;
                }
            }
        }
        let b = vec![0.0_f64; n_dof];

        let norm_l2 = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let initial = norm_l2(&x);
        assert!(initial > 1e-6, "test setup: initial mode amplitude near zero");

        sm.smooth(&mut x, &b);
        sm.smooth(&mut x, &b);

        let post = norm_l2(&x);
        let damping = post / initial;
        assert!(damping < 0.5,
            "high-frequency damping factor {:.3} exceeds 0.5 threshold \
             after 2 symmetric iterations", damping);
    }

    #[test]
    fn smoother_is_symmetric_operator() {
        let n_nodes = 16;
        let (kr, kc, kv) = scalar_laplacian_3dof_1d(n_nodes);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, n_nodes, 1, 1);

        let n_dof = 3 * n_nodes;
        let r: Vec<f64> = (0..n_dof).map(|i| ((i as f64 + 1.0) * 0.7).sin()).collect();
        let s: Vec<f64> = (0..n_dof).map(|i| ((i as f64 + 1.0) * 1.3).cos()).collect();

        let mut sr = vec![0.0_f64; n_dof];
        sm.smooth(&mut sr, &r);

        let mut ss = vec![0.0_f64; n_dof];
        sm.smooth(&mut ss, &s);

        let lhs = dot(&sr, &s);
        let rhs = dot(&r, &ss);

        let rel_err = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1e-30);
        assert!(rel_err < 1e-10,
            "symmetry check failed: ⟨Sr,s⟩={:.10e}, ⟨r,Ss⟩={:.10e}, rel_err={:.3e}",
            lhs, rhs, rel_err);
    }

    // ── Phase 3 tests — transfer operators ───────────────────────────────

    #[test]
    fn prolongation_preserves_constant_field() {
        let (nx, ny, nz) = (5, 5, 5);
        let p = Prolongation::new(nx, ny, nz);
        let u_coarse = vec![1.0_f64; 3 * p.cx * p.cy * p.cz];
        let mut u_fine = vec![0.0_f64; 3 * nx * ny * nz];
        p.apply(&u_coarse, &mut u_fine);
        for (i, &v) in u_fine.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-14,
                "u_fine[{i}] = {v:.16e}, expected 1.0");
        }
    }

    #[test]
    fn prolongation_preserves_linear_field() {
        let (nx, ny, nz) = (5, 5, 5);
        let p = Prolongation::new(nx, ny, nz);
        let (cx, cy, cz) = (p.cx, p.cy, p.cz);
        let (a, b, c) = (2.0_f64, 3.0, 5.0);
        let mut u_coarse = vec![0.0_f64; 3 * cx * cy * cz];
        for ck in 0..cz {
            for cj in 0..cy {
                for ci in 0..cx {
                    let node = ck * cy * cx + cj * cx + ci;
                    let base = a * (2 * ci) as f64 + b * (2 * cj) as f64 + c * (2 * ck) as f64;
                    for d in 0..3 { u_coarse[3 * node + d] = base + (d + 1) as f64; }
                }
            }
        }
        let mut u_fine = vec![0.0_f64; 3 * nx * ny * nz];
        p.apply(&u_coarse, &mut u_fine);
        for fz in 0..nz {
            for fy in 0..ny {
                for fx in 0..nx {
                    let node = fz * ny * nx + fy * nx + fx;
                    let base = a * fx as f64 + b * fy as f64 + c * fz as f64;
                    for d in 0..3 {
                        let expected = base + (d + 1) as f64;
                        let got = u_fine[3 * node + d];
                        assert!((got - expected).abs() < 1e-12,
                            "f=({fx},{fy},{fz}) d={d}: got {got:.16e}, expected {expected:.16e}");
                    }
                }
            }
        }
    }

    #[test]
    fn restriction_is_transpose_of_prolongation() {
        let (nx, ny, nz) = (5, 5, 5);
        let p = Prolongation::new(nx, ny, nz);
        let r = Restriction::new(nx, ny, nz);
        let n_fine   = 3 * nx * ny * nz;
        let n_coarse = 3 * p.cx * p.cy * p.cz;

        for trial in 0..10_u64 {
            let x_fine: Vec<f64> = (0..n_fine)
                .map(|i| ((i as f64 + trial as f64 + 1.0) * 0.7).sin())
                .collect();
            let y_coarse: Vec<f64> = (0..n_coarse)
                .map(|i| ((i as f64 + trial as f64 + 1.0) * 1.3).cos())
                .collect();

            let mut rx = vec![0.0_f64; n_coarse];
            r.apply(&x_fine, &mut rx);
            let lhs: f64 = rx.iter().zip(&y_coarse).map(|(a, b)| a * b).sum();

            let mut py = vec![0.0_f64; n_fine];
            p.apply(&y_coarse, &mut py);
            let rhs: f64 = x_fine.iter().zip(&py).map(|(a, b)| a * b).sum();

            let scale = lhs.abs().max(rhs.abs()).max(1e-30);
            let rel_err = (lhs - rhs).abs() / scale;
            assert!(rel_err < 1e-13,
                "trial {trial}: ⟨Rx,y⟩={lhs:.10e} ⟨x,Py⟩={rhs:.10e} rel_err={rel_err:.3e}");
        }
    }

    #[test]
    fn restriction_interior_row_sum_equals_8() {
        let (nx, ny, nz) = (9, 9, 9);
        let r = Restriction::new(nx, ny, nz);
        let (cx, cy, cz) = (r.cx, r.cy, r.cz);
        let u_fine = vec![1.0_f64; 3 * nx * ny * nz];
        let mut u_coarse = vec![0.0_f64; 3 * cx * cy * cz];
        r.apply(&u_fine, &mut u_coarse);
        for ck in 1..(cz - 1) {
            for cj in 1..(cy - 1) {
                for ci in 1..(cx - 1) {
                    let node = ck * cy * cx + cj * cx + ci;
                    for d in 0..3 {
                        let got = u_coarse[3 * node + d];
                        assert!((got - 8.0).abs() < 1e-13,
                            "interior ({ci},{cj},{ck}) d={d}: row sum={got:.16e}, expected 8.0");
                    }
                }
            }
        }
    }

    #[test]
    fn prolongate_then_restrict_constant_field_gives_8x() {
        let (nx, ny, nz) = (9, 9, 9);
        let p = Prolongation::new(nx, ny, nz);
        let r = Restriction::new(nx, ny, nz);
        let (cx, cy, cz) = (p.cx, p.cy, p.cz);
        let u_coarse_in = vec![1.0_f64; 3 * cx * cy * cz];
        let mut u_fine = vec![0.0_f64; 3 * nx * ny * nz];
        p.apply(&u_coarse_in, &mut u_fine);
        let mut u_coarse_out = vec![0.0_f64; 3 * cx * cy * cz];
        r.apply(&u_fine, &mut u_coarse_out);
        for ck in 1..(cz - 1) {
            for cj in 1..(cy - 1) {
                for ci in 1..(cx - 1) {
                    let node = ck * cy * cx + cj * cx + ci;
                    for d in 0..3 {
                        let got = u_coarse_out[3 * node + d];
                        assert!((got - 8.0).abs() < 1e-13,
                            "interior ({ci},{cj},{ck}) d={d}: R·P·v={got:.16e}, expected 8.0");
                    }
                }
            }
        }
    }

    // ── Phase 4 tests — Galerkin coarse operator ──────────────────────────

    /// All diagonal entries of A_H must be strictly positive.
    /// Necessary condition for SPD. Catches sign errors and structurally
    /// zero rows from a malformed P or an off-by-one in the scatter.
    #[test]
    fn galerkin_diagonal_all_positive() {
        let (nx, ny, nz) = (5, 5, 5);
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);
        let a_h     = owned_csr_from_triple(&kr, &kc, &kv);
        let p       = Prolongation::new(nx, ny, nz);
        let a_big_h = galerkin_triple_product(&a_h, &p);

        for (i, d) in a_big_h.diagonal().iter().enumerate() {
            assert!(*d > 0.0,
                "A_H diagonal[{i}] = {d:.6e} is not positive");
        }
    }

    /// A_H must be symmetric: A_H[i,j] == A_H[j,i].
    /// Algebraically guaranteed by (PᵀA_hP)ᵀ = PᵀA_hᵀP = PᵀA_hP when A_h
    /// is symmetric. A numerical failure here means spmatmat or transpose_csr
    /// has an index bug, not a floating-point cancellation issue.
    #[test]
    fn galerkin_is_symmetric() {
        let (nx, ny, nz) = (5, 5, 5);
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);
        let a_h     = owned_csr_from_triple(&kr, &kc, &kv);
        let p       = Prolongation::new(nx, ny, nz);
        let a_big_h = galerkin_triple_product(&a_h, &p);

        let n = a_big_h.n_rows();
        // 20 pseudo-random (i,j) pairs spanning the index space.
        for t in 0..20_usize {
            let i = (t * 37 +  3) % n;
            let j = (t * 53 + 11) % n;
            let aij = a_big_h.get(i, j);
            let aji = a_big_h.get(j, i);
            let scale = aij.abs().max(aji.abs()).max(1e-30);
            let rel   = (aij - aji).abs() / scale;
            assert!(rel < 1e-10,
                "A_H[{i},{j}]={aij:.10e}  A_H[{j},{i}]={aji:.10e}  rel={rel:.3e}");
        }
    }

    /// Load-bearing Galerkin condition: ⟨A_H·v_H, v_H⟩ == ⟨A_h·Pv_H, Pv_H⟩.
    ///
    /// This is the defining property of the Galerkin coarse operator in
    /// falsifiable form. Diagonal positivity and symmetry can both pass while
    /// the off-diagonal structure is wrong; the energy identity catches
    /// everything because it involves the full matrix action via matvec.
    /// Tolerance 1e-10 relative.
    #[test]
    fn galerkin_energy_matches_fine() {
        let (nx, ny, nz) = (5, 5, 5);
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);
        let a_h     = owned_csr_from_triple(&kr, &kc, &kv);
        let p       = Prolongation::new(nx, ny, nz);
        let a_big_h = galerkin_triple_product(&a_h, &p);

        let n_coarse = 3 * p.cx * p.cy * p.cz;
        let n_fine   = 3 * nx * ny * nz;

        // Smooth coarse test vector: linear ramp so it has nontrivial energy
        // on both fine and coarse grids.
        let v_h: Vec<f64> = (0..n_coarse).map(|i| (i as f64 + 1.0) * 0.1).collect();

        // LHS = v_H · (A_H · v_H)
        let mut a_h_v = vec![0.0_f64; n_coarse];
        a_big_h.matvec(&v_h, &mut a_h_v);
        let lhs: f64 = v_h.iter().zip(a_h_v.iter()).map(|(a, b)| a * b).sum();

        // RHS = (Pv_H) · (A_h · Pv_H)
        let mut pv = vec![0.0_f64; n_fine];
        p.apply(&v_h, &mut pv);
        let a_h_pv = csr_matvec(&kr, &kc, &kv, &pv);
        let rhs: f64 = pv.iter().zip(a_h_pv.iter()).map(|(a, b)| a * b).sum();

        let scale = lhs.abs().max(rhs.abs()).max(1e-30);
        let rel   = (lhs - rhs).abs() / scale;
        assert!(rel < 1e-10,
            "Galerkin energy: ⟨A_H·v,v⟩={lhs:.12e}  ⟨A_h·Pv,Pv⟩={rhs:.12e}  rel={rel:.3e}");
    }

    /// Build a two-level hierarchy and verify all-positive diagonals at every level.
    /// Exercises GalerkinHierarchy::build at depth — Phase 5 V-cycle requires it.
    ///
    /// Grid: 9×9×9 fine (2187 DOFs) → 5×5×5 coarse (375 DOFs ≤ 512 → terminates).
    #[test]
    fn galerkin_hierarchy_two_levels() {
        let (nx, ny, nz) = (9, 9, 9);
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);

        let hier = GalerkinHierarchy::build(&kr, &kc, &kv, nx, ny, nz, 4);

        assert!(hier.levels.len() >= 2,
            "expected ≥2 levels, got {}", hier.levels.len());

        for (lv, level) in hier.levels.iter().enumerate() {
            for (i, d) in level.a.diagonal().iter().enumerate() {
                assert!(*d > 0.0,
                    "level {lv} diagonal[{i}] = {d:.6e} not positive");
            }
        }
    }

    /// GalerkinHierarchy::build must terminate with the coarsest level having
    /// ≤ COARSEST_DOFS DOFs. This is the structural guarantee Phase 6 relies
    /// on when choosing a coarse-grid solver.
    ///
    /// Grid: 17×17×17 fine (14739 DOFs)
    ///   → 9×9×9  coarse (2187 DOFs)
    ///   → 5×5×5  coarsest (375 DOFs ≤ 512, stop).
    #[test]
    fn galerkin_hierarchy_terminates_at_coarsest_dofs() {
        let (nx, ny, nz) = (17, 17, 17);
        let (kr, kc, kv) = scalar_laplacian_3dof_3d(nx, ny, nz);

        let hier = GalerkinHierarchy::build(&kr, &kc, &kv, nx, ny, nz, 10);

        let coarsest_n = hier.levels.last().unwrap().a.n_rows();
        assert!(coarsest_n <= COARSEST_DOFS,
            "coarsest level has {coarsest_n} DOFs, expected ≤ {COARSEST_DOFS}");

        // Coarsest level must itself be SPD.
        for (i, d) in hier.levels.last().unwrap().a.diagonal().iter().enumerate() {
            assert!(*d > 0.0,
                "coarsest diagonal[{i}] = {d:.6e} not positive");
        }
    }
}