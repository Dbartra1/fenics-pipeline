// Tier 4 phases 2-4 plant infrastructure that Phase 5 (V-cycle) and Phase 6
// (trait integration) wire into the solver. Until then, all items in this
// module are dead code from the compiler's point of view. Remove this
// attribute once the GMG preconditioner is called from cg_solve_direct.
#![allow(dead_code)]

//! Multigrid components for the GMG preconditioner (Tier 4).
//!
//! Phase 2 lands the block-symmetric red-black Gauss-Seidel smoother.
//! Later phases will add restriction/prolongation operators (Phase 3),
//! the level hierarchy with Galerkin coarse operators (Phase 4), and the
//! V-cycle preconditioner (Phases 5-6).
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

    // ── Tests ─────────────────────────────────────────────────────────────

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
                    // Check each face-neighbor if in-grid
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
        // Use a 1D scalar Laplacian replicated on 3 DOFs per node. With
        // Dirichlet-ish boundary (diag=1 at endpoints) the system is SPD
        // and every smoothing iteration must reduce the residual norm.
        let n_nodes = 20;
        let (kr, kc, kv) = scalar_laplacian_3dof_1d(n_nodes);
        // Pass as a 1×1×20 grid so coloring alternates along x.
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
        // 3D scalar Laplacian on 8×8×8. Initialize error to a genuinely
        // oscillatory mode — sin(k·π·x) * sin(k·π·y) * sin(k·π·z) with k
        // near the Nyquist limit. The pure 3D checkerboard is a trivial
        // eigenmode of RB-GS (damping factor ~1, classical MG textbook
        // caveat) so we use sin(3π·ix/n)·sin(3π·iy/n)·sin(3π·iz/n) instead:
        // high-frequency but not the pathological checkerboard itself.
        //
        // Run the homogeneous system K·x = 0 so any amplitude in x after
        // smoothing is pure error. Measure energy norm reduction across
        // 2 smooth() calls (= 4 color sweeps) — textbook expected
        // reduction for RB-GS on a smooth-but-oscillatory mode is 0.3-0.5
        // per symmetric iteration, so 2 iterations should achieve ≤ 0.5.
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
                    // Load mode onto the first DOF of each node; leave
                    // DOFs 1 and 2 zero. Since the 3D Laplacian decouples
                    // across DOFs, the non-zero DOF evolves independently.
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
        // The smoothing operator S maps (r, 0) -> x_out through applying
        // smooth to zero initial guess with b=r. For S to be symmetric,
        //   ⟨S·r, s⟩ == ⟨r, S·s⟩   for arbitrary r, s.
        // This catches forward/backward ordering bugs that would make CG
        // break in Phase 5.
        let n_nodes = 16;
        let (kr, kc, kv) = scalar_laplacian_3dof_1d(n_nodes);
        let sm = RedBlackGSSmoother::new(&kr, &kc, &kv, n_nodes, 1, 1);

        let n_dof = 3 * n_nodes;

        // Two random-ish RHS vectors.
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

    // ── Transfer operator tests ───────────────────────────────────────────

    /// P · ones_coarse = ones_fine: weights per fine node sum to exactly 1.0
    /// (partition of unity). Requires odd fine dimensions so boundary fine
    /// nodes always coincide with a coarse node (no dropped parents).
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

    /// Trilinear interpolation reproduces linear fields exactly.
    /// Coarse values: f(ci,cj,ck) = 2*(2·ci) + 3*(2·cj) + 5*(2·ck) + (d+1).
    /// After prolongation every fine node must satisfy f(fx,fy,fz).
    /// This catches stencil-weight bugs that pass constant fields but fail gradients.
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

    /// ⟨R·x_fine, y_coarse⟩ == ⟨x_fine, P·y_coarse⟩ to 1e-13 across 10 random pairs.
    ///
    /// This is the load-bearing test for Phase 3. If it passes, R = Pᵀ exactly and
    /// the Phase 4 Galerkin operator A_H = Pᵀ·A_h·P is guaranteed to be SPD.
    /// If it fails, do not proceed to Phase 4.
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

    /// R = Pᵀ: for interior coarse nodes the column of P sums to 8, so
    /// R · ones_fine = 8 at those nodes. Boundary nodes receive < 8 (fewer
    /// fine neighbours in domain) — we only check the interior here.
    /// Grid: 9×9×9 fine → 5×5×5 coarse; interior coarse = (1..=3)³ (27 nodes).
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

    /// R·P acts as 8·I on smooth (non-oscillatory) inputs at interior coarse nodes.
    /// Constant fields are maximally smooth: P·ones_coarse = ones_fine exactly
    /// (partition of unity), then R·ones_fine = 8·ones_coarse at interior nodes.
    /// This validates the composition without needing the V-cycle loop from Phase 5.
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
}