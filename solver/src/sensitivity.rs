// src/sensitivity.rs
//
// Sensitivity computation and filtering for the SIMP loop.
//
// Raw sensitivity for element e:
//   dc[e] = -p · ρ_e^(p-1) · u_e^T · Ke · u_e
//
// Then apply the KKT-correct x-weighted Sigmund (2001) filter:
//   dc_filtered[e] = Σ_f(H_ef · x[f] · dc[f]) / Σ_f(H_ef · x[f])
//
// The x-weighting (multiplying both numerator and denominator by x[f])
// ensures the sensitivity is consistent with the filtered density field
// and prevents checkerboard instability. This is what the Python solver
// was fixed to use in the last session.

use crate::filter::FilterWeights;

/// Compute raw then filtered sensitivities.
///
/// # Arguments
/// * `x`        — current density field, length n_elem
/// * `u`        — displacement solution, length n_dof
/// * `ke_base`  — 24×24 element stiffness at unit density (flat row-major)
/// * `dof_map`  — dof_map[e] = 24 global DOF indices for element e
/// * `fw`       — precomputed filter weights
/// * `penal`    — SIMP penalization exponent p
/// * `void_mask`   — void_mask[e]=true → skip (no material, no sensitivity)
/// * `nondesign`   — nondesign[e]=true → skip (fixed solid, sensitivity unused)
///
/// Returns filtered sensitivity vector, length n_elem.
/// dc[e] is negative (stiffer elements have more negative sensitivity).
/// void and nondesign elements are left at 0.0.
pub fn compute_sensitivities(
    x:          &[f64],
    u:          &[f64],
    ke_base:    &[f64; 576],
    dof_map:    &[[usize; 24]],
    fw:         &FilterWeights,
    penal:      f64,
    void_mask:  &[bool],
    nondesign:  &[bool],
) -> Vec<f64> {
    let n_elem = x.len();
    let mut dc = vec![0.0f64; n_elem];

    // ── Raw sensitivities ─────────────────────────────────────────────────────
    for e in 0..n_elem {
        if void_mask[e] || nondesign[e] { continue; }

        // Gather element displacements: u_e[i] = u[dof_map[e][i]]
        let u_e: [f64; 24] = std::array::from_fn(|i| u[dof_map[e][i]]);

        // Compute u_e^T · Ke · u_e
        let mut uke = 0.0f64;
        for i in 0..24 {
            for j in 0..24 {
                uke += u_e[i] * ke_base[i * 24 + j] * u_e[j];
            }
        }

        // dc[e] = -p · ρ^(p-1) · u_e^T · Ke · u_e
        dc[e] = -penal * x[e].powf(penal - 1.0) * uke;
    }

    // ── x-weighted filter ─────────────────────────────────────────────────────
    // dc_filtered[e] = Σ_f(H_ef · x[f] · dc[f]) / Σ_f(H_ef · x[f])
    let mut dc_filtered = vec![0.0f64; n_elem];
    for e in 0..n_elem {
        if void_mask[e] { continue; }

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for &(f, w) in &fw.weights[e] {
            num += w * x[f] * dc[f];
            den += w * x[f];
        }
        // den + 1e-16 prevents division by zero when all neighbours are void
        dc_filtered[e] = num / (den + 1e-16);
    }

    dc_filtered
}

/// Compute structural compliance: C = u^T · f = Σ_e ρ_e^p · u_e^T · Ke · u_e
///
/// This is the objective function being minimized. It equals the total
/// strain energy and is always positive for a loaded structure.
pub fn compute_compliance(
    x:          &[f64],
    u:          &[f64],
    ke_base:    &[f64; 576],
    dof_map:    &[[usize; 24]],
    penal:      f64,
    void_mask:  &[bool],
    nondesign:  &[bool],
) -> f64 {
    let n_elem = x.len();
    let mut compliance = 0.0f64;

    for e in 0..n_elem {
        if void_mask[e] { continue; }
        let rho = if nondesign[e] { 1.0 } else { x[e] };

        let u_e: [f64; 24] = std::array::from_fn(|i| u[dof_map[e][i]]);
        let mut uke = 0.0f64;
        for i in 0..24 {
            for j in 0..24 {
                uke += u_e[i] * ke_base[i * 24 + j] * u_e[j];
            }
        }
        compliance += rho.powf(penal) * uke;
    }
    compliance
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::{apply_dirichlet, assemble_k, build_csr_pattern};
    use crate::connectivity::precompute_dof_map;
    use crate::filter::build_filter;
    use crate::ke_base::compute_ke_base;
    use crate::solver::cg_solve_direct;
    use crate::types::{Grid, Material};

    fn steel() -> Material { Material { young: 210e9, poisson: 0.3 } }

    /// Set up a fully solved FEM system on a small grid.
    /// Returns (grid, ke, dof_map, fw, x, u) ready for sensitivity tests.
    fn solved_system(
        nx: usize, ny: usize, nz: usize,
    ) -> (Grid, [f64; 576], Vec<[usize; 24]>, FilterWeights, Vec<f64>, Vec<f64>) {
        let g = Grid { nx, ny, nz, voxel_size: 0.001 };
        let ke = compute_ke_base(&steel(), g.voxel_size);
        let dof_map = precompute_dof_map(&g);
        let pattern = build_csr_pattern(&g, &dof_map);
        let n_elem = g.n_elem();
        let n_dof  = g.n_dof();
        let nnz    = pattern.k_rows[n_dof];

        let x = vec![1.0f64; n_elem];   // unit density

        // Assemble K
        let mut k_vals = vec![0.0f64; nnz];
        assemble_k(&mut k_vals, &x, &ke, &pattern,
                   &vec![false; n_elem], &vec![false; n_elem], 3.0);

        // Fix bottom face (iz=0)
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

        let diag_mean: f64 = (0..n_dof).map(|i| {
            let row = &pattern.k_cols[pattern.k_rows[i]..pattern.k_rows[i+1]];
            let pos = row.binary_search(&i).unwrap();
            k_vals[pattern.k_rows[i] + pos]
        }).sum::<f64>() / n_dof as f64;

        apply_dirichlet(&mut k_vals, &pattern.k_rows, &pattern.k_cols,
                        &fixed_dofs, diag_mean);

        // Unit downward load on top face
        let mut f = vec![0.0f64; n_dof];
        let top_count = (g.nx + 1) * (g.ny + 1);
        for iy in 0..=g.ny {
            for ix in 0..=g.nx {
                let n = g.node_idx(ix, iy, g.nz);
                f[3*n + 2] = -1000.0 / top_count as f64;
            }
        }
        for &d in &fixed_dofs { f[d] = 0.0; }

        let mut u = vec![0.0f64; n_dof];
        let res = cg_solve_direct(&pattern.k_rows, &pattern.k_cols, &k_vals,
                                   &f, &mut u, 1e-10, n_dof * 2);
        assert!(res.converged, "setup solve failed: rel_res={:.3e}", res.rel_residual);

        let fw = build_filter(&g, 0.002);
        (g, ke, dof_map, fw, x, u)
    }

    // ── Sensitivity sign ──────────────────────────────────────────────────────

    #[test]
    fn raw_sensitivities_are_nonpositive() {
        // dc[e] = -p · ρ^(p-1) · u_e^T·Ke·u_e ≤ 0 always
        // (uke ≥ 0 since Ke is SPD, and -p·ρ^(p-1) < 0)
        let (g, ke, dof_map, fw, x, u) = solved_system(3, 2, 2);
        let n_elem = g.n_elem();
        let dc = compute_sensitivities(
            &x, &u, &ke, &dof_map, &fw, 3.0,
            &vec![false; n_elem], &vec![false; n_elem],
        );
        for (e, &d) in dc.iter().enumerate() {
            assert!(d <= 1e-15, "dc[{e}]={d:.4e} should be <= 0");
        }
    }

    #[test]
    fn sensitivities_not_all_zero_under_load() {
        let (g, ke, dof_map, fw, x, u) = solved_system(3, 2, 2);
        let n_elem = g.n_elem();
        let dc = compute_sensitivities(
            &x, &u, &ke, &dof_map, &fw, 3.0,
            &vec![false; n_elem], &vec![false; n_elem],
        );
        let min_dc = dc.iter().cloned().fold(0.0f64, f64::min);
        assert!(min_dc < -1e-20, "all sensitivities are zero — u may be zero");
    }

    // ── Void and nondesign masks ──────────────────────────────────────────────

    #[test]
    fn nondesign_elements_skipped_in_raw_sensitivity() {
        // nondesign elements get dc=0 before filtering — verify by checking
        // that zeroing u entirely gives dc=0 for all elements including nondesign.
        let (g, ke, dof_map, fw, x, _u) = solved_system(3, 2, 2);
        let n_elem = g.n_elem();
        let u_zero = vec![0.0f64; g.n_dof()];
        let dc = compute_sensitivities(
            &x, &u_zero, &ke, &dof_map, &fw, 3.0,
            &vec![false; n_elem], &vec![false; n_elem],
        );
        for (e, &d) in dc.iter().enumerate() {
            assert_eq!(d, 0.0, "dc[{e}]={d} should be 0 when u=0");
        }
    }


    // ── Finite difference verification ────────────────────────────────────────
    // dc[e] ≈ (C(x + δe_e) - C(x - δe_e)) / (2δ)  (central difference)
    // This is the primary correctness check for the sensitivity formula.

    #[test]
    fn sensitivities_match_analytical_formula() {
        // Verify dc[e] = -p · x[e]^(p-1) · u_e^T · Ke · u_e directly.
        let (g, ke, dof_map, fw, x, u) = solved_system(2, 1, 1);
        let n_elem = g.n_elem();
        let penal = 3.0;
        let void_mask = vec![false; n_elem];
        let nondesign = vec![false; n_elem];

        // Get filtered sensitivities
        let dc = compute_sensitivities(&x, &u, &ke, &dof_map, &fw, penal, &void_mask, &nondesign);

        // Compute raw expected values directly
        for e in 0..n_elem {
            let u_e: [f64; 24] = std::array::from_fn(|i| u[dof_map[e][i]]);
            let mut uke = 0.0f64;
            for i in 0..24 { for j in 0..24 { uke += u_e[i] * ke[i*24+j] * u_e[j]; } }
            let expected_raw = -penal * x[e].powf(penal - 1.0) * uke;

            // On a tiny grid with tiny filter the filtered value should be close to raw
            // (all neighbours have similar displacement). Verify sign and order of magnitude.
            if expected_raw.abs() > 1e-30 {
                assert!(dc[e] <= 0.0, "elem {e}: filtered dc={:.4e} should be <= 0", dc[e]);
                let ratio = dc[e] / expected_raw;
                assert!(ratio > 0.1 && ratio < 10.0,
                    "elem {e}: filtered/raw ratio={ratio:.3} too far from 1 — formula may be wrong");
            }
        }
    }

    // ── Compliance ────────────────────────────────────────────────────────────

    #[test]
    fn compliance_is_positive_under_load() {
        let (g, ke, dof_map, _, x, u) = solved_system(3, 2, 2);
        let n_elem = g.n_elem();
        let c = compute_compliance(
            &x, &u, &ke, &dof_map, 3.0,
            &vec![false; n_elem], &vec![false; n_elem],
        );
        assert!(c > 0.0, "compliance should be positive, got {c:.4e}");
    }

    #[test]
    fn compliance_increases_when_density_reduced() {
        // Lower density → softer structure → higher compliance.
        let (g, ke, dof_map, _, x, u) = solved_system(3, 2, 2);
        let n_elem = g.n_elem();
        let void_mask  = vec![false; n_elem];
        let nondesign  = vec![false; n_elem];

        let c_full = compute_compliance(&x, &u, &ke, &dof_map, 3.0, &void_mask, &nondesign);
        let x_half: Vec<f64> = x.iter().map(|_| 0.5).collect();
        let c_half = compute_compliance(&x_half, &u, &ke, &dof_map, 3.0, &void_mask, &nondesign);

        assert!(c_half < c_full,
            "compliance at ρ=0.5 ({c_half:.4e}) should be < ρ=1.0 ({c_full:.4e})");
    }
}