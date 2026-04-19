// src/ke_base.rs
//
// 24×24 element stiffness matrix for a trilinear hexahedral element (hex8)
// computed once via 8-point Gauss quadrature.
//
// For a uniform structured grid every element is an identical cube of side h,
// so Ke_base is material+geometry dependent but element-position independent.
// During K assembly it is scaled by ρ^p per element.
//
// Voigt strain ordering:  [εxx, εyy, εzz, γxy, γyz, γzx]
//   where γij = ∂ui/∂xj + ∂uj/∂xi  (engineering shear, not tensorial)
//
// Node ordering MUST match connectivity.rs (bottom CCW then top CCW):
//   0:(-1,-1,-1)  1:(+1,-1,-1)  2:(+1,+1,-1)  3:(-1,+1,-1)
//   4:(-1,-1,+1)  5:(+1,-1,+1)  6:(+1,+1,+1)  7:(-1,+1,+1)
//
// DOF ordering: [u0,v0,w0, u1,v1,w1, ..., u7,v7,w7]  (x,y,z per node)

use crate::types::Material;

/// Compute the 24×24 element stiffness matrix for a unit-density hex element.
///
/// Returns flat row-major [f64; 576]:  ke[i*24 + j] = Ke[i][j]
///
/// # Arguments
/// * `material`   — elastic constants (E, ν)
/// * `voxel_size` — cube side length in metres
pub fn compute_ke_base(material: &Material, voxel_size: f64) -> [f64; 576] {
    let lambda   = material.lame_lambda();
    let mu       = material.lame_mu();
    let lam2mu   = lambda + 2.0 * mu;
    let h        = voxel_size;

    // ── Node natural coordinates ─────────────────────────────────────────────
    // Index matches connectivity.rs node ordering exactly.
    const NXI:   [f64; 8] = [-1.,  1.,  1., -1., -1.,  1.,  1., -1.];
    const NETA:  [f64; 8] = [-1., -1.,  1.,  1., -1., -1.,  1.,  1.];
    const NZETA: [f64; 8] = [-1., -1., -1., -1.,  1.,  1.,  1.,  1.];

    // ── Gauss quadrature ─────────────────────────────────────────────────────
    // 8-point rule: ±1/√3 in each direction, weight = 1 each.
    let gp = 1.0_f64 / 3.0_f64.sqrt();
    let gauss = [-gp, gp];

    // ── Scale factor per Gauss point ─────────────────────────────────────────
    // Jacobian for uniform cube: J = diag(h/2, h/2, h/2)
    //   det(J) = (h/2)³
    //   J⁻¹   = diag(2/h, 2/h, 2/h)
    //
    // B_phys = J⁻¹ B_nat  →  physical derivatives = (2/h) * natural derivatives
    //
    // Per Gauss point (weight w=1):
    //   ΔKe = w · B_phys^T · C · B_phys · det(J)
    //       = B_phys^T · C · B_phys · (h/2)³
    let det_j  = (h * 0.5).powi(3);
    let inv_j  = 2.0 / h;   // J⁻¹ scale for natural → physical derivatives

    let mut ke = [0.0f64; 576];

    for &zeta in &gauss {
    for &eta  in &gauss {
    for &xi   in &gauss {

        // ── Shape function derivatives in natural coordinates ─────────────
        // N_i = (1 + ξ_i ξ)(1 + η_i η)(1 + ζ_i ζ) / 8
        // ∂N_i/∂ξ   = ξ_i   · (1 + η_i η)(1 + ζ_i ζ) / 8
        // ∂N_i/∂η   = η_i   · (1 + ξ_i ξ)(1 + ζ_i ζ) / 8
        // ∂N_i/∂ζ   = ζ_i   · (1 + ξ_i ξ)(1 + η_i η) / 8
        let mut dndx = [0.0f64; 8];
        let mut dndy = [0.0f64; 8];
        let mut dndz = [0.0f64; 8];
        for i in 0..8usize {
            let xi_i   = NXI[i];
            let eta_i  = NETA[i];
            let zeta_i = NZETA[i];
            // Natural derivatives
            let dn_xi   = xi_i   * (1.0 + eta_i*eta)   * (1.0 + zeta_i*zeta) / 8.0;
            let dn_eta  = eta_i  * (1.0 + xi_i*xi)     * (1.0 + zeta_i*zeta) / 8.0;
            let dn_zeta = zeta_i * (1.0 + xi_i*xi)     * (1.0 + eta_i*eta)   / 8.0;
            // Physical derivatives via J⁻¹ = (2/h)I
            dndx[i] = inv_j * dn_xi;
            dndy[i] = inv_j * dn_eta;
            dndz[i] = inv_j * dn_zeta;
        }

        // ── B matrix: 6 rows × 24 cols ────────────────────────────────────
        // Voigt ordering: [εxx, εyy, εzz, γxy, γyz, γzx]
        //
        // For node i, DOF columns [3i, 3i+1, 3i+2] = [u, v, w]:
        //   u-col: [ dNi/dx,       0,       0,  dNi/dy,       0,  dNi/dz ]^T
        //   v-col: [       0, dNi/dy,       0,  dNi/dx, dNi/dz,       0  ]^T
        //   w-col: [       0,       0, dNi/dz,       0,  dNi/dy, dNi/dx  ]^T
        let mut b = [[0.0f64; 24]; 6];
        for i in 0..8usize {
            let (dx, dy, dz) = (dndx[i], dndy[i], dndz[i]);
            // u-DOF (col 3i): εxx, γxy, γzx
            b[0][3*i]   = dx;
            b[3][3*i]   = dy;
            b[5][3*i]   = dz;
            // v-DOF (col 3i+1): εyy, γxy, γyz
            b[1][3*i+1] = dy;
            b[3][3*i+1] = dx;
            b[4][3*i+1] = dz;
            // w-DOF (col 3i+2): εzz, γyz, γzx
            b[2][3*i+2] = dz;
            b[4][3*i+2] = dy;
            b[5][3*i+2] = dx;
        }

        // ── C·B: 6×24, exploit C's block-diagonal structure ──────────────
        // C = [ λ+2μ  λ     λ     0  0  0 ]
        //     [ λ     λ+2μ  λ     0  0  0 ]
        //     [ λ     λ     λ+2μ  0  0  0 ]
        //     [ 0     0     0     μ  0  0 ]
        //     [ 0     0     0     0  μ  0 ]
        //     [ 0     0     0     0  0  μ ]
        let mut cb = [[0.0f64; 24]; 6];
        for j in 0..24usize {
            cb[0][j] = lam2mu*b[0][j] + lambda*b[1][j] + lambda*b[2][j];
            cb[1][j] = lambda*b[0][j] + lam2mu*b[1][j] + lambda*b[2][j];
            cb[2][j] = lambda*b[0][j] + lambda*b[1][j] + lam2mu*b[2][j];
            cb[3][j] = mu * b[3][j];
            cb[4][j] = mu * b[4][j];
            cb[5][j] = mu * b[5][j];
        }

        // ── Ke += det(J) · B^T · C · B ───────────────────────────────────
        for i in 0..24usize {
            for j in 0..24usize {
                let mut sum = 0.0f64;
                for k in 0..6usize {
                    sum += b[k][i] * cb[k][j];
                }
                ke[i*24 + j] += det_j * sum;
            }
        }

    }}} // Gauss loops

    ke
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Material;

    fn steel() -> Material {
        Material { young: 210e9, poisson: 0.3 }
    }

    /// Ke[i][j] for any (i,j) pair.
    fn ke_entry(ke: &[f64; 576], i: usize, j: usize) -> f64 {
        ke[i * 24 + j]
    }

    /// Ke · u  (matrix-vector product)
    fn matvec(ke: &[f64; 576], u: &[f64; 24]) -> [f64; 24] {
        let mut f = [0.0f64; 24];
        for i in 0..24 {
            for j in 0..24 {
                f[i] += ke_entry(ke, i, j) * u[j];
            }
        }
        f
    }

    // ── Symmetry ──────────────────────────────────────────────────────────────

    #[test]
    fn ke_is_symmetric() {
        let ke = compute_ke_base(&steel(), 0.0025);
        for i in 0..24 {
            for j in 0..24 {
                let diff = (ke_entry(&ke, i, j) - ke_entry(&ke, j, i)).abs();
                let scale = ke_entry(&ke, i, i).abs().max(ke_entry(&ke, j, j).abs()).max(1.0);
                assert!(
                    diff / scale < 1e-12,
                    "Ke[{i}][{j}]={:.6e} != Ke[{j}][{i}]={:.6e}  (rel diff {:.2e})",
                    ke_entry(&ke, i, j), ke_entry(&ke, j, i), diff/scale
                );
            }
        }
    }

    // ── Diagonal positivity ───────────────────────────────────────────────────

    #[test]
    fn ke_diagonal_entries_are_positive() {
        let ke = compute_ke_base(&steel(), 0.0025);
        for i in 0..24 {
            let d = ke_entry(&ke, i, i);
            assert!(d > 0.0, "Ke[{i}][{i}] = {d:.6e} is not positive");
        }
    }

    // ── Rigid body modes → zero force ────────────────────────────────────────
    // For rigid body translation Ke·u must be exactly zero (up to floating
    // point). This is the partition-of-unity property of isoparametric elements
    // and is the primary correctness check for the Gauss integration.

    #[test]
    fn rigid_body_translation_x_zero_force() {
        let ke = compute_ke_base(&steel(), 0.0025);
        // All nodes: u=1, v=0, w=0
        let mut u = [0.0f64; 24];
        for i in 0..8 { u[3*i] = 1.0; }
        let f = matvec(&ke, &u);
        let ke_diag_max = (0..24).map(|i| ke_entry(&ke, i, i)).fold(0.0f64, f64::max);
        for (i, &fi) in f.iter().enumerate() {
            assert!(
                fi.abs() < ke_diag_max * 1e-12,
                "f[{i}]={fi:.3e} for x-translation (max diag={ke_diag_max:.3e})"
            );
        }
    }

    #[test]
    fn rigid_body_translation_y_zero_force() {
        let ke = compute_ke_base(&steel(), 0.0025);
        let mut u = [0.0f64; 24];
        for i in 0..8 { u[3*i + 1] = 1.0; }
        let f = matvec(&ke, &u);
        let ke_diag_max = (0..24).map(|i| ke_entry(&ke, i, i)).fold(0.0f64, f64::max);
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() < ke_diag_max * 1e-12, "f[{i}]={fi:.3e} for y-translation");
        }
    }

    #[test]
    fn rigid_body_translation_z_zero_force() {
        let ke = compute_ke_base(&steel(), 0.0025);
        let mut u = [0.0f64; 24];
        for i in 0..8 { u[3*i + 2] = 1.0; }
        let f = matvec(&ke, &u);
        let ke_diag_max = (0..24).map(|i| ke_entry(&ke, i, i)).fold(0.0f64, f64::max);
        for (i, &fi) in f.iter().enumerate() {
            assert!(fi.abs() < ke_diag_max * 1e-12, "f[{i}]={fi:.3e} for z-translation");
        }
    }

    // ── Strain energy is positive for non-rigid deformation ──────────────────

    #[test]
    fn uniaxial_stretch_has_positive_strain_energy() {
        // Pure x-stretch: u_i = x_i_natural (±1), v=w=0.
        // This is a non-rigid deformation → u^T Ke u > 0.
        let ke = compute_ke_base(&steel(), 0.0025);
        let u: [f64; 24] = {
            let mut u = [0.0f64; 24];
            // NXI gives the natural x-coords of each node: ±1
            let nxi = [-1., 1., 1., -1., -1., 1., 1., -1.];
            for i in 0..8 { u[3*i] = nxi[i]; }
            u
        };
        let f = matvec(&ke, &u);
        let strain_energy: f64 = u.iter().zip(f.iter()).map(|(ui, fi)| ui * fi).sum();
        assert!(strain_energy > 0.0, "strain energy = {strain_energy:.6e} should be positive");
    }

    // ── Physical units check ──────────────────────────────────────────────────
    // Ke has units N/m (stiffness). Diagonal entries scale as E·h.
    // For steel (E=210GPa, h=2.5mm): expect order ~1e8 N/m.

    #[test]
    fn ke_diagonal_order_of_magnitude() {
        let ke = compute_ke_base(&steel(), 0.0025);
        let diag_avg: f64 = (0..24).map(|i| ke_entry(&ke, i, i)).sum::<f64>() / 24.0;
        // Expect ~E*h = 210e9 * 0.0025 = 5.25e8; allow 2 orders of magnitude
        assert!(diag_avg > 1e6, "diagonal avg {diag_avg:.3e} suspiciously small");
        assert!(diag_avg < 1e11, "diagonal avg {diag_avg:.3e} suspiciously large");
        // Print the actual value for first run — we'll add an exact regression later.
        println!("Ke diagonal avg: {diag_avg:.6e} N/m");
    }

    // ── Voxel size scaling ────────────────────────────────────────────────────
    // Stiffness scales as E·h¹ for 3D elements.
    // Doubling h should double diagonal entries.

    #[test]
    fn ke_scales_linearly_with_voxel_size() {
        let mat = steel();
        let ke1 = compute_ke_base(&mat, 0.0025);
        let ke2 = compute_ke_base(&mat, 0.0050);  // 2× voxel size
        for i in 0..24 {
            let ratio = ke_entry(&ke2, i, i) / ke_entry(&ke1, i, i);
            assert!(
                (ratio - 2.0).abs() < 1e-10,
                "Ke[{i}][{i}] ratio = {ratio:.8}, expected 2.0 (linear in h)"
            );
        }
    }

    // ── Young's modulus scaling ───────────────────────────────────────────────
    // Ke is linear in E. Doubling E must double every entry.

    #[test]
    fn ke_scales_linearly_with_youngs_modulus() {
        let mat1 = Material { young: 210e9, poisson: 0.3 };
        let mat2 = Material { young: 420e9, poisson: 0.3 };
        let ke1 = compute_ke_base(&mat1, 0.0025);
        let ke2 = compute_ke_base(&mat2, 0.0025);
        for i in 0..24 {
            for j in 0..24 {
                let e1 = ke_entry(&ke1, i, j);
                let e2 = ke_entry(&ke2, i, j);
                if e1.abs() > 1e-10 * ke_entry(&ke1, 0, 0).abs() {
                    let ratio = e2 / e1;
                    assert!(
                        (ratio - 2.0).abs() < 1e-10,
                        "Ke[{i}][{j}] ratio={ratio:.8}, expected 2.0"
                    );
                }
            }
        }
    }
}