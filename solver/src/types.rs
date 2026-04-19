// src/types.rs
//
// Core data types for the SIMP topology optimizer.
//
// Ownership model summary for C programmers:
//   - Structs with `Vec<T>` fields own their heap data. When the struct is
//     dropped, the Vec is freed — no malloc/free needed.
//   - Methods taking `&self` borrow the struct immutably (read-only, like
//     `const MyStruct*` in C). Methods taking `&mut self` borrow mutably.
//   - `#[derive(Debug)]` gives us free `{:?}` printing, like a toString().

// ─── Physical constant ────────────────────────────────────────────────────────

/// Minimum allowable element density.  Prevents singularity in K when an
/// element is "void" in the design space.  Never serialised — always 1e-3.
pub const RHO_MIN: f64 = 1e-3;

// ─── Grid ─────────────────────────────────────────────────────────────────────

/// Structured hexahedral voxel grid.
///
/// Element count:  nx  * ny  * nz
/// Node count:    (nx+1)*(ny+1)*(nz+1)
///
/// Indexing convention (z-major, matching NumPy's default reshape(nz,ny,nx)):
///   node_idx(ix, iy, iz) = ix + iy*(nx+1) + iz*(nx+1)*(ny+1)
///   elem_idx(ix, iy, iz) = ix + iy*nx     + iz*nx*ny
///
/// `voxel_size` is in metres (e.g. 0.0025 = 2.5mm).
#[derive(Debug, Clone)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub voxel_size: f64,
}

impl Grid {
    /// Total number of hex elements.
    #[inline]
    pub fn n_elem(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Total number of nodes (element corners).
    #[inline]
    pub fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
    }

    /// Total degrees of freedom (3 translational DOFs per node).
    #[inline]
    pub fn n_dof(&self) -> usize {
        self.n_nodes() * 3
    }

    /// Global node index from (ix, iy, iz) grid coordinates.
    ///
    /// Bounds: ix in [0, nx], iy in [0, ny], iz in [0, nz].
    /// Panics in debug builds if any coordinate is out of range.
    #[inline]
    pub fn node_idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        debug_assert!(ix <= self.nx, "ix={ix} out of range [0, {}]", self.nx);
        debug_assert!(iy <= self.ny, "iy={iy} out of range [0, {}]", self.ny);
        debug_assert!(iz <= self.nz, "iz={iz} out of range [0, {}]", self.nz);
        ix + iy * (self.nx + 1) + iz * (self.nx + 1) * (self.ny + 1)
    }

    /// Global element index from (ix, iy, iz) element coordinates.
    ///
    /// Bounds: ix in [0, nx-1], iy in [0, ny-1], iz in [0, nz-1].
    #[inline]
    pub fn elem_idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        debug_assert!(ix < self.nx, "ix={ix} out of range [0, {})", self.nx);
        debug_assert!(iy < self.ny, "iy={iy} out of range [0, {})", self.ny);
        debug_assert!(iz < self.nz, "iz={iz} out of range [0, {})", self.nz);
        ix + iy * self.nx + iz * self.nx * self.ny
    }

    /// Physical size of the domain in metres: (Lx, Ly, Lz).
    #[inline]
    pub fn domain_size(&self) -> (f64, f64, f64) {
        (
            self.nx as f64 * self.voxel_size,
            self.ny as f64 * self.voxel_size,
            self.nz as f64 * self.voxel_size,
        )
    }

    /// Element centroid in metres for element (ix, iy, iz).
    #[inline]
    pub fn centroid(&self, ix: usize, iy: usize, iz: usize) -> (f64, f64, f64) {
        let h = self.voxel_size;
        (
            (ix as f64 + 0.5) * h,
            (iy as f64 + 0.5) * h,
            (iz as f64 + 0.5) * h,
        )
    }
}

// ─── Material ─────────────────────────────────────────────────────────────────

/// Isotropic linear elastic material.
///
/// `young`   — Young's modulus in Pascals (e.g. 210e9 for steel).
/// `poisson` — Poisson's ratio (dimensionless, must be in (-1, 0.5)).
#[derive(Debug, Clone)]
pub struct Material {
    pub young: f64,
    pub poisson: f64,
}

impl Material {
    /// First Lamé parameter λ = Eν / ((1+ν)(1−2ν)).
    ///
    /// Measures resistance to volumetric change.
    #[inline]
    pub fn lame_lambda(&self) -> f64 {
        let e = self.young;
        let v = self.poisson;
        e * v / ((1.0 + v) * (1.0 - 2.0 * v))
    }

    /// Second Lamé parameter μ (shear modulus) = E / (2(1+ν)).
    #[inline]
    pub fn lame_mu(&self) -> f64 {
        self.young / (2.0 * (1.0 + self.poisson))
    }
}

// ─── LoadCase ─────────────────────────────────────────────────────────────────

/// Boundary conditions for one load case.
///
/// All indices are *global* DOF indices: DOF = 3*node + axis (0=x, 1=y, 2=z).
/// `fixed_dofs` are clamped to zero displacement (Dirichlet BCs).
/// `load_dofs`/`load_vals` are paired — load_vals[i] Newtons at load_dofs[i].
#[derive(Debug)]
pub struct LoadCase {
    pub fixed_dofs: Vec<usize>,
    pub load_dofs: Vec<usize>,
    pub load_vals: Vec<f64>,
}

impl LoadCase {
    /// Validate that load_dofs and load_vals have matching length.
    /// Call after construction from binary files.
    pub fn validate(&self) -> Result<(), String> {
        if self.load_dofs.len() != self.load_vals.len() {
            return Err(format!(
                "load_dofs length {} != load_vals length {}",
                self.load_dofs.len(),
                self.load_vals.len()
            ));
        }
        Ok(())
    }

    /// Total applied force magnitude (sum of absolute values).
    pub fn total_force(&self) -> f64 {
        self.load_vals.iter().map(|v| v.abs()).sum()
    }
}

/// Solver configuration — all SIMP algorithm parameters.
///
/// `volume_fraction`       — target fraction of solid material (0 < vf < 1).
/// `penal`                 — penalization exponent p (typically 3.0).
/// `filter_radius`         — sensitivity filter radius in metres.
/// `max_iterations`        — hard stop if convergence not reached.
/// `min_iterations`        — minimum iterations before convergence checks fire.
///                           Default 10. Set higher (e.g. 50) for stage-2 runs
///                           that start on a plateau and would otherwise
///                           terminate prematurely.
/// `convergence_tol`       — legacy single tolerance, now a fallback.
///                           If `compliance_spread_tol` / `density_change_tol`
///                           are None, both checks use this value.
/// `compliance_spread_tol` — optional separate threshold for the rolling
///                           10-iteration compliance spread check.
///                           When Some(x), overrides `convergence_tol`
///                           for compliance-flatness detection.
/// `density_change_tol`    — optional separate threshold for the per-iteration
///                           density-change (Δρ) check.
///                           When Some(x), overrides `convergence_tol`
///                           for density-change detection.
/// `move_limit`            — OC update step size (fraction of density range).
/// `damping`               — OC damping η; 0.5 breaks 2-cycle oscillation.
/// `checkpoint_every`      — write density.bin every N iterations (0 = off).
#[derive(Debug, Clone)]
pub struct SimpConfig {
    pub volume_fraction: f64,
    pub penal: f64,
    pub filter_radius: f64,
    pub max_iterations: usize,
    pub min_iterations: usize,
    pub convergence_tol: f64,
    pub compliance_spread_tol: Option<f64>,
    pub density_change_tol: Option<f64>,
    pub move_limit: f64,
    pub damping: f64,
    pub checkpoint_every: usize,
}

impl SimpConfig {
    /// Effective spread tolerance: overridden value if set, else `convergence_tol`.
    #[inline]
    pub fn spread_tol(&self) -> f64 {
        self.compliance_spread_tol.unwrap_or(self.convergence_tol)
    }

    /// Effective density-change tolerance: overridden value if set, else `convergence_tol`.
    #[inline]
    pub fn density_tol(&self) -> f64 {
        self.density_change_tol.unwrap_or(self.convergence_tol)
    }
}

impl SimpConfig {
    /// Minimum element density — prevents K singularity on void elements.
    /// Not a user parameter; always 1e-3.
    #[inline]
    pub fn rho_min(&self) -> f64 {
        RHO_MIN
    }

    /// Sanity-check config values before starting a solve.
    /// Returns the first error found, or Ok(()) if all pass.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 < self.volume_fraction && self.volume_fraction < 1.0) {
            return Err(format!(
                "volume_fraction={} must be in (0, 1)",
                self.volume_fraction
            ));
        }
        if self.penal < 1.0 {
            return Err(format!("penal={} must be >= 1.0", self.penal));
        }
        if self.filter_radius <= 0.0 {
            return Err(format!(
                "filter_radius={} must be positive",
                self.filter_radius
            ));
        }
        if !(0.0 < self.damping && self.damping <= 1.0) {
            return Err(format!(
                "damping={} must be in (0, 1]",
                self.damping
            ));
        }
        if self.move_limit <= 0.0 || self.move_limit > 1.0 {
            return Err(format!(
                "move_limit={} must be in (0, 1]",
                self.move_limit
            ));
        }
        Ok(())
    }
}

// ─── Problem ──────────────────────────────────────────────────────────────────

/// Everything the solver needs to run a complete SIMP solve.
///
/// `nondesign[e]` — element is forced solid regardless of optimization.
/// `void_mask[e]` — element is always void; not assembled into K.
///  void takes priority over nondesign when both are true.
/// `x_init`       — optional warm-start density field; None → uniform vf.
#[derive(Debug)]
pub struct Problem {
    pub grid: Grid,
    pub material: Material,
    pub load_case: LoadCase,
    pub config: SimpConfig,
    pub nondesign: Vec<bool>,
    pub void_mask: Vec<bool>,
    pub x_init: Option<Vec<f64>>,
}

impl Problem {
    /// Validate all sub-fields and check mask lengths match n_elem.
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.load_case.validate()?;

        let n = self.grid.n_elem();
        if self.nondesign.len() != n {
            return Err(format!(
                "nondesign length {} != n_elem {}",
                self.nondesign.len(),
                n
            ));
        }
        if self.void_mask.len() != n {
            return Err(format!(
                "void_mask length {} != n_elem {}",
                self.void_mask.len(),
                n
            ));
        }
        if let Some(ref xi) = self.x_init {
            if xi.len() != n {
                return Err(format!(
                    "x_init length {} != n_elem {}",
                    xi.len(),
                    n
                ));
            }
            // Allow 0.0 for void elements — clamped to RHO_MIN in run_simp()
            for (i, &v) in xi.iter().enumerate() {
                if !(0.0..=1.0).contains(&v) {
                    return Err(format!(
                        "x_init[{}]={} out of range [0.0, 1.0]",
                        i, v, 
                    ));
                }
            }
        }
        Ok(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // The canonical development grid from the handoff (§11).
    fn dev_grid() -> Grid {
        Grid { nx: 10, ny: 6, nz: 4, voxel_size: 0.0025 }
    }

    // Steel from §5.1.
    fn steel() -> Material {
        Material { young: 210e9, poisson: 0.3 }
    }

    // ── Grid: counts ──────────────────────────────────────────────────────────

    #[test]
    fn grid_n_elem() {
        // 10×6×4 = 240 elements
        assert_eq!(dev_grid().n_elem(), 240);
    }

    #[test]
    fn grid_n_nodes() {
        // 11×7×5 = 385 nodes
        assert_eq!(dev_grid().n_nodes(), 385);
    }

    #[test]
    fn grid_n_dof() {
        // 385 × 3 = 1155 DOFs
        assert_eq!(dev_grid().n_dof(), 1155);
    }

    // ── Grid: node_idx corners ────────────────────────────────────────────────

    #[test]
    fn node_idx_origin_is_zero() {
        assert_eq!(dev_grid().node_idx(0, 0, 0), 0);
    }

    #[test]
    fn node_idx_max_corner_is_n_nodes_minus_one() {
        let g = dev_grid();
        assert_eq!(g.node_idx(g.nx, g.ny, g.nz), g.n_nodes() - 1);
    }

    #[test]
    fn node_idx_x_stride_is_one() {
        // Stepping ix by 1 should step the flat index by exactly 1.
        let g = dev_grid();
        assert_eq!(g.node_idx(1, 0, 0) - g.node_idx(0, 0, 0), 1);
        assert_eq!(g.node_idx(5, 3, 2) - g.node_idx(4, 3, 2), 1);
    }

    #[test]
    fn node_idx_y_stride_is_nx_plus_one() {
        let g = dev_grid();
        let stride = g.nx + 1; // 11
        assert_eq!(g.node_idx(0, 1, 0) - g.node_idx(0, 0, 0), stride);
        assert_eq!(g.node_idx(3, 4, 1) - g.node_idx(3, 3, 1), stride);
    }

    #[test]
    fn node_idx_z_stride_is_nx_plus_one_times_ny_plus_one() {
        let g = dev_grid();
        let stride = (g.nx + 1) * (g.ny + 1); // 11*7 = 77
        assert_eq!(g.node_idx(0, 0, 1) - g.node_idx(0, 0, 0), stride);
        assert_eq!(g.node_idx(2, 3, 3) - g.node_idx(2, 3, 2), stride);
    }

    #[test]
    fn node_idx_known_interior_value() {
        // node(1,1,1) = 1 + 1*11 + 1*77 = 89
        let g = dev_grid();
        assert_eq!(g.node_idx(1, 1, 1), 89);
    }

    // ── Grid: elem_idx corners ────────────────────────────────────────────────

    #[test]
    fn elem_idx_origin_is_zero() {
        assert_eq!(dev_grid().elem_idx(0, 0, 0), 0);
    }

    #[test]
    fn elem_idx_max_corner_is_n_elem_minus_one() {
        let g = dev_grid();
        assert_eq!(g.elem_idx(g.nx - 1, g.ny - 1, g.nz - 1), g.n_elem() - 1);
    }

    #[test]
    fn elem_idx_x_stride_is_one() {
        let g = dev_grid();
        assert_eq!(g.elem_idx(1, 0, 0) - g.elem_idx(0, 0, 0), 1);
    }

    #[test]
    fn elem_idx_y_stride_is_nx() {
        let g = dev_grid();
        assert_eq!(g.elem_idx(0, 1, 0) - g.elem_idx(0, 0, 0), g.nx);
    }

    #[test]
    fn elem_idx_z_stride_is_nx_times_ny() {
        let g = dev_grid();
        assert_eq!(g.elem_idx(0, 0, 1) - g.elem_idx(0, 0, 0), g.nx * g.ny);
    }

    // ── Grid: node_idx and elem_idx are injective (no collisions) ────────────

    #[test]
    fn node_idx_no_collisions() {
        // Use a small grid to keep test fast.
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let mut seen = std::collections::HashSet::new();
        for iz in 0..=g.nz {
            for iy in 0..=g.ny {
                for ix in 0..=g.nx {
                    let idx = g.node_idx(ix, iy, iz);
                    assert!(seen.insert(idx), "collision at ({ix},{iy},{iz}): idx={idx}");
                }
            }
        }
        assert_eq!(seen.len(), g.n_nodes());
    }

    #[test]
    fn elem_idx_no_collisions() {
        let g = Grid { nx: 4, ny: 3, nz: 2, voxel_size: 0.001 };
        let mut seen = std::collections::HashSet::new();
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let idx = g.elem_idx(ix, iy, iz);
                    assert!(seen.insert(idx), "collision at ({ix},{iy},{iz}): idx={idx}");
                }
            }
        }
        assert_eq!(seen.len(), g.n_elem());
    }

    // ── Grid: centroid ────────────────────────────────────────────────────────

    #[test]
    fn centroid_origin_element() {
        let g = dev_grid();
        let h = g.voxel_size;
        let (cx, cy, cz) = g.centroid(0, 0, 0);
        let eps = 1e-15;
        assert!((cx - 0.5 * h).abs() < eps);
        assert!((cy - 0.5 * h).abs() < eps);
        assert!((cz - 0.5 * h).abs() < eps);
    }

    #[test]
    fn domain_size() {
        let g = dev_grid();
        let (lx, ly, lz) = g.domain_size();
        let eps = 1e-12;
        assert!((lx - 0.025).abs() < eps, "Lx expected 0.025m, got {lx}");
        assert!((ly - 0.015).abs() < eps, "Ly expected 0.015m, got {ly}");
        assert!((lz - 0.010).abs() < eps, "Lz expected 0.010m, got {lz}");
    }

    // ── Material: Lamé parameters ─────────────────────────────────────────────

    #[test]
    fn lame_lambda_steel() {
        // λ = 210e9 * 0.3 / (1.3 * 0.4) = 63e9 / 0.52
        let expected = 210e9_f64 * 0.3 / (1.3 * 0.4);
        let got = steel().lame_lambda();
        let rel_err = (got - expected).abs() / expected;
        assert!(rel_err < 1e-12, "λ rel_err={rel_err:.2e}, got={got:.6e}, expected={expected:.6e}");
    }

    #[test]
    fn lame_mu_steel() {
        // μ = 210e9 / (2 * 1.3)
        let expected = 210e9_f64 / 2.6;
        let got = steel().lame_mu();
        let rel_err = (got - expected).abs() / expected;
        assert!(rel_err < 1e-12, "μ rel_err={rel_err:.2e}, got={got:.6e}, expected={expected:.6e}");
    }

    #[test]
    fn lame_mu_steel_known_value() {
        // Textbook value for steel μ ≈ 80.769 GPa
        let mu = steel().lame_mu();
        assert!((mu - 80.769_230_769e9).abs() < 1e3, "μ={mu:.3e}");
    }

    #[test]
    fn lame_lambda_steel_known_value() {
        // Textbook value λ ≈ 121.154 GPa
        let lambda = steel().lame_lambda();
        assert!((lambda - 121.153_846_154e9).abs() < 1e3, "λ={lambda:.3e}");
    }

    // ── LoadCase: validate ────────────────────────────────────────────────────

    #[test]
    fn loadcase_validate_ok_when_lengths_match() {
        let lc = LoadCase {
            fixed_dofs: vec![0, 3, 6],
            load_dofs: vec![100, 101],
            load_vals: vec![-5000.0, -5000.0],
        };
        assert!(lc.validate().is_ok());
    }

    #[test]
    fn loadcase_validate_err_on_length_mismatch() {
        let lc = LoadCase {
            fixed_dofs: vec![0],
            load_dofs: vec![100, 101],
            load_vals: vec![-5000.0],    // length mismatch
        };
        assert!(lc.validate().is_err());
    }

    #[test]
    fn loadcase_total_force() {
        let lc = LoadCase {
            fixed_dofs: vec![],
            load_dofs: vec![0, 1, 2],
            load_vals: vec![-3000.0, -3000.0, -4000.0],
        };
        assert!((lc.total_force() - 10000.0).abs() < 1e-9);
    }

    // ── SimpConfig: validate ──────────────────────────────────────────────────

    fn default_config() -> SimpConfig {
        SimpConfig {
            volume_fraction: 0.45,
            penal: 3.0,
            filter_radius: 0.008,
            max_iterations: 200,
            min_iterations: 10,
            convergence_tol: 0.002,
            compliance_spread_tol: None,
            density_change_tol: None,
            move_limit: 0.05,
            damping: 0.5,
            checkpoint_every: 10,
        }
    }

    #[test]
    fn config_validate_baseline_ok() {
        assert!(default_config().validate().is_ok());
    }

    #[test]
    fn config_validate_rejects_bad_volume_fraction() {
        let mut c = default_config();
        c.volume_fraction = 1.1;
        assert!(c.validate().is_err());
        c.volume_fraction = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_bad_penal() {
        let mut c = default_config();
        c.penal = 0.5;
        assert!(c.validate().is_err());
    }

    #[test]
    fn config_rho_min_is_constant() {
        assert_eq!(default_config().rho_min(), RHO_MIN);
        assert_eq!(RHO_MIN, 1e-3);
    }

    // ── Problem: validate ─────────────────────────────────────────────────────

    #[test]
    fn problem_validate_ok_on_clean_problem() {
        let g = dev_grid();
        let n = g.n_elem();
        let p = Problem {
            grid: g,
            material: steel(),
            load_case: LoadCase {
                fixed_dofs: vec![0, 1, 2],
                load_dofs: vec![1000],
                load_vals: vec![-10000.0],
            },
            config: default_config(),
            nondesign: vec![false; n],
            void_mask: vec![false; n],
            x_init: None,
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn problem_validate_catches_mask_length_mismatch() {
        let g = dev_grid();
        let n = g.n_elem();
        let p = Problem {
            grid: g,
            material: steel(),
            load_case: LoadCase {
                fixed_dofs: vec![0],
                load_dofs: vec![100],
                load_vals: vec![-1.0],
            },
            config: default_config(),
            nondesign: vec![false; n + 1],   // wrong length
            void_mask: vec![false; n],
            x_init: None,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn problem_validate_catches_x_init_out_of_range() {
        let g = dev_grid();
        let n = g.n_elem();
        let mut xi = vec![0.45f64; n];
        xi[42] = 1.5;   // out of range
        let p = Problem {
            grid: g,
            material: steel(),
            load_case: LoadCase {
                fixed_dofs: vec![0],
                load_dofs: vec![100],
                load_vals: vec![-1.0],
            },
            config: default_config(),
            nondesign: vec![false; n],
            void_mask: vec![false; n],
            x_init: Some(xi),
        };
        assert!(p.validate().is_err());
    }

    // ── SimpConfig: spread_tol / density_tol helpers ──────────────────────────

    #[test]
    fn spread_tol_falls_back_to_convergence_tol() {
        let c = default_config();
        assert_eq!(c.spread_tol(), c.convergence_tol);
        assert_eq!(c.density_tol(), c.convergence_tol);
    }

    #[test]
    fn spread_tol_uses_override_when_set() {
        let mut c = default_config();
        c.compliance_spread_tol = Some(1e-5);
        assert_eq!(c.spread_tol(), 1e-5);
        // Density tolerance still falls through to convergence_tol
        assert_eq!(c.density_tol(), c.convergence_tol);
    }

    #[test]
    fn density_tol_uses_override_when_set() {
        let mut c = default_config();
        c.density_change_tol = Some(5e-4);
        assert_eq!(c.density_tol(), 5e-4);
        // Spread tolerance still falls through to convergence_tol
        assert_eq!(c.spread_tol(), c.convergence_tol);
    }

    #[test]
    fn both_overrides_independent() {
        let mut c = default_config();
        c.compliance_spread_tol = Some(1e-5);
        c.density_change_tol    = Some(5e-4);
        c.convergence_tol       = 999.0;      // both ignore this
        assert_eq!(c.spread_tol(),  1e-5);
        assert_eq!(c.density_tol(), 5e-4);
    }
}