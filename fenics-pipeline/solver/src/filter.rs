// src/filter.rs
//
// Structured-grid sensitivity filter (Sigmund 2001, linear cone weights).
//
// On a uniform hex grid, element centroids are at ((ix+0.5)h, (iy+0.5)h,
// (iz+0.5)h). The filter radius is specified in metres. For each element e,
// we collect all neighbours f whose centroid is within filter_radius, compute
// weight w_ef = filter_radius - dist(e,f), and store (f, w_ef).
//
// The x-weighted filtered sensitivity (KKT-correct form) is:
//   dc_filtered[e] = Σ_f(H_ef · x[f] · dc[f]) / Σ_f(H_ef · x[f])
//
// This is applied in sensitivity.rs. This module only builds the weight table.
//
// No KDTree needed — the structured grid lets us iterate a bounding box of
// ceil(filter_radius / h) cells in each direction and test exact distance.

use crate::types::Grid;

/// Precomputed filter weights for every element.
///
/// `weights[e]`     — Vec of (neighbour_elem_index, weight) pairs for element e.
///                    Always includes e itself (self-weight = filter_radius).
/// `weight_sums[e]` — Σ w_ef over all neighbours (for plain unweighted filter).
pub struct FilterWeights {
    pub weights:      Vec<Vec<(usize, f64)>>,
    pub weight_sums:  Vec<f64>,
}

/// Build the filter weight table for the given grid and radius.
///
/// O(n_elem · (2·r_cells+1)³) — typically fast even for production grids
/// because r_cells is small (filter_radius=8mm, h=2.5mm → r_cells=4 → 9³=729
/// candidates per element).
pub fn build_filter(grid: &Grid, filter_radius: f64) -> FilterWeights {
    let h = grid.voxel_size;
    let r_cells = (filter_radius / h).ceil() as isize;

    let n = grid.n_elem();
    let mut weights:     Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut weight_sums: Vec<f64>               = vec![0.0; n];

    for iz in 0..grid.nz as isize {
    for iy in 0..grid.ny as isize {
    for ix in 0..grid.nx as isize {
        let e = grid.elem_idx(ix as usize, iy as usize, iz as usize);

        for dz in -r_cells..=r_cells {
        for dy in -r_cells..=r_cells {
        for dx in -r_cells..=r_cells {
            let fx = ix + dx;
            let fy = iy + dy;
            let fz = iz + dz;

            if fx < 0 || fy < 0 || fz < 0 { continue; }
            if fx >= grid.nx as isize { continue; }
            if fy >= grid.ny as isize { continue; }
            if fz >= grid.nz as isize { continue; }

            // Centroid-to-centroid distance in metres
            let dist = ((dx*dx + dy*dy + dz*dz) as f64).sqrt() * h;
            if dist >= filter_radius { continue; }

            let w = filter_radius - dist;   // linear cone weight
            let f = grid.elem_idx(fx as usize, fy as usize, fz as usize);
            weights[e].push((f, w));
            weight_sums[e] += w;
        }}}
    }}}

    FilterWeights { weights, weight_sums }
}

/// Apply the plain (unweighted) filter to a scalar field.
///
/// out[e] = Σ_f(H_ef · field[f]) / Σ_f(H_ef)
///
/// Used for filtering the density field itself if needed.
/// The x-weighted sensitivity filter is in sensitivity.rs.
pub fn apply_filter(fw: &FilterWeights, field: &[f64]) -> Vec<f64> {
    let n = field.len();
    assert_eq!(fw.weights.len(), n, "filter/field length mismatch");
    let mut out = vec![0.0f64; n];
    for e in 0..n {
        let mut num = 0.0f64;
        for &(f, w) in &fw.weights[e] {
            num += w * field[f];
        }
        out[e] = num / (fw.weight_sums[e] + 1e-16);
    }
    out
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Grid;

    fn small_grid() -> Grid {
        // 10×10×4: enough cells to have a fully-interior element at radius 2
        Grid { nx: 10, ny: 10, nz: 4, voxel_size: 0.001 }
    }

    // ── Self-weight ───────────────────────────────────────────────────────────

    #[test]
    fn every_element_includes_itself() {
        let g = small_grid();
        let fw = build_filter(&g, 0.0025);
        for e in 0..g.n_elem() {
            let has_self = fw.weights[e].iter().any(|&(f, _)| f == e);
            assert!(has_self, "element {e} missing self-weight");
        }
    }

    #[test]
    fn self_weight_equals_filter_radius() {
        // dist(e,e)=0, so w = filter_radius - 0 = filter_radius
        let g = small_grid();
        let radius = 0.0025;
        let fw = build_filter(&g, radius);
        for e in 0..g.n_elem() {
            let self_w = fw.weights[e].iter()
                .find(|&&(f, _)| f == e)
                .map(|&(_, w)| w)
                .expect("self-weight missing");
            assert!(
                (self_w - radius).abs() < 1e-14,
                "element {e}: self_weight={self_w:.4e}, expected {radius:.4e}"
            );
        }
    }

    // ── Weight positivity and bound ───────────────────────────────────────────

    #[test]
    fn all_weights_are_positive() {
        let g = small_grid();
        let fw = build_filter(&g, 0.003);
        for e in 0..g.n_elem() {
            for &(f, w) in &fw.weights[e] {
                assert!(w > 0.0, "element {e} neighbour {f}: weight {w:.4e} <= 0");
            }
        }
    }

    #[test]
    fn all_weights_at_most_filter_radius() {
        let radius = 0.003;
        let g = small_grid();
        let fw = build_filter(&g, radius);
        for e in 0..g.n_elem() {
            for &(f, w) in &fw.weights[e] {
                assert!(
                    w <= radius + 1e-14,
                    "element {e} neighbour {f}: weight {w:.4e} > radius {radius:.4e}"
                );
            }
        }
    }

    // ── Zero outside radius ───────────────────────────────────────────────────

    #[test]
    fn no_neighbours_outside_radius() {
        let radius = 0.0025;
        let h = 0.001;
        let g = Grid { nx: 10, ny: 10, nz: 10, voxel_size: h };
        let fw = build_filter(&g, radius);

        for iz in 0..g.nz as isize {
        for iy in 0..g.ny as isize {
        for ix in 0..g.nx as isize {
            let e = g.elem_idx(ix as usize, iy as usize, iz as usize);
            for &(f, _) in &fw.weights[e] {
                // Recover neighbour grid coords
                let fz = f / (g.nx * g.ny);
                let rem = f % (g.nx * g.ny);
                let fy = rem / g.nx;
                let fx = rem % g.nx;

                let dx = (ix - fx as isize) as f64;
                let dy = (iy - fy as isize) as f64;
                let dz = (iz - fz as isize) as f64;
                let dist = (dx*dx + dy*dy + dz*dz).sqrt() * h;
                assert!(
                    dist < radius,
                    "element {e} has neighbour {f} at dist {dist:.4e} >= radius {radius:.4e}"
                );
            }
        }}}
    }

    // ── Uniform field is preserved ────────────────────────────────────────────
    // Filtering a constant field must return the same constant exactly.

    #[test]
    fn filter_preserves_uniform_field() {
        let g = small_grid();
        let fw = build_filter(&g, 0.003);
        let field = vec![0.45f64; g.n_elem()];
        let out = apply_filter(&fw, &field);
        for e in 0..g.n_elem() {
            assert!(
                (out[e] - 0.45).abs() < 1e-12,
                "element {e}: filtered={:.6e}, expected 0.45", out[e]
            );
        }
    }

    // ── Weight sum symmetry ───────────────────────────────────────────────────
    // Interior elements (far from boundaries) all have the same neighbourhood
    // and therefore the same weight sum.

    #[test]
    fn interior_elements_have_equal_weight_sums() {
        let radius = 0.002;
        let h = 0.001;
        let g = Grid { nx: 20, ny: 20, nz: 20, voxel_size: h };
        let fw = build_filter(&g, radius);

        // r_cells = ceil(0.002/0.001) = 2, so interior starts at ix=2
        let margin = (radius / h).ceil() as usize;
        let mut interior_sums: Vec<f64> = Vec::new();

        for iz in margin..g.nz-margin {
        for iy in margin..g.ny-margin {
        for ix in margin..g.nx-margin {
            let e = g.elem_idx(ix, iy, iz);
            interior_sums.push(fw.weight_sums[e]);
        }}}

        assert!(!interior_sums.is_empty(), "no interior elements found");
        let first = interior_sums[0];
        for (i, &s) in interior_sums.iter().enumerate() {
            assert!(
                (s - first).abs() < 1e-12,
                "interior element {i}: weight_sum={s:.6e} != {first:.6e}"
            );
        }
    }

    // ── Boundary elements have fewer neighbours ───────────────────────────────

    #[test]
    fn corner_element_has_fewer_neighbours_than_interior() {
        let radius = 0.003;
        let h = 0.001;
        let g = Grid { nx: 20, ny: 20, nz: 20, voxel_size: h };
        let fw = build_filter(&g, radius);

        let corner   = g.elem_idx(0, 0, 0);
        let interior = g.elem_idx(10, 10, 10);

        assert!(
            fw.weights[corner].len() < fw.weights[interior].len(),
            "corner ({}) has >= neighbours as interior ({})",
            fw.weights[corner].len(), fw.weights[interior].len()
        );
    }

    // ── Radius=0 edge case ────────────────────────────────────────────────────
    // A radius smaller than h but > 0 should give only the self-weight.

    #[test]
    fn tiny_radius_gives_only_self_weight() {
        let g = small_grid();
        let h = g.voxel_size;
        // radius = 0.1 * h — smaller than one cell, so only dist=0 qualifies
        let fw = build_filter(&g, 0.1 * h);
        for e in 0..g.n_elem() {
            assert_eq!(
                fw.weights[e].len(), 1,
                "element {e}: expected 1 neighbour (self), got {}", fw.weights[e].len()
            );
            assert_eq!(fw.weights[e][0].0, e, "element {e}: self-weight not pointing to self");
        }
    }
}