// src/oc_update.rs
//
// Optimality Criteria (OC) density update with bisection and damping.
//
// The OC update finds the Lagrange multiplier λ that enforces the volume
// constraint, then updates each element density:
//
//   x_oc[e] = clamp(x[e] * sqrt(-dc[e] / λ), x[e]±move, [ρ_min, 1])
//
// Bisection on λ finds the value where total volume = target volume.
// η=0.5 damping (x_new = 0.5*x_oc + 0.5*x_old) breaks the 2-cycle
// oscillation that appears without it.
//
// Note: the handoff §6.7 snippet had `l1` declared without `mut` — fixed here.

use crate::types::{RHO_MIN, SimpConfig};

/// Result of one OC update step.
pub struct OcResult {
    /// Updated density field, length n_elem.
    pub x_new: Vec<f64>,
    /// Max absolute density change max|x_new - x_old|.
    pub rho_change: f64,
    /// Actual volume fraction after update.
    pub vol_frac: f64,
}

/// Apply one OC update step.
///
/// # Arguments
/// * `x`          — current density field
/// * `dc`         — filtered sensitivities (negative values)
/// * `config`     — SIMP configuration (volume_fraction, move_limit, damping)
/// * `void_mask`  — void elements: forced to ρ=0, excluded from volume
/// * `nondesign`  — nondesign elements: forced to ρ=1, excluded from bisection
///
/// Volume constraint is enforced only over design elements
/// (not void, not nondesign).
pub fn oc_update(
    x:          &[f64],
    dc:         &[f64],
    config:     &SimpConfig,
    void_mask:  &[bool],
    nondesign:  &[bool],
) -> OcResult {
    let n_elem = x.len();

    // ── Design mask ───────────────────────────────────────────────────────────
    let design: Vec<bool> = (0..n_elem)
        .map(|e| !void_mask[e] && !nondesign[e])
        .collect();

    // ── OC numerator: x[e] * sqrt(max(-dc[e], 0)) ────────────────────────────
    // Using max(...,0) guards against any numerically positive dc values.
    let ocp: Vec<f64> = (0..n_elem)
        .map(|e| {
            if !design[e] { 0.0 }
            else { x[e] * (-dc[e]).max(0.0).sqrt() }
        })
        .collect();

    // ── Bisection bounds ──────────────────────────────────────────────────────
    // l1=0 always brackets from below (λ→0 gives maximum density everywhere).
    // l2 = ocp_max / rho_min guarantees x_oc ≤ rho_min at all design elements,
    // so volume is at its minimum — brackets from above.
    let ocp_max = ocp.iter().cloned().fold(0.0f64, f64::max);
    let mut l1 = 0.0f64;
    let mut l2 = ocp_max / (RHO_MIN + 1e-30);

    // Target volume = vf * (number of design elements)
    // We use element count as a proxy for volume (uniform grid, cell_vol cancels).
    let n_design = design.iter().filter(|&&b| b).count();
    let target_vol = config.volume_fraction * n_design as f64;

    // ── Bisect for λ ──────────────────────────────────────────────────────────
    let apply_oc = |lm: f64| -> Vec<f64> {
        (0..n_elem).map(|e| {
            if void_mask[e]   { return 0.0; }
            if nondesign[e]   { return 1.0; }
            if lm < 1e-30     { return (x[e] + config.move_limit).min(1.0); }
            let x_oc = (ocp[e] / lm)
                .clamp(x[e] - config.move_limit, x[e] + config.move_limit)
                .clamp(RHO_MIN, 1.0);
            x_oc
        }).collect()
    };

    for _ in 0..200 {
        let lm = 0.5 * (l1 + l2);
        let x_trial = apply_oc(lm);
        let vol: f64 = (0..n_elem)
            .filter(|&e| design[e])
            .map(|e| x_trial[e])
            .sum();
        if vol > target_vol { l1 = lm; } else { l2 = lm; }
        if (l2 - l1) < 1e-9 * (l1 + l2 + 1e-30) { break; }
    }

    // ── Apply final λ with η=0.5 damping ──────────────────────────────────────
    let lm = 0.5 * (l1 + l2);
    let x_oc = apply_oc(lm);
    let x_new: Vec<f64> = (0..n_elem)
        .map(|e| {
            if void_mask[e]  { return 0.0; }
            if nondesign[e]  { return 1.0; }
            config.damping * x_oc[e] + (1.0 - config.damping) * x[e]
        })
        .collect();

    // ── Diagnostics ───────────────────────────────────────────────────────────
    let rho_change = x.iter().zip(x_new.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let vol_frac = if n_design > 0 {
        x_new.iter().enumerate()
            .filter(|&(e, _)| design[e])
            .map(|(_, &v)| v)
            .sum::<f64>() / n_design as f64
    } else { 0.0 };

    OcResult { x_new, rho_change, vol_frac }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SimpConfig;

    fn default_config() -> SimpConfig {
        SimpConfig {
            volume_fraction: 0.5,
            penal: 3.0,
            filter_radius: 0.005,
            max_iterations: 200,
            convergence_tol: 0.002,
            move_limit: 0.2,
            damping: 0.5,
            checkpoint_every: 10,
        }
    }

    // ── Volume constraint ─────────────────────────────────────────────────────

    #[test]
    fn volume_fraction_tracks_target_on_uniform_sensitivity() {
        // Uniform dc → bisection should converge to exactly vf=0.5.
        let n = 100;
        let x   = vec![0.5f64; n];
        let dc  = vec![-1.0f64; n];   // uniform sensitivity
        let cfg = default_config();
        let void_mask  = vec![false; n];
        let nondesign  = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        let err = (res.vol_frac - cfg.volume_fraction).abs();
        assert!(err < 1e-6,
            "vol_frac={:.6}, target={:.6}, err={err:.2e}", res.vol_frac, cfg.volume_fraction);
    }

    #[test]
    fn volume_fraction_trends_toward_target() {
        // Starting below target, each OC step should move vol_frac toward target.
        // Damping (η=0.5) means we won't reach target in one step — that's correct.
        let n = 200;
        let x   = vec![0.3f64; n];
        let dc  = vec![-0.5f64; n];
        let mut cfg = default_config();
        cfg.volume_fraction = 0.4;

        let void_mask = vec![false; n];
        let nondesign = vec![false; n];
        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);

        // vol_frac should be between starting value (0.3) and target (0.4)
        // — damping prevents overshooting or reaching target in one step.
        assert!(res.vol_frac > 0.3,
            "vol_frac={:.4} should be > starting value 0.3", res.vol_frac);
        assert!(res.vol_frac <= 0.4 + 1e-6,
            "vol_frac={:.4} should be <= target 0.4", res.vol_frac);
    }

    // ── Density bounds ────────────────────────────────────────────────────────

    #[test]
    fn all_densities_within_rho_min_to_one() {
        let n = 50;
        let x  = vec![0.5f64; n];
        let dc = (0..n).map(|i| -(i as f64 + 1.0) * 0.01).collect::<Vec<_>>();
        let cfg = default_config();
        let void_mask = vec![false; n];
        let nondesign = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        for (e, &v) in res.x_new.iter().enumerate() {
            assert!(v >= RHO_MIN - 1e-12,
                "x_new[{e}]={v:.6e} < RHO_MIN={RHO_MIN}");
            assert!(v <= 1.0 + 1e-12,
                "x_new[{e}]={v:.6e} > 1.0");
        }
    }

    #[test]
    fn move_limit_is_respected() {
        // x[e] can change by at most move_limit before damping.
        // After damping η=0.5, max change is 0.5 * move_limit.
        // We check against move_limit as a conservative bound.
        let n = 50;
        let x  = vec![0.5f64; n];
        let dc = vec![-1.0f64; n];
        let cfg = default_config();   // move_limit=0.2
        let void_mask = vec![false; n];
        let nondesign = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        for (e, (&x_old, &x_new)) in x.iter().zip(res.x_new.iter()).enumerate() {
            let change = (x_new - x_old).abs();
            assert!(change <= cfg.move_limit + 1e-10,
                "elem {e}: |x_new-x_old|={change:.4e} > move_limit={}", cfg.move_limit);
        }
    }

    // ── Void and nondesign ────────────────────────────────────────────────────

    #[test]
    fn void_elements_are_zero() {
        let n = 20;
        let x  = vec![0.5f64; n];
        let dc = vec![-1.0f64; n];
        let cfg = default_config();
        let mut void_mask = vec![false; n];
        void_mask[0] = true;
        void_mask[5] = true;
        let nondesign = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        assert_eq!(res.x_new[0], 0.0, "void element 0 should be 0");
        assert_eq!(res.x_new[5], 0.0, "void element 5 should be 0");
    }

    #[test]
    fn nondesign_elements_are_one() {
        let n = 20;
        let x  = vec![0.5f64; n];
        let dc = vec![-1.0f64; n];
        let cfg = default_config();
        let void_mask = vec![false; n];
        let mut nondesign = vec![false; n];
        nondesign[3] = true;
        nondesign[7] = true;

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        assert_eq!(res.x_new[3], 1.0, "nondesign element 3 should be 1");
        assert_eq!(res.x_new[7], 1.0, "nondesign element 7 should be 1");
    }

    // ── Sensitivity-driven redistribution ────────────────────────────────────

    #[test]
    fn high_sensitivity_elements_gain_material() {
        // Elements with larger |dc| are more structurally important and
        // should receive more material after the OC update.
        let n = 10;
        let x = vec![0.5f64; n];
        // First 5 elements have 10× higher sensitivity magnitude
        let dc: Vec<f64> = (0..n).map(|i| if i < 5 { -10.0 } else { -1.0 }).collect();
        let cfg = default_config();
        let void_mask = vec![false; n];
        let nondesign = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);

        let avg_high: f64 = res.x_new[0..5].iter().sum::<f64>() / 5.0;
        let avg_low:  f64 = res.x_new[5..10].iter().sum::<f64>() / 5.0;
        assert!(avg_high > avg_low,
            "high-sensitivity avg={avg_high:.4} should > low-sensitivity avg={avg_low:.4}");
    }

    // ── Bisection convergence ─────────────────────────────────────────────────

    #[test]
    fn bisection_converges_when_already_at_target() {
        // If x is already at the target volume fraction with uniform sensitivity,
        // the update should keep it approximately there.
        let n = 100;
        let x   = vec![0.5f64; n];
        let dc  = vec![-1.0f64; n];
        let cfg = default_config();   // target vf = 0.5
        let void_mask = vec![false; n];
        let nondesign = vec![false; n];

        let res = oc_update(&x, &dc, &cfg, &void_mask, &nondesign);
        assert!(res.rho_change < cfg.move_limit,
            "rho_change={:.4e} should be < move_limit={}", res.rho_change, cfg.move_limit);
    }
}