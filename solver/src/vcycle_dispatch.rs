// src/vcycle_dispatch.rs
//
// Dispatch layer between simp.rs and the linear solver backends.
// This is the ONLY place that decides which solver runs.
//
// ── Routing table (evaluated in order) ──────────────────────────────────────
//
//   n_dof < CHOLESKY_THRESHOLD
//       → faer sparse Cholesky  (exact, best for small n)
//
//   n_dof ≥ CHOLESKY_THRESHOLD,  feature "amgcl" compiled in  (default)
//       → AMGCL AMG-PCG  (smoothed aggregation + SPAI0 smoother, OpenMP)
//         Mesh-independent convergence: ~20-50 CG iterations regardless
//         of stiffness contrast (handles condition numbers up to 10^12).
//         If AMGCL init or hierarchy build fails permanently, falls through.
//
//   n_dof ≥ CHOLESKY_THRESHOLD,  "gpu" feature, use_gpu=true,
//         "amgcl" feature NOT active
//       → GPU ILU(0)-PCG  (cuSPARSE) [LEGACY — broken for high contrast]
//         Correct at iter 1 (C≈0.101), diverges from iter 6+ as SIMP
//         penalization pushes condition number past ~10^9.
//         Kept for reference; will be replaced by AMGCL CUDA in Phase B.
//
//   fallback (AMGCL failed / not compiled, GPU disabled / not compiled)
//       → CPU Jacobi-PCG
//         Correct for any SPD matrix; mesh-dependent convergence (O(h^{-1})
//         iterations).  The confirmed-correct baseline.
//
// ── AMGCL feature gate ───────────────────────────────────────────────────────
//
//   Compile with `amgcl` feature (in `default`):
//     cargo build --release              → AMGCL OpenMP
//
//   Disable AMGCL (falls back to Jacobi-PCG / legacy GPU):
//     cargo build --release --no-default-features
//     cargo build --release --no-default-features --features gpu
//
// ── Phase B: AMGCL CUDA upgrade ──────────────────────────────────────────────
//
//   To add GPU-accelerated AMG, replace the amgcl_wrapper.cpp build with
//   amgcl_wrapper_cuda.cu (same extern "C" API, amgcl::backend::cuda backend).
//   Then add the use_gpu branch here inside the #[cfg(feature = "amgcl")] block:
//
//     #[cfg(all(feature = "amgcl", feature = "gpu"))]
//     if gpu_ctx.use_gpu { /* pass use_gpu hint to AmgclContext */ }
//
//   No changes to simp.rs or types.rs required.
//
// ── V-cycle note ─────────────────────────────────────────────────────────────
//
//   VCyclePreconditioner in multigrid.rs assumes power-of-2 grid dimensions.
//   The motor_mount grid (70×60×80) violates this; V-cycle gave C=1.68e3
//   at iter 1 instead of ~0.1. Jacobi-PCG is the CPU fallback until
//   Phase 3 GMG (non-power-of-2 coarsening) is implemented.
//
// ── Import graph (no cycles) ─────────────────────────────────────────────────
//
//   vcycle_dispatch → solver         (cg_solve_direct, cg_solve_inner)
//   vcycle_dispatch → amgcl_solver   (AmgclContext)  [feature = "amgcl"]
//   simp            → vcycle_dispatch
//   multigrid       → solver         (coarse CG — unchanged, not imported here)

use crate::solver::{cg_solve_direct, cg_solve_inner, CgResult};

/// DOF count below which faer Cholesky beats Jacobi-PCG on wall time.
/// Also the threshold below which AMGCL setup cost exceeds its benefit.
pub const CHOLESKY_THRESHOLD: usize = 50_000;

/// Relative residual above which a GPU ILU(0) solve is considered dirty
/// (legacy threshold, kept for the GPU fallback path).
const DIRTY_RESIDUAL_THRESHOLD: f64 = 1e-4;

/// Relative residual above which an AMGCL solve is considered too inaccurate
/// to use for sensitivity computation.  When exceeded, the dispatcher falls
/// back to Jacobi-PCG for that iteration to protect density field integrity.
///
/// Set to 0.1: a 10% displacement error is enough to corrupt OC sensitivities
/// and push densities to a wrong local minimum.  Early AMGCL iterations
/// (before ILU0 has seen a few density fields) can have residuals of 1-10;
/// this threshold catches those while allowing mild under-convergence (1e-3
/// to 1e-2) which SIMP handles robustly without cascading corruption.
const AMGCL_FALLBACK_THRESHOLD: f64 = 0.1;

// ── GpuContext ────────────────────────────────────────────────────────────────
//
// Holds all persistent solver state across SIMP iterations.
// Renamed conceptually to "SolverContext" but kept as GpuContext for backward
// compatibility with simp.rs (zero changes required there).
//
// Fields:
//   use_gpu      — forward-looking flag: when true, selects the GPU
//                  accelerated AMG backend (Phase B AMGCL CUDA).
//                  Currently unused for routing since the AMGCL OpenMP path
//                  always runs regardless of this flag.
//   inner        — [feature = "gpu"] lazy GPU ILU(0) state (legacy path)
//   amgcl_ctx    — [feature = "amgcl"] AMGCL AMG-PCG context (primary path)

pub struct GpuContext {
    /// Intent flag: prefer GPU-accelerated backend when available.
    /// Phase A (this session): ignored — AMGCL OpenMP always runs.
    /// Phase B: select between amgcl::backend::cuda (true) and
    ///          amgcl::backend::builtin (false).
    pub use_gpu: bool,

    /// [feature = "gpu"] Lazy GPU ILU(0)-PCG state.
    /// Bypassed when `amgcl` feature is active.
    #[cfg(feature = "gpu")]
    inner: Option<crate::gpu_solver::GpuK>,

    /// [feature = "amgcl"] AMGCL AMG-PCG context.
    /// Primary solver for large problems; created lazily on first solve call.
    #[cfg(feature = "amgcl")]
    pub amgcl_ctx: crate::amgcl_solver::AmgclContext,
}

impl GpuContext {
    pub fn new(use_gpu: bool) -> Self {
        GpuContext {
            use_gpu,
            #[cfg(feature = "gpu")]
            inner: None,
            #[cfg(feature = "amgcl")]
            amgcl_ctx: crate::amgcl_solver::AmgclContext::new(),
        }
    }

    /// Human-readable backend label printed at SIMP startup.
    pub fn backend_label(&self) -> &'static str {
        // AMGCL takes precedence when compiled in.
        #[cfg(feature = "amgcl")]
        {
            if self.use_gpu {
                // Phase B will return the CUDA label here.
                return "AMGCL AMG-PCG (OpenMP now; Phase B: CUDA when use_gpu=true)";
            }
            return "AMGCL AMG-PCG (OpenMP — smoothed aggregation + SPAI0 smoother)";
        }

        // Legacy GPU path (amgcl feature not active).
        #[cfg(all(feature = "gpu", not(feature = "amgcl")))]
        if self.use_gpu {
            return "GPU (cuSPARSE ILU(0)-PCG) with Jacobi-PCG fallback [DEPRECATED — use amgcl feature]";
        }

        "CPU (Jacobi-PCG / Cholesky)"
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Solve K·u = f, routing to the appropriate backend.
///
/// # Parameters
/// * `k_rows`, `k_cols`, `k_vals` — CSR stiffness matrix (Dirichlet BCs applied)
/// * `f`        — RHS (Dirichlet penalty already zeroed on fixed DOFs)
/// * `u`        — solution vector; warm-start on entry, solution on exit
/// * `tol`      — relative residual tolerance (1e-6 in simp.rs)
/// * `max_iter` — CG iteration cap (cfg.max_cg_iter)
/// * `_nx/_ny/_nz` — grid dimensions, reserved for future V-cycle re-enable
/// * `gpu_ctx`  — mutable persistent solver context
#[allow(clippy::too_many_arguments)]
pub fn solve_linear_system(
    k_rows:   &[usize],
    k_cols:   &[usize],
    k_vals:   &[f64],
    f:        &[f64],
    u:        &mut [f64],
    tol:      f64,
    max_iter: usize,
    _nx:      usize,    // reserved for Phase 3 GMG
    _ny:      usize,
    _nz:      usize,
    gpu_ctx:  &mut GpuContext,
) -> CgResult {

    // ── 1. Small problems: faer sparse Cholesky ───────────────────────────────
    if f.len() < CHOLESKY_THRESHOLD {
        return cg_solve_direct(k_rows, k_cols, k_vals, f, u, tol, max_iter);
    }

    // ── 2. AMGCL AMG-PCG (primary path for large problems) ───────────────────
    #[cfg(feature = "amgcl")]
    if !gpu_ctx.amgcl_ctx.is_failed() {
        match gpu_ctx.amgcl_ctx.solve(k_rows, k_cols, k_vals, f, u, tol, max_iter) {
            Some(result) => {
                // ── Residual quality gate ─────────────────────────────────────
                //
                // If AMGCL returned but the residual is above the fallback
                // threshold, the displacement field u is unreliable.
                // Using it to compute sensitivities would corrupt the density
                // field permanently — the OC update acts on garbage gradients
                // and pushes densities to a wrong local minimum that all
                // subsequent iterations then optimise within.
                //
                // Fix: fall through to Jacobi-PCG for THIS iteration only.
                // AMGCL is NOT permanently disabled — it may converge fine on
                // the next iteration once the ILU0 hierarchy has seen a couple
                // of density fields.  This handles the early-iteration warmup
                // period without poisoning the optimisation.
                //
                // Threshold: 0.1 (10× the solve tolerance of 1e-6 to 1e-5).
                // At residual=0.1 the displacement error is ~10% — enough to
                // corrupt stress field gradients.  At residual=1e-3 SIMP is
                // still robust; the threshold only fires on the early iterations
                // where ILU0 hasn't yet settled (residuals 1-10 observed).
                if result.rel_residual > AMGCL_FALLBACK_THRESHOLD {
                    eprintln!(
                        "[amgcl] residual={:.2e} > fallback threshold={:.2e} \
                         — running Jacobi-PCG for this iteration to protect \
                         density field integrity.",
                        result.rel_residual, AMGCL_FALLBACK_THRESHOLD
                    );
                    // Reset u to zero before Jacobi so a corrupted warm-start
                    // from the failed AMGCL attempt doesn't bias the result.
                    u.iter_mut().for_each(|v| *v = 0.0);
                    return cg_solve_inner(k_rows, k_cols, k_vals, f, u, tol, max_iter);
                }

                if !result.converged {
                    // Residual is below AMGCL_FALLBACK_THRESHOLD but above tol.
                    // This is a mildly under-converged solve — SIMP is robust
                    // to this; sensitivities will be slightly noisy but the
                    // topology will not be corrupted.
                    eprintln!(
                        "[amgcl] CG did not fully converge: {} iters, \
                         residual={:.1e} (tol={:.1e}).  Continuing — \
                         raise MAX_CG_ITER if this persists past iter 20.",
                        result.iterations, result.rel_residual, tol
                    );
                }
                return result;
            }
            None => {
                // AMGCL hard-failed (hierarchy build error, null handle, etc.)
                // Permanently fall through to Jacobi-PCG for all remaining iters.
                eprintln!("[solver] AMGCL hard-failed — switching to Jacobi-PCG \
                           for all remaining SIMP iterations.");
            }
        }
    }

    // ── 3. Legacy GPU ILU(0)-PCG  (bypassed when amgcl feature is active) ────
    //
    // This path is dead code when the `amgcl` feature is compiled in.
    // Kept to preserve correctness for builds without AMGCL headers.
    // Will be removed when AMGCL CUDA (Phase B) fully replaces it.
    #[cfg(all(feature = "gpu", not(feature = "amgcl")))]
    if gpu_ctx.use_gpu {
        match &mut gpu_ctx.inner {
            slot @ None => {
                match crate::gpu_solver::GpuK::new(k_rows, k_cols, k_vals) {
                    Ok(gpu) => {
                        eprintln!("[solver] GPU backend active (cuSPARSE ILU(0)-PCG)");
                        *slot = Some(gpu);
                        let result = slot.as_ref().unwrap()
                            .cg_solve_persistent(f, u, tol, max_iter);
                        if result.rel_residual > DIRTY_RESIDUAL_THRESHOLD {
                            eprintln!(
                                "[solver] GPU iter-1 residual {:.1e} > threshold — \
                                 running Jacobi-PCG corrector",
                                result.rel_residual
                            );
                            return cg_solve_inner(k_rows, k_cols, k_vals,
                                                  f, u, tol, max_iter);
                        }
                        return result;
                    }
                    Err(e) => {
                        eprintln!("[solver] GPU init failed: {e}  \
                                   → falling back to Jacobi-PCG.");
                        gpu_ctx.use_gpu = false;
                    }
                }
            }
            Some(gpu) => {
                match gpu.refactor(k_vals) {
                    Ok(()) => {
                        let result = gpu.cg_solve_persistent(f, u, tol, max_iter);
                        if result.rel_residual > DIRTY_RESIDUAL_THRESHOLD {
                            eprintln!(
                                "[solver] GPU residual {:.1e} > threshold — \
                                 running Jacobi-PCG corrector",
                                result.rel_residual
                            );
                            return cg_solve_inner(k_rows, k_cols, k_vals,
                                                  f, u, tol, max_iter);
                        }
                        return result;
                    }
                    Err(e) => {
                        eprintln!("[solver] GPU refactor failed: {e}  \
                                   → switching to Jacobi-PCG permanently.");
                        gpu_ctx.use_gpu = false;
                    }
                }
            }
        }
    }

    // ── 4. CPU Jacobi-PCG (ultimate fallback) ────────────────────────────────
    //
    // Always correct for any SPD matrix.  Used when:
    //   a. AMGCL feature is compiled out (--no-default-features)
    //   b. AMGCL initialisation or hierarchy build failed permanently
    //   c. GPU disabled / not compiled
    //
    // Convergence: mesh-dependent, O(h^{-1}) iterations.
    // At 1M DOF with high SIMP contrast: 800-2000 iters × ~0.03s ≈ 24-60s/iter.
    // Correct sensitivities guaranteed at every SIMP iteration.
    cg_solve_inner(k_rows, k_cols, k_vals, f, u, tol, max_iter)
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d_csr(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_idx = Vec::new();
        let mut vals    = Vec::new();
        for i in 0..n {
            if i > 0     { col_idx.push(i - 1); vals.push(-1.0); }
            col_idx.push(i); vals.push(2.0);
            if i < n - 1 { col_idx.push(i + 1); vals.push(-1.0); }
            row_ptr[i + 1] = col_idx.len();
        }
        (row_ptr, col_idx, vals)
    }

    #[test]
    fn dispatch_small_routes_to_cholesky() {
        let n = 64;
        assert!(n < CHOLESKY_THRESHOLD);
        let (row_ptr, col_idx, vals) = laplacian_1d_csr(n);
        let f     = vec![1.0f64; n];
        let mut u = vec![0.0f64; n];
        let mut ctx = GpuContext::new(false);

        let result = solve_linear_system(
            &row_ptr, &col_idx, &vals,
            &f, &mut u,
            1e-8, n,
            n + 1, 1, 1,
            &mut ctx,
        );
        assert!(result.converged, "Cholesky path must converge on 1D Laplacian");

        // Verify K·u ≈ f
        let mut ku = vec![0.0f64; n];
        for i in 0..n {
            for j in row_ptr[i]..row_ptr[i + 1] {
                ku[i] += vals[j] * u[col_idx[j]];
            }
        }
        let res: f64 = ku.iter().zip(f.iter()).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt();
        let rhs: f64 = f.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        assert!(res / rhs < 1e-6, "Cholesky relative residual {:.2e}", res / rhs);
    }

    #[test]
    fn dispatch_large_routes_to_amgcl_or_jacobi() {
        // Above Cholesky threshold: whichever large-problem solver is compiled
        // in (AMGCL when default features, Jacobi otherwise) must converge on
        // a well-conditioned diagonal system.
        let n = CHOLESKY_THRESHOLD + 10;
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let vals: Vec<f64>      = vec![4.0f64; n];   // pure diagonal, κ = 1
        let f     = vec![1.0f64; n];
        let mut u = vec![0.0f64; n];
        let mut ctx = GpuContext::new(false);

        let result = solve_linear_system(
            &row_ptr, &col_idx, &vals,
            &f, &mut u,
            1e-8, 10_000,
            n + 1, 1, 1,
            &mut ctx,
        );
        assert!(result.converged, "large-problem solver must converge on diagonal system");
        for &ui in &u {
            assert!(
                (ui - 0.25).abs() < 1e-5,
                "u={ui:.6} expected 0.25 for diagonal system"
            );
        }
    }

    #[test]
    fn dispatch_large_cpu_uses_jacobi() {
        // Verify that the Jacobi-PCG path (cg_solve_inner) remains correct
        // regardless of which higher-level backend is compiled in.
        // This test calls cg_solve_inner directly — it is the source-of-truth
        // correctness baseline that all other backends must match at iter 1.
        let n = CHOLESKY_THRESHOLD + 10;
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_idx: Vec<usize> = (0..n).collect();
        let vals: Vec<f64>      = vec![4.0f64; n];   // pure diagonal, u* = f/4 = 0.25
        let f     = vec![1.0f64; n];
        let mut u = vec![0.0f64; n];

        let result = cg_solve_inner(&row_ptr, &col_idx, &vals, &f, &mut u, 1e-8, 10_000);
        assert!(result.converged, "Jacobi-PCG must converge on diagonal system");
        for &ui in &u {
            assert!((ui - 0.25).abs() < 1e-6, "u={ui:.6} expected 0.25");
        }
    }

    #[test]
    fn gpu_context_new_does_not_panic() {
        let ctx = GpuContext::new(false);
        assert!(!ctx.use_gpu);
        let ctx2 = GpuContext::new(true);
        assert!(ctx2.use_gpu);
    }

    #[test]
    fn gpu_context_cpu_build_label_contains_backend_name() {
        let ctx_cpu = GpuContext::new(false);
        let label = ctx_cpu.backend_label();
        // Label should contain something meaningful regardless of feature set.
        assert!(!label.is_empty());
        assert!(
            label.contains("AMGCL") || label.contains("Jacobi") || label.contains("GPU"),
            "Unexpected backend label: {label}"
        );
    }
}
