// src/amgcl_solver.rs
//
// Rust side of the AMGCL integration.
// The critical parameter change vs the prior version: amgcl_create now takes
// dofs_per_node=3, which sets block_size=3 in smoothed_aggregation.
// Without this, AMG diverges on 3D elasticity problems.  See amgcl_wrapper.cpp.

use crate::solver::CgResult;

#[repr(C)]
pub struct AmgclHandle { _opaque: [u8; 0] }

#[link(name = "amgcl_wrapper", kind = "static")]
extern "C" {
    // dofs_per_node=3 for 3D elasticity.  Added vs prior version — this is
    // what sets block_size in smoothed_aggregation and fixes AMG convergence.
    fn amgcl_create(
        n: i32, nnz: i32, dofs_per_node: i32,
        row_ptr: *const i32, col_idx: *const i32, vals: *const f64,
        tol: f64, max_iter: i32,
    ) -> *mut AmgclHandle;

    fn amgcl_update(h: *mut AmgclHandle, vals: *const f64) -> i32;

    fn amgcl_solve(
        h: *mut AmgclHandle, rhs: *const f64, x: *mut f64,
        out_iters: *mut i32, out_residual: *mut f64,
    ) -> i32;

    fn amgcl_last_error(h: *const AmgclHandle) -> *const std::os::raw::c_char;
    fn amgcl_destroy(h: *mut AmgclHandle);
}

pub struct AmgclContext {
    handle:      *mut AmgclHandle,
    failed:      bool,
    row_ptr_i32: Vec<i32>,
    col_idx_i32: Vec<i32>,
}

impl AmgclContext {
    pub fn new() -> Self {
        AmgclContext {
            handle:      std::ptr::null_mut(),
            failed:      false,
            row_ptr_i32: Vec::new(),
            col_idx_i32: Vec::new(),
        }
    }

    #[inline] pub fn is_failed(&self) -> bool { self.failed }

    pub fn backend_label(&self) -> &'static str {
        "AMGCL AMG-PCG (OpenMP, smoothed aggregation block_size=3 + ILU(0))"
    }

    pub fn solve(
        &mut self,
        k_rows:   &[usize], k_cols: &[usize], k_vals: &[f64],
        f: &[f64], u: &mut [f64],
        tol: f64, max_iter: usize,
    ) -> Option<CgResult> {
        if self.failed { return None; }

        let n   = f.len() as i32;
        let nnz = k_vals.len() as i32;

        if self.handle.is_null() {
            self.row_ptr_i32 = k_rows.iter().map(|&x| x as i32).collect();
            self.col_idx_i32 = k_cols.iter().map(|&x| x as i32).collect();

            let h = unsafe {
                amgcl_create(
                    n, nnz,
                    3,  // dofs_per_node: always 3 for 3D hex FEM (u_x, u_y, u_z)
                    self.row_ptr_i32.as_ptr(),
                    self.col_idx_i32.as_ptr(),
                    k_vals.as_ptr(),
                    tol, max_iter as i32,
                )
            };

            if h.is_null() {
                eprintln!("[amgcl] amgcl_create returned null → Jacobi-PCG fallback");
                self.failed = true;
                return None;
            }
            let err = self.error_str(h);
            if !err.is_empty() {
                eprintln!("[amgcl] hierarchy build failed: {err} → Jacobi-PCG fallback");
                unsafe { amgcl_destroy(h); }
                self.failed = true;
                return None;
            }
            self.handle = h;
            eprintln!("[solver] AMGCL backend active: {}", self.backend_label());
        } else {
            let rc = unsafe { amgcl_update(self.handle, k_vals.as_ptr()) };
            if rc != 0 {
                let msg = self.error_str(self.handle);
                eprintln!("[amgcl] amgcl_update failed: {msg} → Jacobi-PCG fallback");
                self.failed = true;
                return None;
            }
        }

        let mut iters:    i32 = 0;
        let mut residual: f64 = 0.0;
        let rc = unsafe {
            amgcl_solve(self.handle, f.as_ptr(), u.as_mut_ptr(),
                        &mut iters, &mut residual)
        };
        if rc != 0 {
            let msg = self.error_str(self.handle);
            eprintln!("[amgcl] amgcl_solve failed: {msg} → Jacobi-PCG fallback");
            self.failed = true;
            return None;
        }

        Some(CgResult {
            iterations:   iters as usize,
            rel_residual: residual,
            converged:    residual <= tol,
        })
    }

    fn error_str(&self, h: *const AmgclHandle) -> String {
        if h.is_null() { return "null handle".into(); }
        let ptr = unsafe { amgcl_last_error(h) };
        if ptr.is_null() { return String::new(); }
        unsafe { std::ffi::CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

impl Default for AmgclContext { fn default() -> Self { Self::new() } }

impl Drop for AmgclContext {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { amgcl_destroy(self.handle); }
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn laplacian_1d(n: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let mut rp = vec![0usize; n + 1];
        let mut ci = Vec::new();
        let mut v  = Vec::new();
        for i in 0..n {
            if i > 0 { ci.push(i-1); v.push(-1.0); }
            ci.push(i); v.push(2.0);
            if i < n-1 { ci.push(i+1); v.push(-1.0); }
            rp[i+1] = ci.len();
        }
        (rp, ci, v)
    }

    #[test]
    fn amgcl_context_new_does_not_panic() {
        let ctx = AmgclContext::new();
        assert!(!ctx.is_failed());
        assert!(ctx.handle.is_null());
    }

    #[test]
    fn amgcl_solves_1d_laplacian() {
        let n = 100;
        let (rp, ci, v) = laplacian_1d(n);
        let rhs = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let mut ctx = AmgclContext::new();
        let res = ctx.solve(&rp, &ci, &v, &rhs, &mut x, 1e-8, 500)
            .expect("AMGCL must solve 1D Laplacian");
        assert!(res.converged,
            "AMGCL failed: res={:.2e} iters={}", res.rel_residual, res.iterations);
        assert!(res.iterations <= 30,
            "AMG used {} iters on 100-DOF Laplacian", res.iterations);
    }

    #[test]
    fn amgcl_context_update_and_resolves() {
        let n = 50;
        let (rp, ci, mut v) = laplacian_1d(n);
        let rhs = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let mut ctx = AmgclContext::new();
        let r1 = ctx.solve(&rp, &ci, &v, &rhs, &mut x, 1e-8, 500).expect("first solve");
        assert!(r1.converged);
        for val in v.iter_mut() { if *val > 0.0 { *val *= 1.01; } }
        x.iter_mut().for_each(|v| *v = 0.0);
        let r2 = ctx.solve(&rp, &ci, &v, &rhs, &mut x, 1e-8, 500).expect("update solve");
        assert!(r2.converged);
    }

    #[test]
    fn amgcl_context_is_not_failed_after_valid_solve() {
        let n = 20;
        let (rp, ci, v) = laplacian_1d(n);
        let rhs = vec![1.0f64; n];
        let mut x = vec![0.0f64; n];
        let mut ctx = AmgclContext::new();
        let _ = ctx.solve(&rp, &ci, &v, &rhs, &mut x, 1e-8, 200);
        assert!(!ctx.is_failed());
    }
}