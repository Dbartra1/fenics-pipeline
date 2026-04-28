// src/gpu_solver.rs
//
// GPU-accelerated ILU(0)-preconditioned CG.
//
// Tier 5 Session 2 changes vs Session 1:
//   mat_K:  persistent cusparseSpMatDescr_t — no Create/Destroy per SpMV call
//   cublas: cuBLAS handle for DAXPY / DSCAL / DDOT on device-resident vectors
//   p_u:    device solution buffer — stays on GPU for the entire CG loop
//
//   cg_solve_persistent():
//     H2D cost: n f64 for f (once at entry).
//     Per-iter cost: SpMV (cuSPARSE) + trisol x2 (cuSPARSE) +
//                    3x DDOT scalar D2H (α, β, convergence check) +
//                    3x DAXPY + 1x DSCAL (cuBLAS, all device-side).
//     D2H cost: n f64 for u (once at exit).
//
// Public legacy methods matvec() / precondition() are retained for diagnostics;
// they delegate to the private *_dev() variants to avoid code duplication.
//
// cuBLAS note: if .result() does not compile on cublasStatus_t in your cudarc
// build, replace with:
//   if st != cbl::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
//       return Err(format!("cublas error: {:?}", st));
//   }

use std::sync::Arc;
use cudarc::cublas::sys  as cbl;
use cudarc::cusparse::sys as cusp;
use cudarc::driver::{result, sys, CudaContext};

type CUdev = sys::CUdeviceptr;

pub struct GpuK {
    _ctx:   Arc<CudaContext>,
    handle: cusp::cusparseHandle_t,   // cuSPARSE
    cublas: cbl::cublasHandle_t,      // cuBLAS  (NEW)

    // ── K matrix ───────────────────────────────────────────────────────────────
    p_val:       CUdev,                          // K values on device
    mat_K:       cusp::cusparseSpMatDescr_t,     // persistent descriptor  (NEW)
    p_spmv_buf:  CUdev,                          // SpMV scratch

    // ── ILU(0) factor (overwritten in-place by cusparseDcsrilu02) ─────────────
    p_ilu_val:   CUdev,
    p_ilu_buf:   CUdev,
    info_ilu:    cusp::csrilu02Info_t,
    descr_M:     cusp::cusparseMatDescr_t,

    // ── SpSV triangular solve resources ───────────────────────────────────────
    mat_L:        cusp::cusparseSpMatDescr_t,
    mat_U:        cusp::cusparseSpMatDescr_t,
    spsv_L:       cusp::cusparseSpSVDescr_t,
    spsv_U:       cusp::cusparseSpSVDescr_t,
    p_spsv_buf_L: CUdev,
    p_spsv_buf_U: CUdev,

    // ── Shared row/col (same sparsity for K, L, U) ────────────────────────────
    p_row: CUdev,   // i32[n+1]
    p_col: CUdev,   // i32[nnz]

    // ── CG work vectors (f64[n], all GPU-resident) ─────────────────────────────
    p_u:  CUdev,   // solution                        (NEW — stays on GPU)
    p_r:  CUdev,   // residual / preconditioner input
    p_y:  CUdev,   // L⁻¹·r  (intermediate)
    p_z:  CUdev,   // M⁻¹·r  (preconditioner output)
    p_x:  CUdev,   // search direction p  (SpMV input)
    p_kp: CUdev,   // K·p                (SpMV output)

    n:   usize,
    nnz: usize,

    /// Jacobi diagonal — CPU fallback if ILU produces a zero pivot.
    pub diag: Vec<f64>,
}

impl GpuK {
    // ── Upload + factorize (once per SIMP step) ───────────────────────────────

    /// Upload K, compute ILU(0) factorization and SpSV analysis.
    /// K structure (p_row / p_col) is fixed across SIMP steps; only values change.
    pub fn upload(k_rows: &[usize], k_cols: &[usize], k_vals: &[f64]) -> Result<Self, String> {
        let n   = k_rows.len().saturating_sub(1);
        let nnz = k_vals.len();

        let row_i32: Vec<i32> = k_rows.iter().map(|&v| v as i32).collect();
        let col_i32: Vec<i32> = k_cols.iter().map(|&v| v as i32).collect();

        // Jacobi diagonal — CPU-side, fallback only.
        let mut diag = vec![1.0_f64; n];
        for i in 0..n {
            let row = &col_i32[row_i32[i] as usize..row_i32[i + 1] as usize];
            if let Ok(pos) = row.binary_search(&(i as i32)) {
                let d = k_vals[row_i32[i] as usize + pos];
                if d.abs() > 1e-30 { diag[i] = d; }
            }
        }

        let ctx = CudaContext::new(0)
            .map_err(|e| format!("[gpu] CudaContext::new: {e}"))?;

        // ── Device allocations ────────────────────────────────────────────────
        let p_row     = unsafe { alloc_upload(4 * (n + 1), &row_i32)? };
        let p_col     = unsafe { alloc_upload(4 * nnz,     &col_i32)? };
        let p_val     = unsafe { alloc_upload(8 * nnz,     k_vals)?   };
        let p_ilu_val = unsafe { alloc_upload(8 * nnz,     k_vals)?   };

        let alloc = |s: usize| unsafe {
            result::malloc_sync(s).map_err(|e| format!("[gpu] malloc: {e}"))
        };
        let p_u  = alloc(8 * n)?;   // NEW
        let p_r  = alloc(8 * n)?;
        let p_y  = alloc(8 * n)?;
        let p_z  = alloc(8 * n)?;
        let p_x  = alloc(8 * n)?;
        let p_kp = alloc(8 * n)?;

        // ── cuSPARSE handle ────────────────────────────────────────────────────
        let mut handle: cusp::cusparseHandle_t = std::ptr::null_mut();
        unsafe {
            cusp::cusparseCreate(&mut handle)
                .result().map_err(|e| format!("[gpu] cusparseCreate: {e:?}"))?;
            cusp::cusparseSetStream(handle, std::ptr::null_mut())
                .result().map_err(|e| format!("[gpu] SetStream: {e:?}"))?;
        }

        // ── ILU(0) factorization ──────────────────────────────────────────────
        let (descr_M, info_ilu, p_ilu_buf) = unsafe {
            run_ilu02(handle, p_row, p_col, p_ilu_val, n, nnz)?
        };

        // ── SpSV: L and U sparse matrix descriptors ───────────────────────────
        let (mat_L, mat_U) = unsafe {
            make_lu_matrices(p_row, p_col, p_ilu_val, n, nnz)?
        };

        // ── SpSV: analysis for L and U ────────────────────────────────────────
        let (spsv_L, p_spsv_buf_L) = unsafe { spsv_analyze(handle, mat_L, p_r, p_y, n)? };
        let (spsv_U, p_spsv_buf_U) = unsafe { spsv_analyze(handle, mat_U, p_y, p_z, n)? };

        // ── Persistent K descriptor (NEW) ─────────────────────────────────────
        // Created once here; reused by every matvec_dev() call.
        // p_val pointer is stable for the lifetime of this GpuK instance.
        let mat_K = unsafe {
            let mut mat: cusp::cusparseSpMatDescr_t = std::ptr::null_mut();
            cusp::cusparseCreateCsr(
                &mut mat,
                n as i64, n as i64, nnz as i64,
                p_row as *mut std::ffi::c_void,
                p_col as *mut std::ffi::c_void,
                p_val as *mut std::ffi::c_void,
                cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                cusp::cudaDataType::CUDA_R_64F,
            ).result().map_err(|e| format!("[gpu] CreateCsr mat_K: {e:?}"))?;
            mat
        };

        // ── SpMV scratch buffer — queried against mat_K (NEW helper) ──────────
        let p_spmv_buf = unsafe {
            query_spmv_buf_for_mat(handle, mat_K, p_x, p_kp, n)?
        };

        // ── cuBLAS handle (NEW) ────────────────────────────────────────────────
        // HOST pointer mode: DDOT / DAXPY / DSCAL scalar args passed as CPU ptrs.
        // Each DDOT call issues an implicit stream sync and returns scalar to CPU.
        // 3 syncs per CG iter << current ~12 MB PCIe transfer per iter.
        let cublas = unsafe {
            let mut h: cbl::cublasHandle_t = std::ptr::null_mut();
            cbl::cublasCreate_v2(&mut h)
                .result().map_err(|e| format!("[gpu] cublasCreate: {e:?}"))?;
            cbl::cublasSetPointerMode_v2(
                h,
                cbl::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
            ).result().map_err(|e| format!("[gpu] cublasPointerMode: {e:?}"))?;
            h
        };

        Ok(Self {
            _ctx: ctx, handle, cublas,
            p_val, mat_K, p_spmv_buf,
            p_ilu_val, p_ilu_buf, info_ilu, descr_M,
            mat_L, mat_U, spsv_L, spsv_U, p_spsv_buf_L, p_spsv_buf_U,
            p_row, p_col,
            p_u, p_r, p_y, p_z, p_x, p_kp,
            n, nnz, diag,
        })
    }

    // ── Primary solve path (Tier 5 Session 2) ────────────────────────────────

    /// Solve K·u = f entirely on GPU.  Called once per SIMP step.
    ///
    /// Transfer budget: 1 H2D (f, n doubles) + 3 scalar D2H per CG iter
    /// (DDOT results) + 1 D2H (u, n doubles) at exit.
    /// All SpMV, triangular solves, AXPY, SCAL, DOT run on device.
    pub fn cg_solve_persistent(
        &self,
        f:        &[f64],
        u:        &mut [f64],
        tol:      f64,
        max_iter: usize,
    ) -> crate::solver::CgResult {
        use crate::solver::CgResult;

        let n      = self.n;
        let f_norm = f.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);

        // ── Upload once ───────────────────────────────────────────────────────
        // u = 0, r = f on device.
        // memset via zero-vec upload: simple and correct (IEEE 754 +0.0 = 0x00…).
        let zeros = vec![0.0_f64; n];
        unsafe {
            if result::memcpy_htod_sync(self.p_u, &zeros).is_err() {
                eprintln!("[gpu_cg] upload zeros failed");
                return CgResult { iterations: 0, rel_residual: f64::INFINITY, converged: false };
            }
            if result::memcpy_htod_sync(self.p_r, f).is_err() {
                eprintln!("[gpu_cg] upload f failed");
                return CgResult { iterations: 0, rel_residual: f64::INFINITY, converged: false };
            }
        }

        // ── First preconditioner: z = M⁻¹·r ──────────────────────────────────
        if let Err(e) = self.precondition_dev() {
            eprintln!("[gpu_cg] first precondition: {e}");
            return CgResult { iterations: 0, rel_residual: f64::INFINITY, converged: false };
        }

        // ── p = z ─────────────────────────────────────────────────────────────
        unsafe {
            if cbl::cublasDcopy_v2(
                self.cublas, n as i32,
                self.p_z as *const f64, 1,
                self.p_x as *mut   f64, 1,
            ).result().is_err() {
                eprintln!("[gpu_cg] copy p=z failed");
                return CgResult { iterations: 0, rel_residual: f64::INFINITY, converged: false };
            }
        }

        // ── rz = r · z ────────────────────────────────────────────────────────
        let mut rz = match self.ddot(self.p_r, self.p_z) {
            Ok(v)  => v,
            Err(e) => {
                eprintln!("[gpu_cg] dot rz init: {e}");
                return CgResult { iterations: 0, rel_residual: f64::INFINITY, converged: false };
            }
        };

        let mut iterations   = 0;
        let mut rel_residual = 1.0_f64;

        'cg: for iter in 0..max_iter {
            iterations = iter + 1;

            // Convergence: ‖r‖₂ / ‖f‖₂
            let rr = match self.ddot(self.p_r, self.p_r) {
                Ok(v)  => v,
                Err(e) => { eprintln!("[gpu_cg] dot rr iter {iter}: {e}"); break 'cg; }
            };
            rel_residual = rr.sqrt() / f_norm;
            if rel_residual < tol { break 'cg; }

            // kp = K · p_x  (SpMV, device-side)
            if let Err(e) = self.matvec_dev() {
                eprintln!("[gpu_cg] matvec iter {iter}: {e}");
                break 'cg;
            }

            // pkp = p · kp
            let pkp = match self.ddot(self.p_x, self.p_kp) {
                Ok(v)  => v,
                Err(e) => { eprintln!("[gpu_cg] dot pkp iter {iter}: {e}"); break 'cg; }
            };
            if pkp.abs() < 1e-30 { break 'cg; }

            let alpha     =  rz / pkp;
            let neg_alpha = -alpha;

            unsafe {
                // u += α·p
                if cbl::cublasDaxpy_v2(
                    self.cublas, n as i32, &alpha,
                    self.p_x as *const f64, 1,
                    self.p_u as *mut   f64, 1,
                ).result().is_err() { break 'cg; }

                // r -= α·kp
                if cbl::cublasDaxpy_v2(
                    self.cublas, n as i32, &neg_alpha,
                    self.p_kp as *const f64, 1,
                    self.p_r  as *mut   f64, 1,
                ).result().is_err() { break 'cg; }
            }

            // z = M⁻¹·r  (ILU trisol, device-side)
            if let Err(e) = self.precondition_dev() {
                eprintln!("[gpu_cg] precondition iter {iter}: {e}");
                break 'cg;
            }

            // rz_new = r · z
            let rz_new = match self.ddot(self.p_r, self.p_z) {
                Ok(v)  => v,
                Err(e) => { eprintln!("[gpu_cg] dot rz_new iter {iter}: {e}"); break 'cg; }
            };

            let beta = rz_new / rz.max(1e-30);

            unsafe {
                // p = β·p + z  (scale first, then add z)
                if cbl::cublasDscal_v2(
                    self.cublas, n as i32, &beta,
                    self.p_x as *mut f64, 1,
                ).result().is_err() { break 'cg; }

                if cbl::cublasDaxpy_v2(
                    self.cublas, n as i32, &1.0_f64,
                    self.p_z as *const f64, 1,
                    self.p_x as *mut   f64, 1,
                ).result().is_err() { break 'cg; }
            }

            rz = rz_new;
        }

        // ── D2H: solution (single full-vector transfer at exit) ───────────────
        if let Err(e) = unsafe { result::memcpy_dtoh_sync(u, self.p_u) } {
            eprintln!("[gpu_cg] download u: {e}");
            return CgResult { iterations, rel_residual: f64::INFINITY, converged: false };
        }

        CgResult { iterations, rel_residual, converged: rel_residual < tol }
    }

    // ── Legacy public methods (diagnostics / backward compat) ─────────────────

    /// kp = K·p  (H2D p → SpMV → D2H kp).
    /// Session 1 interface; kept for diagnostic use and existing call sites.
    pub fn matvec(&self, p: &[f64], kp: &mut [f64]) -> Result<(), String> {
        unsafe {
            result::memcpy_htod_sync(self.p_x, p)
                .map_err(|e| format!("[gpu] upload p: {e}"))?;
        }
        self.matvec_dev()?;
        unsafe {
            result::memcpy_dtoh_sync(kp, self.p_kp)
                .map_err(|e| format!("[gpu] download kp: {e}"))?;
        }
        Ok(())
    }

    /// z = M⁻¹·r  (H2D r → ILU trisol → D2H z).
    /// Session 1 interface; kept for diagnostic use and existing call sites.
    pub fn precondition(&self, r: &[f64], z: &mut [f64]) -> Result<(), String> {
        unsafe {
            result::memcpy_htod_sync(self.p_r, r)
                .map_err(|e| format!("[gpu] upload r: {e}"))?;
        }
        self.precondition_dev()?;
        unsafe {
            result::memcpy_dtoh_sync(z, self.p_z)
                .map_err(|e| format!("[gpu] download z: {e}"))?;
        }
        Ok(())
    }

    // ── Private GPU-resident operations ───────────────────────────────────────

    /// SpMV: p_kp = K · p_x.  No H2D/D2H.  Uses persistent mat_K.
    fn matvec_dev(&self) -> Result<(), String> {
        let n = self.n as i64;
        let (a, b) = (1.0_f64, 0.0_f64);
        unsafe {
            let mut dn_x:  cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
            let mut dn_kp: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
            cusp::cusparseCreateDnVec(
                &mut dn_x,  n, self.p_x  as *mut _, cusp::cudaDataType::CUDA_R_64F,
            ).result().ok();
            cusp::cusparseCreateDnVec(
                &mut dn_kp, n, self.p_kp as *mut _, cusp::cudaDataType::CUDA_R_64F,
            ).result().ok();

            let st = cusp::cusparseSpMV(
                self.handle,
                cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a as *const f64 as _,
                self.mat_K as cusp::cusparseConstSpMatDescr_t,
                dn_x       as cusp::cusparseConstDnVecDescr_t,
                &b as *const f64 as _,
                dn_kp,
                cusp::cudaDataType::CUDA_R_64F,
                cusp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                self.p_spmv_buf as *mut _,
            );
            cusp::cusparseDestroyDnVec(dn_x);
            cusp::cusparseDestroyDnVec(dn_kp);
            st.result().map_err(|e| format!("[gpu] matvec_dev SpMV: {e:?}"))?;
        }
        Ok(())
    }

    /// ILU(0) solve: p_z = U⁻¹·(L⁻¹·p_r).  No H2D/D2H.
    fn precondition_dev(&self) -> Result<(), String> {
        let n   = self.n as i64;
        let one = 1.0_f64;
        unsafe {
            // Helpers — DnVec descriptors are cheap pointer wrappers.
            let mk_const = |p: CUdev| -> cusp::cusparseConstDnVecDescr_t {
                let mut d: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
                cusp::cusparseCreateDnVec(
                    &mut d, n, p as *mut _, cusp::cudaDataType::CUDA_R_64F,
                ).result().ok();
                d as cusp::cusparseConstDnVecDescr_t
            };
            let mk_mut = |p: CUdev| -> cusp::cusparseDnVecDescr_t {
                let mut d: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
                cusp::cusparseCreateDnVec(
                    &mut d, n, p as *mut _, cusp::cudaDataType::CUDA_R_64F,
                ).result().ok();
                d
            };

            // L·y = r  (forward)
            let dn_r = mk_const(self.p_r);
            let dn_y = mk_mut(self.p_y);
            let st_l = cusp::cusparseSpSV_solve(
                self.handle,
                cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one as *const f64 as *const _,
                self.mat_L as cusp::cusparseConstSpMatDescr_t,
                dn_r, dn_y,
                cusp::cudaDataType::CUDA_R_64F,
                cusp::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
                self.spsv_L,
            );
            cusp::cusparseDestroyDnVec(dn_y);
            cusp::cusparseDestroyDnVec(dn_r as cusp::cusparseDnVecDescr_t);
            st_l.result().map_err(|e| format!("[gpu] precondition_dev L: {e:?}"))?;

            // U·z = y  (backward)
            let dn_yc = mk_const(self.p_y);
            let dn_z  = mk_mut(self.p_z);
            let st_u = cusp::cusparseSpSV_solve(
                self.handle,
                cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one as *const f64 as *const _,
                self.mat_U as cusp::cusparseConstSpMatDescr_t,
                dn_yc, dn_z,
                cusp::cudaDataType::CUDA_R_64F,
                cusp::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
                self.spsv_U,
            );
            cusp::cusparseDestroyDnVec(dn_z);
            cusp::cusparseDestroyDnVec(dn_yc as cusp::cusparseDnVecDescr_t);
            st_u.result().map_err(|e| format!("[gpu] precondition_dev U: {e:?}"))?;
        }
        Ok(())
    }

    /// cuBLAS DDOT with HOST pointer mode.
    /// Internally issues a stream synchronization; returns scalar to CPU.
    /// Cost: ~10 µs overhead per call — negligible vs prior full-vector transfers.
    fn ddot(&self, p_a: CUdev, p_b: CUdev) -> Result<f64, String> {
        let mut out = 0.0_f64;
        unsafe {
            cbl::cublasDdot_v2(
                self.cublas, self.n as i32,
                p_a as *const f64, 1,
                p_b as *const f64, 1,
                &mut out,
            ).result().map_err(|e| format!("[gpu] cublasDdot: {e:?}"))?;
        }
        Ok(out)
    }
}

impl Drop for GpuK {
    fn drop(&mut self) {
        unsafe {
            for p in [
                self.p_val, self.p_spmv_buf,
                self.p_ilu_val, self.p_ilu_buf,
                self.p_spsv_buf_L, self.p_spsv_buf_U,
                self.p_row, self.p_col,
                self.p_u,                          // NEW
                self.p_r, self.p_y, self.p_z, self.p_x, self.p_kp,
            ] {
                if p != 0 { result::free_sync(p).ok(); }
            }
            if !self.info_ilu.is_null()  { cusp::cusparseDestroyCsrilu02Info(self.info_ilu); }
            if !self.descr_M.is_null()   { cusp::cusparseDestroyMatDescr(self.descr_M); }
            if !self.mat_K.is_null()     { cusp::cusparseDestroySpMat(self.mat_K); }   // NEW
            if !self.mat_L.is_null()     { cusp::cusparseDestroySpMat(self.mat_L); }
            if !self.mat_U.is_null()     { cusp::cusparseDestroySpMat(self.mat_U); }
            if !self.spsv_L.is_null()    { cusp::cusparseSpSV_destroyDescr(self.spsv_L); }
            if !self.spsv_U.is_null()    { cusp::cusparseSpSV_destroyDescr(self.spsv_U); }
            if !self.handle.is_null()    { cusp::cusparseDestroy(self.handle); }
            if !self.cublas.is_null()    { cbl::cublasDestroy_v2(self.cublas); }       // NEW
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

unsafe fn alloc_upload<T>(bytes: usize, src: &[T]) -> Result<CUdev, String> {
    let p = result::malloc_sync(bytes)
        .map_err(|e| format!("[gpu] malloc {bytes}B: {e}"))?;
    result::memcpy_htod_sync(p, src)
        .map_err(|e| format!("[gpu] upload: {e}"))?;
    Ok(p)
}

unsafe fn run_ilu02(
    handle:    cusp::cusparseHandle_t,
    p_row:     CUdev,
    p_col:     CUdev,
    p_ilu_val: CUdev,
    n: usize, nnz: usize,
) -> Result<(cusp::cusparseMatDescr_t, cusp::csrilu02Info_t, CUdev), String> {
    let mut descr_M: cusp::cusparseMatDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateMatDescr(&mut descr_M)
        .result().map_err(|e| format!("[gpu] CreateMatDescr M: {e:?}"))?;
    cusp::cusparseSetMatIndexBase(
        descr_M, cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
    ).result().ok();
    cusp::cusparseSetMatType(
        descr_M, cusp::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL,
    ).result().ok();

    let mut info: cusp::csrilu02Info_t = std::ptr::null_mut();
    cusp::cusparseCreateCsrilu02Info(&mut info)
        .result().map_err(|e| format!("[gpu] CreateILU02Info: {e:?}"))?;

    let in_  = n   as i32;
    let innz = nnz as i32;
    let rp   = p_row     as *const i32;
    let cp   = p_col     as *const i32;
    let vp   = p_ilu_val as *mut   f64;
    let vp_c = p_ilu_val as *const f64;
    let pol  = cusp::cusparseSolvePolicy_t::CUSPARSE_SOLVE_POLICY_NO_LEVEL;

    let mut sz: i32 = 0;
    cusp::cusparseDcsrilu02_bufferSize(handle, in_, innz, descr_M, vp, rp, cp, info, &mut sz)
        .result().map_err(|e| format!("[gpu] ilu02_bufferSize: {e:?}"))?;

    let p_buf = result::malloc_sync(sz.max(1) as usize)
        .map_err(|e| format!("[gpu] malloc ilu_buf: {e}"))?;
    let ibuf = p_buf as *mut std::ffi::c_void;

    cusp::cusparseDcsrilu02_analysis(handle, in_, innz, descr_M, vp_c, rp, cp, info, pol, ibuf)
        .result().map_err(|e| format!("[gpu] ilu02_analysis: {e:?}"))?;
    cusp::cusparseDcsrilu02(handle, in_, innz, descr_M, vp, rp, cp, info, pol, ibuf)
        .result().map_err(|e| format!("[gpu] ilu02_factorize: {e:?}"))?;

    Ok((descr_M, info, p_buf))
}

unsafe fn make_lu_matrices(
    p_row: CUdev, p_col: CUdev, p_ilu_val: CUdev,
    n: usize, nnz: usize,
) -> Result<(cusp::cusparseSpMatDescr_t, cusp::cusparseSpMatDescr_t), String> {
    let make = |fill: cusp::cusparseFillMode_t, diag: cusp::cusparseDiagType_t|
        -> Result<cusp::cusparseSpMatDescr_t, String>
    {
        let mut mat: cusp::cusparseSpMatDescr_t = std::ptr::null_mut();
        cusp::cusparseCreateCsr(
            &mut mat,
            n as i64, n as i64, nnz as i64,
            p_row     as *mut std::ffi::c_void,
            p_col     as *mut std::ffi::c_void,
            p_ilu_val as *mut std::ffi::c_void,
            cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
            cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
            cusp::cudaDataType::CUDA_R_64F,
        ).result().map_err(|e| format!("[gpu] CreateCsr L/U: {e:?}"))?;

        cusp::cusparseSpMatSetAttribute(
            mat,
            cusp::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_FILL_MODE,
            &fill as *const cusp::cusparseFillMode_t as *mut std::ffi::c_void,
            std::mem::size_of::<cusp::cusparseFillMode_t>(),
        ).result().map_err(|e| format!("[gpu] SetAttr fill: {e:?}"))?;

        cusp::cusparseSpMatSetAttribute(
            mat,
            cusp::cusparseSpMatAttribute_t::CUSPARSE_SPMAT_DIAG_TYPE,
            &diag as *const cusp::cusparseDiagType_t as *mut std::ffi::c_void,
            std::mem::size_of::<cusp::cusparseDiagType_t>(),
        ).result().map_err(|e| format!("[gpu] SetAttr diag: {e:?}"))?;

        Ok(mat)
    };

    let mat_L = make(
        cusp::cusparseFillMode_t::CUSPARSE_FILL_MODE_LOWER,
        cusp::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_UNIT,
    )?;
    let mat_U = make(
        cusp::cusparseFillMode_t::CUSPARSE_FILL_MODE_UPPER,
        cusp::cusparseDiagType_t::CUSPARSE_DIAG_TYPE_NON_UNIT,
    )?;
    Ok((mat_L, mat_U))
}

unsafe fn spsv_analyze(
    handle:   cusp::cusparseHandle_t,
    mat:      cusp::cusparseSpMatDescr_t,
    p_x_in:   CUdev,
    p_y_out:  CUdev,
    n:        usize,
) -> Result<(cusp::cusparseSpSVDescr_t, CUdev), String> {
    let mut descr: cusp::cusparseSpSVDescr_t = std::ptr::null_mut();
    cusp::cusparseSpSV_createDescr(&mut descr)
        .result().map_err(|e| format!("[gpu] SpSV_createDescr: {e:?}"))?;

    let mut dn_x: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    let mut dn_y: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateDnVec(
        &mut dn_x, n as i64, p_x_in  as *mut _, cusp::cudaDataType::CUDA_R_64F,
    ).result().ok();
    cusp::cusparseCreateDnVec(
        &mut dn_y, n as i64, p_y_out as *mut _, cusp::cudaDataType::CUDA_R_64F,
    ).result().ok();

    let one   = 1.0_f64;
    let mat_c = mat   as cusp::cusparseConstSpMatDescr_t;
    let x_c   = dn_x  as cusp::cusparseConstDnVecDescr_t;

    let mut buf_sz: usize = 0;
    cusp::cusparseSpSV_bufferSize(
        handle,
        cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one as *const f64 as *const _,
        mat_c, x_c, dn_y,
        cusp::cudaDataType::CUDA_R_64F,
        cusp::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        descr, &mut buf_sz,
    ).result().map_err(|e| format!("[gpu] SpSV_bufferSize: {e:?}"))?;

    let p_buf = result::malloc_sync(buf_sz.max(1))
        .map_err(|e| format!("[gpu] malloc spsv_buf: {e}"))?;

    cusp::cusparseSpSV_analysis(
        handle,
        cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &one as *const f64 as *const _,
        mat_c, x_c, dn_y,
        cusp::cudaDataType::CUDA_R_64F,
        cusp::cusparseSpSVAlg_t::CUSPARSE_SPSV_ALG_DEFAULT,
        descr, p_buf as *mut _,
    ).result().map_err(|e| format!("[gpu] SpSV_analysis: {e:?}"))?;

    cusp::cusparseDestroyDnVec(dn_x);
    cusp::cusparseDestroyDnVec(dn_y);
    Ok((descr, p_buf))
}

/// Query SpMV scratch buffer size against the persistent mat_K descriptor.
/// Replaces the old query_spmv_buf() which created a throwaway K descriptor.
unsafe fn query_spmv_buf_for_mat(
    handle: cusp::cusparseHandle_t,
    mat_K:  cusp::cusparseSpMatDescr_t,
    p_x:    CUdev,
    p_kp:   CUdev,
    n:      usize,
) -> Result<CUdev, String> {
    let mut dn_x:  cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    let mut dn_kp: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateDnVec(
        &mut dn_x,  n as i64, p_x  as *mut _, cusp::cudaDataType::CUDA_R_64F,
    ).result().ok();
    cusp::cusparseCreateDnVec(
        &mut dn_kp, n as i64, p_kp as *mut _, cusp::cudaDataType::CUDA_R_64F,
    ).result().ok();

    let (a, b) = (1.0_f64, 0.0_f64);
    let mut buf_sz: usize = 0;
    cusp::cusparseSpMV_bufferSize(
        handle,
        cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a as *const f64 as _,
        mat_K as cusp::cusparseConstSpMatDescr_t,
        dn_x  as cusp::cusparseConstDnVecDescr_t,
        &b as *const f64 as _,
        dn_kp,
        cusp::cudaDataType::CUDA_R_64F,
        cusp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        &mut buf_sz,
    ).result().map_err(|e| format!("[gpu] spmv_bufferSize: {e:?}"))?;

    cusp::cusparseDestroyDnVec(dn_x);
    cusp::cusparseDestroyDnVec(dn_kp);

    result::malloc_sync(buf_sz.max(1))
        .map_err(|e| format!("[gpu] malloc spmv_buf: {e}"))
}
