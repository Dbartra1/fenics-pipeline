/// GPU-accelerated ILU(0)-preconditioned CG — Phase 3.
///
/// Triangular solves use cusparseSpSV (generic API, CUDA 12.x).
/// ILU factorization uses cusparseDcsrilu02 (still available in CUDA 12.x).
///
/// Per SIMP iteration (K changes):
///   1. Upload K to VRAM; copy to ILU factor array
///   2. cusparseDcsrilu02: factorize ILU array in-place → L and U share storage
///   3. cusparseSpSV_analysis: sparsity analysis for L and U solves (once)
///
/// Per CG iteration (preconditioner z = M⁻¹·r):
///   4. cusparseSpSV_solve L: y = L⁻¹·r
///   5. cusparseSpSV_solve U: z = U⁻¹·y

use std::sync::Arc;
use cudarc::driver::{result, sys, CudaContext};
use cudarc::cusparse::sys as cusp;

type CUdev = sys::CUdeviceptr;

pub struct GpuK {
    _ctx:   Arc<CudaContext>,
    handle: cusp::cusparseHandle_t,

    // ── K matrix (SpMV only, never overwritten) ───────────────────────────────
    p_val:       CUdev,
    p_spmv_buf:  CUdev,

    // ── ILU factor storage (overwritten in-place by cusparseDcsrilu02) ────────
    p_ilu_val:   CUdev,
    p_ilu_buf:   CUdev,       // scratch for ILU factorization
    info_ilu:    cusp::csrilu02Info_t,
    descr_M:     cusp::cusparseMatDescr_t,   // general, for ILU analysis

    // ── SpSV triangular solve resources ──────────────────────────────────────
    /// Sparse matrix descriptors for L (lower/unit) and U (upper/non-unit).
    /// Created from p_ilu_val; attributes set at upload, reused every solve.
    mat_L:       cusp::cusparseSpMatDescr_t,
    mat_U:       cusp::cusparseSpMatDescr_t,
    /// SpSV opaque descriptors (carry analysis results).
    spsv_L:      cusp::cusparseSpSVDescr_t,
    spsv_U:      cusp::cusparseSpSVDescr_t,
    /// External buffers for SpSV analysis/solve.
    p_spsv_buf_L: CUdev,
    p_spsv_buf_U: CUdev,

    // ── Shared row/col (same sparsity for K, L, U) ────────────────────────────
    p_row: CUdev,   // i32, n+1
    p_col: CUdev,   // i32, nnz

    // ── Work vectors (all f64, length n) ─────────────────────────────────────
    p_r:  CUdev,   // preconditioner input
    p_y:  CUdev,   // intermediate (L⁻¹·r)
    p_z:  CUdev,   // preconditioner output (U⁻¹·y)
    p_x:  CUdev,   // SpMV input  (search direction p)
    p_kp: CUdev,   // SpMV output (K·p)

    n:   usize,
    nnz: usize,

    /// Jacobi diagonal — fallback if ILU factorization produces zero pivot.
    pub diag: Vec<f64>,
}

impl GpuK {
    /// Upload K and compute ILU(0) factorization + SpSV analysis.
    /// Called once per SIMP iteration.
    pub fn upload(k_rows: &[usize], k_cols: &[usize], k_vals: &[f64]) -> Result<Self, String> {
        let n   = k_rows.len().saturating_sub(1);
        let nnz = k_vals.len();

        let row_i32: Vec<i32> = k_rows.iter().map(|&v| v as i32).collect();
        let col_i32: Vec<i32> = k_cols.iter().map(|&v| v as i32).collect();

        // ── Jacobi diagonal (CPU fallback) ────────────────────────────────────
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
        let p_row = unsafe { alloc_upload(4 * (n + 1), &row_i32)? };
        let p_col = unsafe { alloc_upload(4 * nnz,     &col_i32)? };
        let p_val = unsafe { alloc_upload(8 * nnz,     k_vals)?   };
        let p_ilu_val = unsafe { alloc_upload(8 * nnz, k_vals)?   };

        let alloc = |s: usize| unsafe {
            result::malloc_sync(s).map_err(|e| format!("[gpu] malloc: {e}"))
        };
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

        // ── ILU(0) factorization ───────────────────────────────────────────────
        let (descr_M, info_ilu, p_ilu_buf) = unsafe {
            run_ilu02(handle, p_row, p_col, p_ilu_val, n, nnz)?
        };

        // ── SpSV: create L and U sparse matrix descriptors ─────────────────────
        // Both point to p_ilu_val (same data, different fill/diag attributes).
        let (mat_L, mat_U) = unsafe {
            make_lu_matrices(p_row, p_col, p_ilu_val, n, nnz)?
        };

        // ── SpSV: analysis for L and U ─────────────────────────────────────────
        // Temporary DnVec descriptors (just for analysis; data values don't
        // matter — only dimensions and type are used).
        let (spsv_L, p_spsv_buf_L) = unsafe {
            spsv_analyze(handle, mat_L, p_r, p_y, n)?
        };
        let (spsv_U, p_spsv_buf_U) = unsafe {
            spsv_analyze(handle, mat_U, p_y, p_z, n)?
        };

        // ── SpMV scratch buffer ────────────────────────────────────────────────
        let p_spmv_buf = unsafe {
            query_spmv_buf(handle, p_row, p_col, p_val, p_x, p_kp, n, nnz)?
        };

        Ok(Self {
            _ctx: ctx, handle,
            p_val, p_spmv_buf,
            p_ilu_val, p_ilu_buf, info_ilu, descr_M,
            mat_L, mat_U, spsv_L, spsv_U, p_spsv_buf_L, p_spsv_buf_U,
            p_row, p_col,
            p_r, p_y, p_z, p_x, p_kp,
            n, nnz, diag,
        })
    }

    /// Compute kp = K·p  (GPU SpMV).
    pub fn matvec(&self, p: &[f64], kp: &mut [f64]) -> Result<(), String> {
        unsafe {
            result::memcpy_htod_sync(self.p_x, p)
                .map_err(|e| format!("[gpu] upload p: {e}"))?;

            let mut sp_mat: cusp::cusparseSpMatDescr_t = std::ptr::null_mut();
            cusp::cusparseCreateCsr(
                &mut sp_mat,
                self.n as i64, self.n as i64, self.nnz as i64,
                self.p_row as *mut std::ffi::c_void,
                self.p_col as *mut std::ffi::c_void,
                self.p_val as *mut std::ffi::c_void,
                cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                cusp::cudaDataType::CUDA_R_64F,
            ).result().map_err(|e| format!("[gpu] CreateCsr: {e:?}"))?;

            let (mut dn_x, mut dn_kp) = (std::ptr::null_mut(), std::ptr::null_mut());
            cusp::cusparseCreateDnVec(&mut dn_x,  self.n as i64, self.p_x  as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();
            cusp::cusparseCreateDnVec(&mut dn_kp, self.n as i64, self.p_kp as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();

            let (a, b) = (1.0_f64, 0.0_f64);
            let st = cusp::cusparseSpMV(
                self.handle,
                cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a as *const f64 as _, sp_mat, dn_x,
                &b as *const f64 as _, dn_kp,
                cusp::cudaDataType::CUDA_R_64F,
                cusp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                self.p_spmv_buf as *mut _,
            );
            cusp::cusparseDestroySpMat(sp_mat);
            cusp::cusparseDestroyDnVec(dn_x);
            cusp::cusparseDestroyDnVec(dn_kp);
            st.result().map_err(|e| format!("[gpu] SpMV: {e:?}"))?;

            result::memcpy_dtoh_sync(kp, self.p_kp)
                .map_err(|e| format!("[gpu] download kp: {e}"))?;
        }
        Ok(())
    }

    /// Apply ILU(0) preconditioner: z = U⁻¹·(L⁻¹·r).
    pub fn precondition(&self, r: &[f64], z: &mut [f64]) -> Result<(), String> {
        let n   = self.n as i64;
        let one = 1.0_f64;

        unsafe {
            result::memcpy_htod_sync(self.p_r, r)
                .map_err(|e| format!("[gpu] upload r: {e}"))?;

            // Build transient DnVec descriptors (cheap — just pointer wrappers)
            let mk_const = |p: CUdev| -> cusp::cusparseConstDnVecDescr_t {
                let mut d: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
                cusp::cusparseCreateDnVec(&mut d, n, p as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();
                d as cusp::cusparseConstDnVecDescr_t
            };
            let mk_mut = |p: CUdev| -> cusp::cusparseDnVecDescr_t {
                let mut d: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
                cusp::cusparseCreateDnVec(&mut d, n, p as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();
                d
            };

            // L·y = r  (forward)
            let dn_r  = mk_const(self.p_r);
            let dn_y  = mk_mut(self.p_y);
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
            st_l.result().map_err(|e| format!("[gpu] SpSV L solve: {e:?}"))?;

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
            st_u.result().map_err(|e| format!("[gpu] SpSV U solve: {e:?}"))?;

            result::memcpy_dtoh_sync(z, self.p_z)
                .map_err(|e| format!("[gpu] download z: {e}"))?;
        }
        Ok(())
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Allocate device memory of `bytes` and upload `src` to it.
unsafe fn alloc_upload<T>(bytes: usize, src: &[T]) -> Result<CUdev, String> {
    let p = result::malloc_sync(bytes)
        .map_err(|e| format!("[gpu] malloc {bytes}B: {e}"))?;
    result::memcpy_htod_sync(p, src)
        .map_err(|e| format!("[gpu] upload: {e}"))?;
    Ok(p)
}

/// Run ILU(0) factorization on p_ilu_val in-place.
/// Returns (descr_M, info_ilu, p_ilu_buf).
unsafe fn run_ilu02(
    handle:    cusp::cusparseHandle_t,
    p_row:     CUdev,
    p_col:     CUdev,
    p_ilu_val: CUdev,
    n: usize, nnz: usize,
) -> Result<(cusp::cusparseMatDescr_t, cusp::csrilu02Info_t, CUdev), String> {
    // General matrix descriptor (no fill mode needed for ILU factorization)
    let mut descr_M: cusp::cusparseMatDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateMatDescr(&mut descr_M)
        .result().map_err(|e| format!("[gpu] CreateMatDescr M: {e:?}"))?;
    cusp::cusparseSetMatIndexBase(descr_M, cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO).result().ok();
    cusp::cusparseSetMatType(descr_M, cusp::cusparseMatrixType_t::CUSPARSE_MATRIX_TYPE_GENERAL).result().ok();

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

/// Create mat_L (lower/unit) and mat_U (upper/non-unit) sparse descriptors
/// pointing into the ILU factor array.  Attributes are set via SpMatSetAttribute.
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

/// Run cusparseSpSV_analysis for one triangular factor.
/// Returns (spsv_descr, p_buffer).
/// p_x_in / p_y_out are placeholder device pointers (just for size queries).
unsafe fn spsv_analyze(
    handle: cusp::cusparseHandle_t,
    mat:    cusp::cusparseSpMatDescr_t,
    p_x_in: CUdev,
    p_y_out: CUdev,
    n: usize,
) -> Result<(cusp::cusparseSpSVDescr_t, CUdev), String> {
    let mut descr: cusp::cusparseSpSVDescr_t = std::ptr::null_mut();
    cusp::cusparseSpSV_createDescr(&mut descr)
        .result().map_err(|e| format!("[gpu] SpSV_createDescr: {e:?}"))?;

    let mut dn_x: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    let mut dn_y: cusp::cusparseDnVecDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateDnVec(&mut dn_x, n as i64, p_x_in  as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();
    cusp::cusparseCreateDnVec(&mut dn_y, n as i64, p_y_out as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();

    let one = 1.0_f64;
    let mat_c = mat as cusp::cusparseConstSpMatDescr_t;
    let x_c   = dn_x as cusp::cusparseConstDnVecDescr_t;

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

/// Query and allocate the cuSPARSE SpMV scratch buffer for K.
unsafe fn query_spmv_buf(
    handle: cusp::cusparseHandle_t,
    p_row: CUdev, p_col: CUdev, p_val: CUdev,
    p_x: CUdev, p_kp: CUdev,
    n: usize, nnz: usize,
) -> Result<CUdev, String> {
    let mut sp_mat: cusp::cusparseSpMatDescr_t = std::ptr::null_mut();
    cusp::cusparseCreateCsr(
        &mut sp_mat, n as i64, n as i64, nnz as i64,
        p_row as *mut _, p_col as *mut _, p_val as *mut _,
        cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
        cusp::cusparseIndexType_t::CUSPARSE_INDEX_32I,
        cusp::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
        cusp::cudaDataType::CUDA_R_64F,
    ).result().map_err(|e| format!("[gpu] spmv CreateCsr: {e:?}"))?;

    let (mut dn_x, mut dn_kp) = (std::ptr::null_mut(), std::ptr::null_mut());
    cusp::cusparseCreateDnVec(&mut dn_x,  n as i64, p_x  as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();
    cusp::cusparseCreateDnVec(&mut dn_kp, n as i64, p_kp as *mut _, cusp::cudaDataType::CUDA_R_64F).result().ok();

    let (a, b) = (1.0_f64, 0.0_f64);
    let mut buf_sz: usize = 0;
    cusp::cusparseSpMV_bufferSize(
        handle,
        cusp::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a as *const f64 as _, sp_mat, dn_x,
        &b as *const f64 as _, dn_kp,
        cusp::cudaDataType::CUDA_R_64F,
        cusp::cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
        &mut buf_sz,
    ).result().map_err(|e| format!("[gpu] spmv_bufferSize: {e:?}"))?;

    cusp::cusparseDestroySpMat(sp_mat);
    cusp::cusparseDestroyDnVec(dn_x);
    cusp::cusparseDestroyDnVec(dn_kp);

    result::malloc_sync(buf_sz.max(1))
        .map_err(|e| format!("[gpu] malloc spmv_buf: {e}"))
}

impl Drop for GpuK {
    fn drop(&mut self) {
        unsafe {
            for p in [self.p_val, self.p_spmv_buf, self.p_ilu_val, self.p_ilu_buf,
                      self.p_spsv_buf_L, self.p_spsv_buf_U,
                      self.p_row, self.p_col,
                      self.p_r, self.p_y, self.p_z, self.p_x, self.p_kp] {
                if p != 0 { result::free_sync(p).ok(); }
            }
            if !self.info_ilu.is_null()    { cusp::cusparseDestroyCsrilu02Info(self.info_ilu); }
            if !self.descr_M.is_null()     { cusp::cusparseDestroyMatDescr(self.descr_M); }
            if !self.mat_L.is_null()       { cusp::cusparseDestroySpMat(self.mat_L); }
            if !self.mat_U.is_null()       { cusp::cusparseDestroySpMat(self.mat_U); }
            if !self.spsv_L.is_null()      { cusp::cusparseSpSV_destroyDescr(self.spsv_L); }
            if !self.spsv_U.is_null()      { cusp::cusparseSpSV_destroyDescr(self.spsv_U); }
            if !self.handle.is_null()      { cusp::cusparseDestroy(self.handle); }
        }
    }
}
