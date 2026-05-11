// solver/vendor/amgcl_wrapper.cpp
//
// Thin C API wrapping AMGCL algebraic multigrid for Rust FFI.
//
// ── Smoother choice history ───────────────────────────────────────────────────
//
// v1: spai0  — diverged (res > 1); too weak for 1M-DOF elasticity with SIMP
//              stiffness contrast.  The sparse approximate inverse (order 0)
//              cannot adequately damp high-frequency errors in the fine-grid
//              system, leaving the V-cycle near-indefinite.
//
// v2: spai0 + block_size=3 — residuals declining (12→4.7→2.8) but still
//              failing to converge within 200 iters.  Near-null-space is now
//              correct but smoother is still the bottleneck.
//
// v3 (this): ilu0 + block_size=3 — ILU(0) smoother is the standard
//              recommendation for AMG on structural elasticity systems.
//              Each V-cycle costs more than SPAI0 but requires 5-15× fewer
//              CG iterations to reach tolerance.  Expected: 20-80 iters.
//
// ── Phase B CUDA note ─────────────────────────────────────────────────────────
// amgcl::relaxation::ilu0 is CPU-only.  For CUDA backend, switch to
// amgcl::relaxation::spai1 (first-order SPAI, stronger than spai0, GPU-able).

#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include <stdexcept>

#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/ilu0.hpp>          // ← was spai0; ilu0 is much stronger
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

typedef amgcl::backend::builtin<double> Backend;

typedef amgcl::make_solver<
    amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::ilu0                // ← was spai0
    >,
    amgcl::solver::cg<Backend>
> SolverType;

struct AmgclHandle {
    int    n, nnz, dofs_per_node;
    double tol;
    int    max_iter;
    std::vector<int>    row_ptr, col_idx;
    std::vector<double> vals;
    std::unique_ptr<SolverType> solver;
    char last_error[512];
};

static void rebuild(AmgclHandle* h) {
    h->solver.reset();
    h->last_error[0] = '\0';
    try {
        SolverType::params prm;
        prm.solver.tol     = h->tol;
        prm.solver.maxiter = static_cast<size_t>(h->max_iter);

        // block_size=3: group u_x/u_y/u_z DOFs per node so smoothed_aggregation
        // preserves rigid body modes in the coarse hierarchy.  Required for
        // any 3D vector-valued FEM problem.
        prm.precond.coarsening.aggr.block_size = h->dofs_per_node;

        auto A = std::tie(h->n, h->row_ptr, h->col_idx, h->vals);
        h->solver.reset(new SolverType(A, prm));
    } catch (const std::exception& e) {
        snprintf(h->last_error, sizeof(h->last_error), "rebuild: %s", e.what());
    } catch (...) {
        snprintf(h->last_error, sizeof(h->last_error), "rebuild: unknown exception");
    }
}

extern "C" {

AmgclHandle* amgcl_create(
    int n, int nnz, int dofs_per_node,
    const int* row_ptr, const int* col_idx, const double* vals,
    double tol, int max_iter)
{
    AmgclHandle* h = new (std::nothrow) AmgclHandle();
    if (!h) return nullptr;
    h->n = n; h->nnz = nnz; h->dofs_per_node = dofs_per_node;
    h->tol = tol; h->max_iter = max_iter; h->last_error[0] = '\0';
    try {
        h->row_ptr.assign(row_ptr, row_ptr + n + 1);
        h->col_idx.assign(col_idx, col_idx + nnz);
        h->vals   .assign(vals,    vals    + nnz);
    } catch (...) { delete h; return nullptr; }
    rebuild(h);
    return h;
}

int amgcl_update(AmgclHandle* h, const double* vals) {
    if (!h) return 1;
    std::memcpy(h->vals.data(), vals,
                static_cast<size_t>(h->nnz) * sizeof(double));
    rebuild(h);
    return h->solver ? 0 : 1;
}

int amgcl_solve(
    AmgclHandle* h, const double* rhs, double* x,
    int* out_iters, double* out_residual)
{
    if (!h || !h->solver) return 1;
    std::vector<double> rhs_v(rhs, rhs + h->n);
    std::vector<double> x_v  (x,   x   + h->n);
    size_t iters = 0; double residual = 0.0;
    try {
        std::tie(iters, residual) = (*h->solver)(rhs_v, x_v);
    } catch (const std::exception& e) {
        snprintf(h->last_error, sizeof(h->last_error), "solve: %s", e.what());
        return 1;
    }
    std::memcpy(x, x_v.data(), static_cast<size_t>(h->n) * sizeof(double));
    if (out_iters)    *out_iters    = static_cast<int>(iters);
    if (out_residual) *out_residual = residual;
    return 0;
}

const char* amgcl_last_error(const AmgclHandle* h) {
    return h ? h->last_error : "null handle";
}

void amgcl_destroy(AmgclHandle* h) { delete h; }

}  // extern "C"
