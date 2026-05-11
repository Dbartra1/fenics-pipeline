// solver/build.rs
//
// Compile-time build script for simp_solver.
//
// When the `amgcl` feature is enabled (default):
//   - Compiles vendor/amgcl_wrapper.cpp with g++ + OpenMP + C++14.
//   - Links the resulting object into the simp_solver binary via a
//     static archive.  No shared-library install needed at runtime
//     beyond libgomp.so.1 (present on any Ubuntu WSL2 / dolfinx Docker).
//
// If AMGCL headers are missing (i.e. setup_amgcl.sh has not been run),
// the build fails with a clear human-readable error — not a cryptic
// compiler error.
//
// When the `gpu` feature is enabled:
//   - cudarc links are handled by cudarc's own build.rs.
//   - Nothing extra needed here for Phase A (OpenMP).
//   - Phase B (AMGCL CUDA) will add a second nvcc-compiled file here.

use std::env;
use std::path::Path;

fn main() {
    // ── amgcl OpenMP wrapper ──────────────────────────────────────────────────
    if env::var("CARGO_FEATURE_AMGCL").is_ok() {
        let amgcl_header = Path::new("vendor/amgcl/make_solver.hpp");
        if !amgcl_header.exists() {
            // Give a clear error before the compiler produces a wall of noise.
            eprintln!();
            eprintln!("╔══════════════════════════════════════════════════════════════╗");
            eprintln!("║  AMGCL headers not found at vendor/amgcl/amgcl.hpp           ║");
            eprintln!("║                                                              ║");
            eprintln!("║  Run once before building:                                   ║");
            eprintln!("║    bash scripts/setup_amgcl.sh                               ║");
            eprintln!("║                                                              ║");
            eprintln!("║  Or disable the amgcl feature (falls back to Jacobi-PCG):    ║");
            eprintln!("║    cargo build --release --no-default-features               ║");
            eprintln!("╚══════════════════════════════════════════════════════════════╝");
            eprintln!();
            std::process::exit(1);
        }

        cc::Build::new()
            .cpp(true)
            .opt_level(2)
            .flag("-std=c++14")
            .flag("-fopenmp")
            // Eliminate the Boost dependency entirely.  AMGCL's core AMG
            // hierarchy, builtin backend, smoothed_aggregation coarsening,
            // SPAI0 relaxation, and CG solver all work without Boost.
            // String-based parameter parsing (ptree) is the only thing
            // disabled — we set params directly on the C++ structs.
            .flag("-DAMGCL_NO_BOOST")
            // Suppress warnings from AMGCL headers (they're not our code).
            .flag("-w")
            // vendor/ is the include root: #include <amgcl/...> resolves
            // to vendor/amgcl/...
            .include("vendor")
            .file("vendor/amgcl_wrapper.cpp")
            .compile("amgcl_wrapper");

        // Link the OpenMP runtime so the AMGCL builtin backend can
        // parallelise SpMV and inner products across cores.
        println!("cargo:rustc-link-lib=gomp");

        // Rebuild if source or headers change.
        println!("cargo:rerun-if-changed=vendor/amgcl_wrapper.cpp");
        println!("cargo:rerun-if-changed=build.rs");
        // Glob would be ideal but cargo only supports explicit paths here.
        // The setup script is idempotent so a stale header set is low-risk.
        println!("cargo:rerun-if-changed=vendor/amgcl/make_solver.hpp");
    }

    // ── gpu feature: nothing extra needed ────────────────────────────────────
    // cudarc's own build.rs handles all CUDA linking.
    // Phase B AMGCL CUDA will add an nvcc invocation here.
}
