// src/main.rs
//
// CLI entry point for simp_solver.
// Usage: simp_solver <path/to/problem.json>
//
// Reads problem.json and binary inputs, runs SIMP, writes density.bin,
// result.json, and periodic checkpoint files to the same directory as
// problem.json.
//
// Checkpoint / resume behaviour:
//   On every cfg.checkpoint_every iterations, checkpoint.bin and
//   checkpoint_meta.json are written to out_dir.  If the process is
//   interrupted (WSL close, Ctrl+C, OOM kill), the next invocation with
//   the same problem.json will detect the checkpoint and resume from the
//   last completed iteration rather than starting from scratch.
//   On clean exit, checkpoint files are deleted automatically.

mod assembly;
mod connectivity;
mod filter;
mod io;
mod ke_base;
mod oc_update;
mod multigrid;
mod preconditioner;
mod sensitivity;
mod simp;
mod solver;
mod types;
mod vcycle_dispatch;

#[cfg(feature = "amgcl")]
mod amgcl_solver;

#[cfg(feature = "gpu")]
mod gpu_solver;

use std::path::Path;
use io::{load_problem, write_density, write_result};
use simp::run_simp;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <problem.json>", args[0]);
        std::process::exit(1);
    }

    let json_path = Path::new(&args[1]);
    let out_dir   = json_path.parent().unwrap_or(Path::new("."));

    let problem = match load_problem(json_path) {
        Ok(p)  => p,
        Err(e) => { eprintln!("Error loading problem: {e}"); std::process::exit(1); }
    };

    println!("Grid: {}×{}×{}  ({} elements, {} DOFs)",
        problem.grid.nx, problem.grid.ny, problem.grid.nz,
        problem.grid.n_elem(), problem.grid.n_dof());
    println!("Material: E={:.3e} Pa, ν={}", problem.material.young, problem.material.poisson);
    println!("Config: vf={}, penal={}, r_filter={:.4}m, max_iter={}, checkpoint_every={}",
        problem.config.volume_fraction, problem.config.penal,
        problem.config.filter_radius, problem.config.max_iterations,
        problem.config.checkpoint_every);

    // out_dir is passed to run_simp so it can read/write checkpoint files.
    // All outputs (density.bin, result.json, checkpoint.*) land in the same
    // directory as problem.json, keeping outputs self-contained per stage.
    let result = run_simp(&problem, out_dir);

    if let Err(e) = write_density(&out_dir.join("density.bin"), &result.final_density) {
        eprintln!("Error writing density.bin: {e}"); std::process::exit(1);
    }
    if let Err(e) = write_result(&out_dir.join("result.json"), &result) {
        eprintln!("Error writing result.json: {e}"); std::process::exit(1);
    }

    println!("Wrote density.bin and result.json to {:?}", out_dir);
    std::process::exit(0);
}
