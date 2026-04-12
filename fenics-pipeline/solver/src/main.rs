// src/main.rs
//
// CLI entry point for simp_solver.
// Usage: simp_solver <path/to/problem.json>
//
// Reads problem.json and binary inputs, runs SIMP, writes density.bin
// and result.json to the same directory as problem.json.

mod assembly;
mod connectivity;
mod filter;
mod io;
mod ke_base;
mod oc_update;
mod sensitivity;
mod simp;
mod solver;
mod types;

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

    // Load
    let problem = match load_problem(json_path) {
        Ok(p)  => p,
        Err(e) => { eprintln!("Error loading problem: {e}"); std::process::exit(1); }
    };

    println!("Grid: {}×{}×{}  ({} elements, {} DOFs)",
        problem.grid.nx, problem.grid.ny, problem.grid.nz,
        problem.grid.n_elem(), problem.grid.n_dof());
    println!("Material: E={:.3e} Pa, ν={}", problem.material.young, problem.material.poisson);
    println!("Config: vf={}, penal={}, r_filter={:.4}m, max_iter={}",
        problem.config.volume_fraction, problem.config.penal,
        problem.config.filter_radius, problem.config.max_iterations);

    // Solve
    let result = run_simp(&problem);

    // Write outputs
    if let Err(e) = write_density(&out_dir.join("density.bin"), &result.final_density) {
        eprintln!("Error writing density.bin: {e}"); std::process::exit(1);
    }

    if let Err(e) = write_result(&out_dir.join("result.json"), &result) {
        eprintln!("Error writing result.json: {e}"); std::process::exit(1);
    }

    println!("Wrote density.bin and result.json to {:?}", out_dir);
    std::process::exit(if result.converged { 0 } else { 0 });  // always 0 for now
}