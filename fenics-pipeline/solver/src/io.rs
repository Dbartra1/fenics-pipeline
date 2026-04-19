// src/io.rs
//
// JSON and binary I/O for the SIMP solver.
//
// Python writes problem.json + binary files → Rust reads → solves → writes outputs.
// All binary files are little-endian, no header, flat arrays.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::types::{Grid, LoadCase, Material, Problem, SimpConfig};

// ─── JSON schema (mirrors problem.json exactly) ───────────────────────────────

#[derive(Deserialize)]
struct ProblemJson {
    grid:           GridJson,
    material:       MaterialJson,
    config:         ConfigJson,
    load_case:      LoadCaseJson,
    nondesign_file: String,
    void_file:      String,
    x_init_file:    Option<String>,
}

#[derive(Deserialize)]
struct GridJson {
    nx: usize, ny: usize, nz: usize,
    voxel_size: f64,
}

#[derive(Deserialize)]
struct MaterialJson {
    young: f64, poisson: f64,
}

fn default_min_iterations() -> usize { 10 }

#[derive(Deserialize)]
struct ConfigJson {
    volume_fraction:       f64,
    penal:                 f64,
    filter_radius:         f64,
    max_iterations:        usize,
    #[serde(default = "default_min_iterations")]
    min_iterations:        usize,
    convergence_tol:       f64,
    #[serde(default)]
    compliance_spread_tol: Option<f64>,
    #[serde(default)]
    density_change_tol:    Option<f64>,
    move_limit:            f64,
    damping:               f64,
    checkpoint_every:      usize,
}

#[derive(Deserialize)]
struct LoadCaseJson {
    fixed_dofs_file: String,
    load_dofs_file:  String,
    load_vals_file:  String,
}

// ─── Binary readers ───────────────────────────────────────────────────────────

fn read_u32_le(path: &Path) -> Result<Vec<u32>, String> {
    let mut buf = Vec::new();
    fs::File::open(path)
        .map_err(|e| format!("Cannot open {:?}: {e}", path))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error {:?}: {e}", path))?;
    if buf.len() % 4 != 0 {
        return Err(format!("{:?}: length {} not a multiple of 4", path, buf.len()));
    }
    Ok(buf.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f64_le(path: &Path) -> Result<Vec<f64>, String> {
    let mut buf = Vec::new();
    fs::File::open(path)
        .map_err(|e| format!("Cannot open {:?}: {e}", path))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error {:?}: {e}", path))?;
    if buf.len() % 8 != 0 {
        return Err(format!("{:?}: length {} not a multiple of 8", path, buf.len()));
    }
    Ok(buf.chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
    let mut buf = Vec::new();
    fs::File::open(path)
        .map_err(|e| format!("Cannot open {:?}: {e}", path))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error {:?}: {e}", path))?;
    if buf.len() % 4 != 0 {
        return Err(format!("{:?}: length {} not a multiple of 4", path, buf.len()));
    }
    Ok(buf.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn read_u8(path: &Path) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    fs::File::open(path)
        .map_err(|e| format!("Cannot open {:?}: {e}", path))?
        .read_to_end(&mut buf)
        .map_err(|e| format!("Read error {:?}: {e}", path))?;
    Ok(buf)
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Load a Problem from a problem.json file.
/// All binary paths are resolved relative to the directory containing problem.json.
pub fn load_problem(json_path: &Path) -> Result<Problem, String> {
    let json_str = fs::read_to_string(json_path)
        .map_err(|e| format!("Cannot read {:?}: {e}", json_path))?;
    let pj: ProblemJson = serde_json::from_str(&json_str)
        .map_err(|e| format!("JSON parse error in {:?}: {e}", json_path))?;

    let dir = json_path.parent().unwrap_or(Path::new("."));
    let p = |name: &str| -> PathBuf { dir.join(name) };

    let grid = Grid {
        nx: pj.grid.nx, ny: pj.grid.ny, nz: pj.grid.nz,
        voxel_size: pj.grid.voxel_size,
    };
    let n_elem = grid.n_elem();

    // Load case
    let fixed_dofs_u32 = read_u32_le(&p(&pj.load_case.fixed_dofs_file))?;
    let load_dofs_u32  = read_u32_le(&p(&pj.load_case.load_dofs_file))?;
    let load_vals      = read_f64_le(&p(&pj.load_case.load_vals_file))?;

    let load_case = LoadCase {
        fixed_dofs: fixed_dofs_u32.iter().map(|&x| x as usize).collect(),
        load_dofs:  load_dofs_u32.iter().map(|&x| x as usize).collect(),
        load_vals,
    };
    load_case.validate()?;

    // Masks
    let nondesign_u8 = read_u8(&p(&pj.nondesign_file))?;
    let void_u8      = read_u8(&p(&pj.void_file))?;

    if nondesign_u8.len() != n_elem {
        return Err(format!("nondesign_file: expected {n_elem} bytes, got {}", nondesign_u8.len()));
    }
    if void_u8.len() != n_elem {
        return Err(format!("void_file: expected {n_elem} bytes, got {}", void_u8.len()));
    }

    let nondesign: Vec<bool> = nondesign_u8.iter().map(|&b| b != 0).collect();
    let void_mask: Vec<bool> = void_u8.iter().map(|&b| b != 0).collect();

    // Optional warm start
    let x_init = if let Some(ref fname) = pj.x_init_file {
        let vals_f32 = read_f32_le(&p(fname))?;
        if vals_f32.len() != n_elem {
            return Err(format!("x_init_file: expected {n_elem} elements, got {}", vals_f32.len()));
        }
        Some(vals_f32.iter().map(|&v| v as f64).collect())
    } else {
        None
    };

    let problem = Problem {
        grid,
        material: Material { young: pj.material.young, poisson: pj.material.poisson },
        load_case,
        config: SimpConfig {
            volume_fraction:       pj.config.volume_fraction,
            penal:                 pj.config.penal,
            filter_radius:         pj.config.filter_radius,
            max_iterations:        pj.config.max_iterations,
            min_iterations:        pj.config.min_iterations,
            convergence_tol:       pj.config.convergence_tol,
            compliance_spread_tol: pj.config.compliance_spread_tol,
            density_change_tol:    pj.config.density_change_tol,
            move_limit:            pj.config.move_limit,
            damping:               pj.config.damping,
            checkpoint_every:      pj.config.checkpoint_every,
        },
        nondesign,
        void_mask,
        x_init,
    };

    problem.validate()?;
    Ok(problem)
}

/// Write density field as f32 little-endian binary.
pub fn write_density(path: &Path, density: &[f64]) -> Result<(), String> {
    let mut f = fs::File::create(path)
        .map_err(|e| format!("Cannot create {:?}: {e}", path))?;
    for &v in density {
        f.write_all(&(v as f32).to_le_bytes())
            .map_err(|e| format!("Write error {:?}: {e}", path))?;
    }
    Ok(())
}

/// Write result.json.
pub fn write_result(path: &Path, result: &SolveResult) -> Result<(), String> {
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| format!("JSON serialise error: {e}"))?;
    fs::write(path, json)
        .map_err(|e| format!("Cannot write {:?}: {e}", path))?;
    Ok(())
}

/// The result written to result.json.
///
/// `final_density` is excluded from JSON serialization: at 120k elements it
/// would embed ~3MB of float text into result.json which the notebook never
/// reads (it reads density.bin directly via np.fromfile). The field exists
/// on the struct so main.rs can call write_density() with it.
#[derive(serde::Serialize)]
pub struct SolveResult {
    pub converged:          bool,
    pub n_iterations:       usize,
    pub final_compliance:   f64,
    pub final_volume_frac:  f64,
    pub compliance_history: Vec<f64>,
    pub volume_history:     Vec<f64>,
    pub duration_s:         f64,
    pub peak_memory_mb:     f64,
    #[serde(skip)]
    pub final_density:      Vec<f64>,
}