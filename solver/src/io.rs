// src/io.rs
//
// JSON and binary I/O for the SIMP solver.
//
// Python writes problem.json + binary files → Rust reads → solves → writes outputs.
// All binary files are little-endian, no header, flat arrays.
//
// Checkpoint format (written every cfg.checkpoint_every iters):
//   {out_dir}/checkpoint.bin       — f32 density, same format as density.bin
//   {out_dir}/checkpoint_meta.json — iter_completed, compliance/volume histories

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::{Grid, LoadCase, Material, Problem, SimpConfig};

// ─── JSON schema (mirrors problem.json exactly) ───────────────────────────────

#[derive(Deserialize)]
struct ProblemJson {
    grid:           GridJson,
    material:       MaterialJson,
    config:         ConfigJson,
    loading:        LoadingJson,
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
fn default_use_gpu()        -> bool  { true }
fn default_max_cg_iter()    -> usize { 2000 }

#[derive(Deserialize)]
struct ConfigJson {
    #[serde(default = "default_use_gpu")]
    use_gpu:               bool,
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
    #[serde(default = "default_max_cg_iter")]
    max_cg_iter:           usize,
}

fn default_weight() -> f64 { 1.0 }

#[derive(Deserialize)]
struct LoadingJson {
    /// Shared supports across all load cases.
    fixed_dofs_file: String,
    load_cases:      Vec<LoadCaseEntryJson>,
}

#[derive(Deserialize)]
struct LoadCaseEntryJson {
    #[serde(default)]
    name:            String,
    #[serde(default = "default_weight")]
    weight:          f64,
    load_dofs_file:  String,
    load_vals_file:  String,
    /// RESERVED — per-case supports (wishlist). Absent on the shared path.
    #[serde(default)]
    fixed_dofs_file: Option<String>,
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

// ─── Checkpoint types ─────────────────────────────────────────────────────────

/// Metadata written alongside checkpoint.bin.
/// Sufficient to fully resume a SIMP run: density comes from checkpoint.bin,
/// histories and iter count come from here.
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckpointMeta {
    /// Number of SIMP iterations fully completed (not including in-flight).
    pub iter_completed:     usize,
    /// Compliance history up to and including iter_completed.
    pub compliance_history: Vec<f64>,
    /// Volume fraction history up to and including iter_completed.
    pub volume_history:     Vec<f64>,
    /// Element count — used to validate checkpoint matches current problem.
    pub n_elem:             usize,
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

    let fixed_dofs: Vec<usize> = read_u32_le(&p(&pj.loading.fixed_dofs_file))?
        .iter().map(|&x| x as usize).collect();

    let mut load_cases: Vec<LoadCase> = Vec::with_capacity(pj.loading.load_cases.len());
    for (i, lc) in pj.loading.load_cases.iter().enumerate() {
        // Per-case supports are reserved; reject if a file is provided (Phase 1).
        if lc.fixed_dofs_file.is_some() {
            return Err(format!(
                "load case {i}: per-case fixed_dofs_file is reserved and not yet \
                 supported (shared supports only)"
            ));
        }
        let load_dofs: Vec<usize> = read_u32_le(&p(&lc.load_dofs_file))?
            .iter().map(|&x| x as usize).collect();
        let load_vals = read_f64_le(&p(&lc.load_vals_file))?;
        let name = if lc.name.is_empty() { format!("lc{i}") } else { lc.name.clone() };
        let case = LoadCase { name, weight: lc.weight, load_dofs, load_vals, fixed_dofs: None };
        case.validate()?;
        load_cases.push(case);
    }
    if load_cases.is_empty() {
        return Err("loading.load_cases must contain at least one load case".to_string());
    }

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
        fixed_dofs,
        load_cases,
        config: SimpConfig {
            use_gpu:               pj.config.use_gpu,
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
            max_cg_iter:           pj.config.max_cg_iter,
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

/// Write a mid-run checkpoint.
/// Two files are written atomically (meta last, so a partial write of the
/// binary doesn't leave a valid meta pointing to stale data):
///   {out_dir}/checkpoint.bin       — f32 density
///   {out_dir}/checkpoint_meta.json — CheckpointMeta
pub fn write_checkpoint(out_dir: &Path, density: &[f64], meta: &CheckpointMeta)
    -> Result<(), String>
{
    // Write density first (largest, most likely to be interrupted)
    write_density(&out_dir.join("checkpoint.bin"), density)?;
    // Write meta last — its presence is the "checkpoint is valid" signal
    let json = serde_json::to_string_pretty(meta)
        .map_err(|e| format!("Checkpoint meta serialise error: {e}"))?;
    fs::write(out_dir.join("checkpoint_meta.json"), json)
        .map_err(|e| format!("Cannot write checkpoint_meta.json: {e}"))?;
    Ok(())
}

/// Try to read a valid checkpoint from {out_dir}.
/// Returns None if checkpoint files are absent, mismatched, or corrupt —
/// in all such cases the caller should start from scratch.
pub fn read_checkpoint(out_dir: &Path, n_elem: usize)
    -> Option<(Vec<f64>, CheckpointMeta)>
{
    let bin_path  = out_dir.join("checkpoint.bin");
    let meta_path = out_dir.join("checkpoint_meta.json");

    if !bin_path.exists() || !meta_path.exists() {
        return None;
    }

    let meta_str = fs::read_to_string(&meta_path).ok()?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_str).ok()?;

    // Validate element count — catches stale checkpoints from different runs
    if meta.n_elem != n_elem {
        eprintln!(
            "[checkpoint] stale checkpoint (n_elem={} vs expected {}) — ignored",
            meta.n_elem, n_elem
        );
        return None;
    }

    let vals_f32 = read_f32_le(&bin_path).ok()?;
    if vals_f32.len() != n_elem {
        eprintln!("[checkpoint] checkpoint.bin has {} elements, expected {} — ignored",
                  vals_f32.len(), n_elem);
        return None;
    }

    let density: Vec<f64> = vals_f32.iter().map(|&v| v as f64).collect();
    Some((density, meta))
}

/// Delete checkpoint files after a clean run completion.
/// Errors are silently ignored — stale checkpoints are harmless on next run
/// (they'll be rejected by the n_elem validation).
pub fn delete_checkpoint(out_dir: &Path) {
    let _ = fs::remove_file(out_dir.join("checkpoint.bin"));
    let _ = fs::remove_file(out_dir.join("checkpoint_meta.json"));
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