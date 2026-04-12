# tests/test_solver_interface.py
#
# Validates the Rust solver binary I/O contract.
# Tests only: binary reads problem.json, writes density.bin + result.json.
# Does NOT validate numerical correctness — that's the simp.rs unit tests.
#
# Run from repo root:
#   python -m pytest tests/test_solver_interface.py -v

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

SOLVER_BINARY = Path("bin/simp_solver")


def _require_binary():
    if not SOLVER_BINARY.exists():
        pytest.skip(f"Solver binary not found at {SOLVER_BINARY} — run `make build-solver`")


def _write_minimal_problem(d: Path, nx=10, ny=6, nz=4) -> dict:
    """Write a minimal valid problem to directory d. Returns the grid config."""
    n_elem  = nx * ny * nz
    n_nodes = (nx+1) * (ny+1) * (nz+1)

    # Bottom face fixed: all iz=0 nodes, all 3 DOFs
    bottom_nodes = [ix + iy*(nx+1) for iy in range(ny+1) for ix in range(nx+1)]
    fixed_dofs = np.array(
        [3*n + dd for n in bottom_nodes for dd in range(3)], dtype=np.uint32
    )
    fixed_dofs.tofile(d / "fixed_dofs.bin")

    # Top face loaded in -z
    top_nodes = [
        ix + iy*(nx+1) + nz*(nx+1)*(ny+1)
        for iy in range(ny+1) for ix in range(nx+1)
    ]
    load_dofs = np.array([3*n + 2 for n in top_nodes], dtype=np.uint32)
    load_vals = np.full(len(top_nodes), -10000.0 / len(top_nodes), dtype=np.float64)
    load_dofs.tofile(d / "load_dofs.bin")
    load_vals.tofile(d / "load_vals.bin")

    np.zeros(n_elem, dtype=np.uint8).tofile(d / "nondesign.bin")
    np.zeros(n_elem, dtype=np.uint8).tofile(d / "void.bin")

    problem = {
        "grid": {"nx": nx, "ny": ny, "nz": nz, "voxel_size": 0.0025},
        "material": {"young": 210e9, "poisson": 0.3},
        "config": {
            "volume_fraction": 0.5,
            "penal": 3.0,
            "filter_radius": 0.005,
            "max_iterations": 5,
            "convergence_tol": 0.01,
            "move_limit": 0.2,
            "damping": 0.5,
            "checkpoint_every": 0,
        },
        "load_case": {
            "fixed_dofs_file": "fixed_dofs.bin",
            "load_dofs_file":  "load_dofs.bin",
            "load_vals_file":  "load_vals.bin",
        },
        "nondesign_file": "nondesign.bin",
        "void_file":      "void.bin",
        "x_init_file":    None,
    }
    (d / "problem.json").write_text(json.dumps(problem, indent=2))
    return {"nx": nx, "ny": ny, "nz": nz}


def test_solver_binary_exists():
    _require_binary()
    assert SOLVER_BINARY.exists()


def test_solver_runs_and_produces_output():
    """Rust binary reads problem.json and writes density.bin + result.json."""
    _require_binary()
    nx, ny, nz = 10, 6, 4
    n_elem = nx * ny * nz

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_minimal_problem(d, nx, ny, nz)

        result = subprocess.run(
            [str(SOLVER_BINARY), str(d / "problem.json")],
            capture_output=True, text=True, timeout=120,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
        assert result.returncode == 0, f"Solver failed:\n{result.stderr}"

        assert (d / "density.bin").exists(),  "density.bin not written"
        assert (d / "result.json").exists(),  "result.json not written"

        density = np.fromfile(d / "density.bin", dtype=np.float32)
        assert len(density) == n_elem, f"Expected {n_elem} elements, got {len(density)}"
        assert density.min() >= 1e-3 - 1e-6,  f"density below rho_min: {density.min()}"
        assert density.max() <= 1.0 + 1e-6,   f"density above 1.0: {density.max()}"

        res = json.loads((d / "result.json").read_text())
        assert "converged"          in res
        assert "n_iterations"       in res
        assert "final_compliance"   in res
        assert "compliance_history" in res
        assert "volume_history"     in res
        assert len(res["compliance_history"]) > 0
        assert res["final_compliance"] > 0.0


def test_solver_stdout_format():
    """Solver stdout contains expected iteration log lines."""
    _require_binary()

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_minimal_problem(d)

        result = subprocess.run(
            [str(SOLVER_BINARY), str(d / "problem.json")],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0

        lines = result.stdout.strip().splitlines()
        iter_lines = [l for l in lines if l.strip().startswith("Iter")]
        assert len(iter_lines) > 0, f"No iteration lines found in stdout:\n{result.stdout}"


def test_solver_density_shape_matches_grid():
    """density.bin length matches nx*ny*nz exactly."""
    _require_binary()

    for nx, ny, nz in [(4, 3, 2), (10, 6, 4)]:
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_minimal_problem(d, nx, ny, nz)
            subprocess.run(
                [str(SOLVER_BINARY), str(d / "problem.json")],
                capture_output=True, timeout=120,
            )
            density = np.fromfile(d / "density.bin", dtype=np.float32)
            assert len(density) == nx * ny * nz, \
                f"Grid {nx}×{ny}×{nz}: expected {nx*ny*nz} elements, got {len(density)}"


def test_solver_exits_nonzero_on_bad_input():
    """Solver should exit with error on missing/invalid problem.json."""
    _require_binary()

    result = subprocess.run(
        [str(SOLVER_BINARY), "/nonexistent/problem.json"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0, "Solver should fail on missing input file"