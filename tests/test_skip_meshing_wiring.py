"""
tests/test_skip_meshing_wiring.py

Regression tests for the SKIP_MESHING / SKIP_FEA flag propagation across
NB02 → NB03 → NB04.

Cheap structural tests: parse each notebook and verify the guards are in
the expected cells. Does not execute any cell (no FEniCSx / gmsh required).
"""
from pathlib import Path

import nbformat
import pytest

REPO = Path(__file__).resolve().parent.parent
NB02 = REPO / "notebooks/02_mesh_gmsh.ipynb"
NB03 = REPO / "notebooks/03_fea_fenicsx.ipynb"
NB04 = REPO / "notebooks/04_simp_optimization.ipynb"


def _cell_source(nb_path: Path, idx: int) -> str:
    nb = nbformat.read(nb_path, as_version=4)
    assert idx < len(nb.cells), f"{nb_path.name} has {len(nb.cells)} cells, requested [{idx}]"
    c = nb.cells[idx]
    assert c.cell_type == "code", f"{nb_path.name} cell[{idx}] is {c.cell_type}, expected code"
    return c.source


# ─────────────────────────────────────────────────────────────────────────────
# NB02 — SKIP_MESHING flag and guards
# ─────────────────────────────────────────────────────────────────────────────

def test_nb02_params_declares_skip_meshing_default_false():
    src = _cell_source(NB02, 2)
    assert "SKIP_MESHING" in src, "SKIP_MESHING flag missing from NB02 params cell"
    # Default must be False — legacy FEniCSx path must not silently regress.
    assert "SKIP_MESHING     = False" in src or "SKIP_MESHING = False" in src, \
        "SKIP_MESHING default must be False (preserves legacy behavior)"


def test_nb02_meshing_cell_guarded():
    src = _cell_source(NB02, 6)
    assert "if SKIP_MESHING:" in src
    # The run_meshing_pipeline call must still be present (under the else branch)
    assert "run_meshing_pipeline" in src


def test_nb02_quality_cell_guarded():
    src = _cell_source(NB02, 8)
    assert "if SKIP_MESHING:" in src
    assert "check_mesh_quality" in src


def test_nb02_viz_cell_guarded():
    src = _cell_source(NB02, 10)
    assert "if SKIP_MESHING:" in src


def test_nb02_handoff_writes_skipped_flag():
    src = _cell_source(NB02, 13)
    assert '"skipped":         True' in src or '"skipped": True' in src
    assert "SKIP_MESHING" in src, "Handoff cell must branch on SKIP_MESHING"


# ─────────────────────────────────────────────────────────────────────────────
# NB03 — auto-propagation via SKIP_FEA
# ─────────────────────────────────────────────────────────────────────────────

def test_nb03_detects_upstream_skip():
    src = _cell_source(NB03, 4)
    assert "SKIP_FEA" in src
    assert 'handoff.get("skipped"' in src, \
        "NB03 must read skipped flag from upstream stage02 handoff"


def test_nb03_solve_cell_guarded():
    src = _cell_source(NB03, 8)
    assert "if SKIP_FEA:" in src
    assert "run_fea" in src


def test_nb03_safety_factor_cell_guarded():
    src = _cell_source(NB03, 10)
    assert "if SKIP_FEA:" in src


def test_nb03_render_cell_guarded():
    src = _cell_source(NB03, 12)
    assert "if SKIP_FEA:" in src


def test_nb03_handoff_writes_skipped_passthrough():
    src = _cell_source(NB03, 14)
    assert "if SKIP_FEA:" in src
    assert '"skipped":            True' in src or '"skipped": True' in src


def test_nb03_heatmap_cell_guarded():
    src = _cell_source(NB03, 15)
    assert "if SKIP_FEA:" in src, "NB03 heatmap cell must guard on SKIP_FEA"


# ─────────────────────────────────────────────────────────────────────────────
# NB04 — xdmf asserts gated on USE_RUST_SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def test_nb04_xdmf_assert_guarded_on_rust_path():
    src = _cell_source(NB04, 4)
    # The xdmf existence asserts must be inside `if not USE_RUST_SOLVER:`
    assert "if not USE_RUST_SOLVER:" in src
    assert "xdmf_path.exists()" in src
    assert "boundaries_xdmf.exists()" in src


def test_nb04_rust_solver_default_still_true():
    """Sanity: NB04 cell 0 still defaults USE_RUST_SOLVER to True."""
    src = _cell_source(NB04, 2)
    assert "USE_RUST_SOLVER" in src
    # Existing default per handoff
    assert "USE_RUST_SOLVER  = True" in src or "USE_RUST_SOLVER = True" in src


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: simulated flag propagation (structural — no execution)
# ─────────────────────────────────────────────────────────────────────────────

def test_skip_meshing_default_preserves_legacy_behavior():
    """With SKIP_MESHING=False (default), NB02 still calls run_meshing_pipeline,
    NB03 still calls run_fea, NB04 still asserts xdmf paths on the FEniCSx path.
    """
    # NB02 cell 6 reaches run_meshing_pipeline under the else branch
    assert "run_meshing_pipeline" in _cell_source(NB02, 6)
    # NB03 cell 8 reaches run_fea under the else branch
    assert "run_fea" in _cell_source(NB03, 8)
    # NB04 cell 4 still has the xdmf asserts (just gated)
    src04 = _cell_source(NB04, 4)
    assert "xdmf_path.exists()" in src04
    assert "boundaries_xdmf.exists()" in src04
