.PHONY: setup check build up down run test clean-outputs shell logs status import-stl

# ── First-time setup ────────────────────────────────────────────────────────
setup:
	@echo ""
	@echo "Run this from a Windows PowerShell terminal (not WSL2):"
	@echo ""
	@echo "  powershell.exe -ExecutionPolicy Bypass -File scripts/setup_wsl.ps1"
	@echo ""
	@echo "Then return here and run: make check"

# ── Pre-flight validation ───────────────────────────────────────────────────
check:
	@bash scripts/check_environment.sh

# ── Docker ──────────────────────────────────────────────────────────────────
build:
	docker compose build

up:
	docker compose up -d
	@echo "JupyterLab: http://localhost:8888"

down:
	docker compose down

# ── Headless pipeline run ────────────────────────────────────────────────────
run:
	docker compose exec fenics-pipeline papermill \
		/workspace/pipeline_full.ipynb \
		/workspace/outputs/executed_nbs/pipeline_full_$(shell date +%Y%m%d_%H%M%S).ipynb \
		--kernel fenics-pipeline

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	docker compose exec fenics-pipeline pytest tests/ -v

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean-outputs:
	rm -rf outputs/meshes/* outputs/stl/* outputs/reports/* outputs/executed_nbs/*

# ── Developer utilities ──────────────────────────────────────────────────────
shell:
	docker compose exec fenics-pipeline bash

logs:
	tail -f outputs/logs/*.log

# ── Pipeline status ──────────────────────────────────────────────────────────
status:
	docker compose exec fenics-pipeline python scripts/check_pipeline_status.py

status-part:
	docker compose exec fenics-pipeline python scripts/check_pipeline_status.py --part $(PART)

# ── STL import ───────────────────────────────────────────────────────────────
# Usage:
#   make import-stl STL=path/to/file.stl PART=my_part SIZE=2.0 FACE=top LOAD=1000
#
# After running, open 02_mesh_gmsh.ipynb and run all cells.
# STL is copied to outputs/meshes/<PART>.stl and a stage01 handoff is written.

import-stl:
ifndef STL
	$(error STL is required. Usage: make import-stl STL=path/to/file.stl PART=name SIZE=2.0 FACE=top LOAD=1000)
endif
ifndef PART
	$(error PART is required. Usage: make import-stl STL=path/to/file.stl PART=name SIZE=2.0 FACE=top LOAD=1000)
endif
	docker compose exec fenics-pipeline python - <<'EOF'
import json, shutil, sys
from pathlib import Path

stl   = Path("$(STL)")
part  = "$(PART)"
size  = float("$(or $(SIZE),2.0)")
face  = "$(or $(FACE),top)"
load  = float("$(or $(LOAD),1000.0)")

if not stl.exists():
    print(f"✗ STL not found: {stl}")
    sys.exit(1)

valid_faces = {"top", "bottom", "left", "right", "front", "back"}
if face not in valid_faces:
    print(f"✗ FACE must be one of {valid_faces}, got: {face}")
    sys.exit(1)

dest = Path("outputs/meshes") / f"{part}.stl"
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(stl, dest)

handoff = {
    "stage":       "01_geometry",
    "origin":      "imported_stl",
    "source_file": str(stl.resolve()),
    "part_name":   part,
    "stl_path":    str(dest),
    "stl_size_kb": round(dest.stat().st_size / 1024, 2),
    "duration_s":  0.0,
    "mesh_hints": {
        "target_element_size": size,
        "refinement_regions":  [],
    },
    "load_hints": {
        "primary_face":     face,
        "load_magnitude_n": load,
    },
}

handoff_path = Path("outputs/meshes") / f"{part}_stage01.json"
handoff_path.write_text(json.dumps(handoff, indent=2))

print(f"✓ STL copied:      {dest}  ({handoff['stl_size_kb']} KB)")
print(f"✓ Handoff written: {handoff_path}")
print(f"")
print(f"  Next: open 02_mesh_gmsh.ipynb and run all cells.")
EOF