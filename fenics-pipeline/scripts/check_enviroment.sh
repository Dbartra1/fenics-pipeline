#!/usr/bin/env bash
# scripts/check_environment.sh
# Run from WSL2: bash scripts/check_environment.sh
# Or via: make check

set -euo pipefail

PASS="✓"
WARN="⚠"
FAIL="✗"
errors=0

echo "── fenics-pipeline environment check ──"
echo ""

# ── WSL2 memory ────────────────────────────────────────────────────────────
total_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
total_gb=$(( total_kb / 1024 / 1024 ))
if (( total_gb >= 16 )); then
    echo "  $PASS WSL2 memory: ${total_gb}GB"
else
    echo "  $FAIL WSL2 memory: ${total_gb}GB (need ≥16GB)"
    echo "       Run: powershell.exe -File scripts/setup_wsl.ps1"
    (( errors++ ))
fi

# ── Docker daemon ──────────────────────────────────────────────────────────
if docker info > /dev/null 2>&1; then
    echo "  $PASS Docker daemon reachable"
else
    echo "  $FAIL Docker daemon not running"
    (( errors++ ))
fi

# ── Docker Compose ─────────────────────────────────────────────────────────
if docker compose version > /dev/null 2>&1; then
    ver=$(docker compose version --short)
    echo "  $PASS Docker Compose: $ver"
else
    echo "  $FAIL docker compose not found (need v2+)"
    (( errors++ ))
fi

# ── /dev/shm size ──────────────────────────────────────────────────────────
# Docker inherits this from WSL2 — must be >1GB for MPI in 03_fea_fenicsx.ipynb
shm_kb=$(df /dev/shm | awk 'NR==2{print $2}')
shm_gb=$(echo "scale=1; $shm_kb / 1024 / 1024" | bc)
if (( shm_kb >= 1048576 )); then
    echo "  $PASS /dev/shm: ${shm_gb}GB"
else
    echo "  $WARN /dev/shm: ${shm_gb}GB (docker-compose.yml sets shm_size: 2gb — ok if Docker is not yet started)"
fi

# ── OpenSCAD (host-side, optional) ─────────────────────────────────────────
# openscad_runner.py uses the container's OpenSCAD — host install is optional
if command -v openscad > /dev/null 2>&1; then
    echo "  $PASS OpenSCAD (host): $(openscad --version 2>&1 | head -1)"
else
    echo "  $WARN OpenSCAD not on host PATH (fine — container has it)"
fi

# ── .env file ──────────────────────────────────────────────────────────────
if [[ -f .env ]]; then
    echo "  $PASS .env present"
    source .env
    if [[ -n "${NOTEBOOK_DIR:-}" ]]; then
        echo "       NOTEBOOK_DIR=${NOTEBOOK_DIR}"
    else
        echo "  $WARN NOTEBOOK_DIR not set in .env — defaulting to ./notebooks"
    fi
else
    echo "  $FAIL .env missing — copy from .env.example and edit"
    (( errors++ ))
fi

# ── Notebook directory ─────────────────────────────────────────────────────
nb_dir="${NOTEBOOK_DIR:-./notebooks}"
if [[ -d "$nb_dir" ]]; then
    count=$(find "$nb_dir" -name "*.ipynb" | wc -l)
    echo "  $PASS notebooks/: $count .ipynb files found"
else
    echo "  $FAIL $nb_dir not found"
    (( errors++ ))
fi

# ── outputs/ subdirs ───────────────────────────────────────────────────────
for subdir in outputs/meshes outputs/stl outputs/reports outputs/executed_nbs; do
    if [[ -d "$subdir" ]]; then
        echo "  $PASS $subdir/"
    else
        echo "  $FAIL $subdir/ missing — run: mkdir -p $subdir"
        (( errors++ ))
    fi
done

echo ""
if (( errors == 0 )); then
    echo "✅ All checks passed — run: make build && make up"
else
    echo "✗  $errors issue(s) found — fix above before proceeding"
    exit 1
fi