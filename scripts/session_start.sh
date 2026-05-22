#!/usr/bin/env bash
# scripts/session_start.sh
#
# fenics-pipeline session startup script.
# Run this at the start of every work session from WSL2.
#
# What it does, in order:
#   1. Confirm we are in the right repo
#   2. Check / start the Docker container
#   3. Verify GPU passthrough
#   4. Check if the Rust solver binary is stale — rebuild if needed
#   5. Run the WSL-safe preflight test suite
#   6. Show current run state (any checkpoint or result from last session)
#   7. Prompt to queue a new NB04 run via the dashboard script
#
# Usage:
#   bash scripts/session_start.sh
#   bash scripts/session_start.sh --part motor_mount   # skip part prompt
#   bash scripts/session_start.sh --skip-build         # skip solver rebuild check
#   bash scripts/session_start.sh --skip-tests         # skip pytest
#
# Detach from tmux once the run starts: Ctrl+B, D
# Reattach:                             tmux attach -t simp_run

set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
R='\033[0;31m'
G='\033[0;32m'
Y='\033[0;33m'
B='\033[0;34m'
C='\033[0;36m'
W='\033[1;37m'
DIM='\033[2m'
X='\033[0m'

pass()  { echo -e "  ${G}✓${X}  $*"; }
fail()  { echo -e "  ${R}✗${X}  $*"; }
warn()  { echo -e "  ${Y}⚠${X}  $*"; }
info()  { echo -e "  ${C}→${X}  $*"; }
step()  { echo -e "\n${W}$*${X}"; echo "  $(printf '─%.0s' {1..56})"; }
die()   { echo -e "\n${R}FATAL:${X} $*\n"; exit 1; }

# ─── Arg parsing ──────────────────────────────────────────────────────────────
PART_OVERRIDE=""
SKIP_BUILD=false
SKIP_TESTS=false
BACKEND_OVERRIDE=""
FILTER_OVERRIDE=""
VF_OVERRIDE=""
MAX_ITER_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --part)        PART_OVERRIDE="$2";     shift 2 ;;
        --skip-build)  SKIP_BUILD=true;        shift   ;;
        --skip-tests)  SKIP_TESTS=true;        shift   ;;
        --backend)     BACKEND_OVERRIDE="$2";  shift 2 ;;
        --filter)      FILTER_OVERRIDE="$2";   shift 2 ;;
        --vf)          VF_OVERRIDE="$2";       shift 2 ;;
        --max-iter)    MAX_ITER_OVERRIDE="$2"; shift 2 ;;
        *)             echo "Unknown flag: $1"
                       echo "Usage: bash scripts/session_start.sh"
                       echo "         [--part NAME]"
                       echo "         [--backend auto|cpu]"
                       echo "         [--filter RADIUS_MM]"
                       echo "         [--vf VOLUME_FRACTION]"
                       echo "         [--max-iter N]"
                       echo "         [--skip-build] [--skip-tests]"
                       exit 1 ;;
    esac
done

# ─── Constants ────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="fenics-pipeline"
BINARY="$REPO_ROOT/bin/simp_solver"
SOLVER_SRC="$REPO_ROOT/solver/src"
SOLVER_VENDOR="$REPO_ROOT/solver/vendor"
BUILD_SCRIPT="$REPO_ROOT/scripts/build_solver.sh"
DASHBOARD_SCRIPT="$REPO_ROOT/scripts/simp_dashboard.sh"
RESULT_JSON="$REPO_ROOT/outputs/problem/result.json"
CHECKPOINT_BIN="$REPO_ROOT/outputs/problem/density.bin"
SCAD_DIR="$REPO_ROOT/scad"

# ─── Header ───────────────────────────────────────────────────────────────────
echo -e "\n${W}╔══════════════════════════════════════════════════════════╗${X}"
echo -e "${W}║        fenics-pipeline  —  session startup               ║${X}"
echo -e "${W}╚══════════════════════════════════════════════════════════╝${X}"
echo -e "${DIM}  $(date '+%A %Y-%m-%d  %H:%M')  •  $(hostname)${X}"

# ─── Step 1: Repo check ───────────────────────────────────────────────────────
step "1 / 7  Repo"

cd "$REPO_ROOT" || die "Cannot cd to $REPO_ROOT"

if [[ ! -f "docker-compose.yml" ]]; then
    die "docker-compose.yml not found — are you in the right directory?\n  Expected: $REPO_ROOT"
fi
pass "Repo root: $REPO_ROOT"

# Git state
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown")
DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')

info "Branch: $BRANCH"
info "HEAD:   $COMMIT"
if [[ "$DIRTY" -gt 0 ]]; then
    warn "$DIRTY uncommitted file(s) — consider committing before a long run"
    git status --short | head -10 | sed 's/^/          /'
else
    pass "Working tree clean"
fi

# ─── Step 2: Docker container ─────────────────────────────────────────────────
step "2 / 7  Docker container"

if ! docker info &>/dev/null; then
    die "Docker daemon not reachable. Start Docker Desktop and retry."
fi
pass "Docker daemon running"

CONTAINER_STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")

case "$CONTAINER_STATUS" in
    running)
        pass "Container '$CONTAINER' is running"
        ;;
    exited|created|paused)
        warn "Container '$CONTAINER' is $CONTAINER_STATUS — starting..."
        docker-compose up -d
        sleep 3
        STATUS2=$(docker inspect --format='{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
        if [[ "$STATUS2" != "running" ]]; then
            die "Container failed to start (status: $STATUS2). Run: docker-compose logs $CONTAINER"
        fi
        pass "Container started"
        ;;
    missing)
        warn "Container not found — building and starting..."
        docker-compose up -d --build
        sleep 5
        STATUS2=$(docker inspect --format='{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo "missing")
        if [[ "$STATUS2" != "running" ]]; then
            die "Container failed to start. Run: docker-compose logs $CONTAINER"
        fi
        pass "Container built and started"
        ;;
    *)
        die "Container in unexpected state: $CONTAINER_STATUS"
        ;;
esac

# ─── Step 3: GPU passthrough ──────────────────────────────────────────────────
step "3 / 7  GPU passthrough"

GPU_INFO=$(docker-compose exec -T "$CONTAINER" nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits 2>/dev/null || echo "FAILED")

if [[ "$GPU_INFO" == "FAILED" || -z "$GPU_INFO" ]]; then
    warn "nvidia-smi failed inside container — solver will fall back to CPU"
    warn "Check: nvidia-container-toolkit installed? CUDA driver up to date?"
else
    GPU_NAME=$(echo "$GPU_INFO" | awk -F', ' '{print $1}' | xargs)
    GPU_MEM=$(echo "$GPU_INFO"  | awk -F', ' '{print $2}' | xargs)
    GPU_DRV=$(echo "$GPU_INFO"  | awk -F', ' '{print $3}' | xargs)
    pass "GPU: $GPU_NAME  (${GPU_MEM} MB  driver $GPU_DRV)"
fi

# ─── Step 4: Rust solver binary ───────────────────────────────────────────────
step "4 / 7  Rust solver binary"

if [[ "$SKIP_BUILD" == true ]]; then
    info "Skipping build check (--skip-build)"
elif [[ ! -f "$BINARY" ]]; then
    warn "Binary not found at bin/simp_solver — building now..."
    docker-compose exec -T "$CONTAINER" bash scripts/build_solver.sh
    pass "Build complete"
else
    BINARY_AGE=$(( $(date +%s) - $(stat -c %Y "$BINARY") ))
    BINARY_AGE_H=$(( BINARY_AGE / 3600 ))

    # Find newest source file (solver/src/*.rs + vendor wrapper)
    NEWEST_SRC=$(find "$SOLVER_SRC" "$SOLVER_VENDOR" \
        -name "*.rs" -o -name "*.cpp" -o -name "*.toml" \
        2>/dev/null | xargs stat -c '%Y %n' 2>/dev/null \
        | sort -rn | head -1 | awk '{print $1}')
    BINARY_MTIME=$(stat -c %Y "$BINARY")

    if [[ -n "$NEWEST_SRC" && "$NEWEST_SRC" -gt "$BINARY_MTIME" ]]; then
        warn "Source files are newer than binary — rebuilding..."
        CHANGED_SRC=$(find "$SOLVER_SRC" "$SOLVER_VENDOR" \
            -name "*.rs" -o -name "*.cpp" -o -name "*.toml" \
            2>/dev/null | xargs stat -c '%Y %n' 2>/dev/null \
            | awk -v mt="$BINARY_MTIME" '$1 > mt {print "         " $2}' \
            | head -5)
        info "Changed files:"
        echo "$CHANGED_SRC"
        docker-compose exec -T "$CONTAINER" bash scripts/build_solver.sh
        pass "Rebuild complete"
    elif [[ "$BINARY_AGE_H" -gt 168 ]]; then
        # Older than 7 days — prompt
        warn "Binary is ${BINARY_AGE_H}h old (>7 days). Rebuild? [y/N]"
        read -r -t 10 REBUILD_ANS || REBUILD_ANS="n"
        if [[ "${REBUILD_ANS,,}" == "y" ]]; then
            docker-compose exec -T "$CONTAINER" bash scripts/build_solver.sh
            pass "Rebuild complete"
        else
            info "Skipping rebuild"
        fi
    else
        BINARY_DATE=$(stat -c '%y' "$BINARY" | cut -d'.' -f1)
        pass "Binary up to date  (built: $BINARY_DATE)"
    fi
fi

# Verify binary is executable inside the container
BIN_CHECK=$(docker-compose exec -T "$CONTAINER" \
    bash -c "test -x /workspace/bin/simp_solver && echo ok || echo missing" 2>/dev/null || echo "missing")
if [[ "$BIN_CHECK" != "ok" ]]; then
    fail "Binary not executable inside container — run: bash scripts/build_solver.sh"
fi

# ─── Step 5: Preflight tests ──────────────────────────────────────────────────
step "5 / 7  Preflight tests  (WSL-safe suite)"

if [[ "$SKIP_TESTS" == true ]]; then
    info "Skipping tests (--skip-tests)"
else
    TEST_OUTPUT=$(python3 -m pytest tests/ \
    --ignore=tests/test_slabs.py \
    --ignore=tests/test_fea_smoke.py \
    --ignore=tests/test_mesh_quality.py \
    -q --tb=no 2>&1) || true

    PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= passed)' | tail -1 || echo "0")
    FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= failed)' | tail -1 || echo "0")
    ERRORS=$(echo "$TEST_OUTPUT" | grep -oP '\d+(?= error)'  | tail -1 || echo "0")

    if [[ "$FAILED" == "0" && "$ERRORS" == "0" ]]; then
        pass "$PASSED tests passed, 0 failed"
    else
        fail "$PASSED passed, $FAILED failed, $ERRORS errors"
        echo ""
        echo "$TEST_OUTPUT" | grep -E "FAILED|ERROR" | head -10 | sed 's/^/         /'
        echo ""
        warn "Test failures detected. Continuing — but investigate before a long run."
    fi
fi

# ─── Step 6: Last run state ───────────────────────────────────────────────────
step "6 / 7  Last run state"

# Check for live tmux session
TMUX_LIVE=$(tmux list-sessions 2>/dev/null | grep "simp_run" || echo "")
if [[ -n "$TMUX_LIVE" ]]; then
    warn "tmux session 'simp_run' is already active — a run may still be in progress"
    info "Reattach:  tmux attach -t simp_run"
    info "Kill:      tmux kill-session -t simp_run"
    echo ""
fi

# Show result.json if present
if [[ -f "$RESULT_JSON" ]]; then
    CONVERGED=$(python3 -c "import json; r=json.load(open('$RESULT_JSON')); print(r.get('converged','?'))" 2>/dev/null || echo "?")
    N_ITER=$(python3 -c   "import json; r=json.load(open('$RESULT_JSON')); print(r.get('n_iterations','?'))" 2>/dev/null || echo "?")
    C_FINAL=$(python3 -c  "import json; r=json.load(open('$RESULT_JSON')); print(f\"{r.get('final_compliance',0):.5f}\")" 2>/dev/null || echo "?")
    DURATION=$(python3 -c "import json; r=json.load(open('$RESULT_JSON')); print(f\"{r.get('duration_s',0)/60:.0f} min\")" 2>/dev/null || echo "?")
    RESULT_AGE=$(( ( $(date +%s) - $(stat -c %Y "$RESULT_JSON") ) / 3600 ))

    if [[ "$CONVERGED" == "True" ]]; then
        pass "Last run: CONVERGED  |  iters=$N_ITER  C=$C_FINAL  wall=$DURATION  (${RESULT_AGE}h ago)"
    else
        warn "Last run: NOT converged  |  iters=$N_ITER  C=$C_FINAL  wall=$DURATION  (${RESULT_AGE}h ago)"
    fi
else
    info "No previous result.json found"
fi

# Checkpoint
if [[ -f "$CHECKPOINT_BIN" ]]; then
    CKPT_AGE=$(( ( $(date +%s) - $(stat -c %Y "$CHECKPOINT_BIN") ) / 3600 ))
    CKPT_SIZE=$(du -h "$CHECKPOINT_BIN" | cut -f1)
    info "Checkpoint present: density.bin  ($CKPT_SIZE, ${CKPT_AGE}h old)"
    info "If starting a fresh run, delete it first: rm outputs/problem/density.bin"
fi

# Show available parts
echo ""
info "Available parts (scad/*.json):"
for f in "$SCAD_DIR"/*_params.json; do
    PART=$(basename "$f" _params.json)
    if [[ -f "$REPO_ROOT/outputs/meshes/${PART}_stage04.json" ]]; then
        C_VAL=$(python3 -c "
import json
s = json.load(open('$REPO_ROOT/outputs/meshes/${PART}_stage04.json'))
print(f\"  C={s.get('final_compliance',0):.4f}  iters={s.get('n_iterations','?')}\")" 2>/dev/null || echo "")
        info "  ${PART}  ✓ run complete${C_VAL}"
    else
        info "  ${PART}"
    fi
done

# ─── Step 7: Prompt to run ────────────────────────────────────────────────────
step "7 / 7  Start a run?"

# Determine part
if [[ -n "$PART_OVERRIDE" ]]; then
    PART="$PART_OVERRIDE"
else
    echo -e "\n  Enter part name to run (or press Enter to skip):"
    echo -e "  ${DIM}(e.g. motor_mount, cantilever_arm, base_part)${X}"
    printf "  > "
    read -r PART
fi

if [[ -z "$PART" ]]; then
    echo ""
    info "No part selected — startup complete."
    info "To start a run manually:"
    info "  bash scripts/simp_dashboard.sh auto 6.0 0.35 200"
    echo ""
    exit 0
fi

# Validate part
PARAMS_FILE="$SCAD_DIR/${PART}_params.json"
if [[ ! -f "$PARAMS_FILE" ]]; then
    die "No params file found: $PARAMS_FILE"
fi
pass "Params file found: scad/${PART}_params.json"

# Load SIMP params from JSON
FILTER_R=$(python3 -c "
import json
p = json.load(open('$PARAMS_FILE'))
print(p.get('simp',{}).get('filter_radius', 4.5))
" 2>/dev/null || echo "4.5")
[[ -n "$FILTER_OVERRIDE"   ]] && FILTER_R="$FILTER_OVERRIDE"

VF=$(python3 -c "
import json
p = json.load(open('$PARAMS_FILE'))
print(p.get('simp',{}).get('volume_fraction', 0.35))
" 2>/dev/null || echo "0.35")
[[ -n "$VF_OVERRIDE"       ]] && VF="$VF_OVERRIDE"

MAX_ITER=$(python3 -c "
import json
p = json.load(open('$PARAMS_FILE'))
print(p.get('simp',{}).get('max_iterations', 200))
" 2>/dev/null || echo "200")
[[ -n "$MAX_ITER_OVERRIDE" ]] && MAX_ITER="$MAX_ITER_OVERRIDE"

BACKEND="auto"
[[ -n "$BACKEND_OVERRIDE"  ]] && BACKEND="$BACKEND_OVERRIDE"

echo ""
echo -e "  ${W}Run parameters (from ${PART}_params.json):${X}"
echo -e "  ${DIM}  part:          ${PART}${X}"
echo -e "  ${DIM}  filter_radius: ${FILTER_R} mm${X}"
echo -e "  ${DIM}  volume_frac:   ${VF}${X}"
echo -e "  ${DIM}  max_cg_iter:   ${MAX_ITER}${X}"
echo ""

# Run preflight DOF check
echo -e "  Running DOF preflight check..."
DOF_CHECK=$(python3 -c "
from src.geometry.param_schema import PipelineParams
from scripts.voxelize import build_load_case
import json

p = PipelineParams.from_json('$PARAMS_FILE')
raw = json.load(open('$PARAMS_FILE'))
simp = raw.get('simp', {})
vox  = simp.get('voxel_size_mm', 1.0)
geom = p.geometry

nx = int(round(geom.length / vox))
ny = int(round(geom.width  / vox))
nz = int(round(geom.height / vox))
grid = {'nx': nx, 'ny': ny, 'nz': nz, 'voxel_size': vox / 1000.0}

lc = build_load_case(
    p.geometry, p.load_hints, grid,
    load_case_config=p.load_case_config,
    attachment_regions=p.attachment_regions,
)
print(f'  load_dofs={len(lc[\"load_dofs\"])}  fixed_dofs={len(lc[\"fixed_dofs\"])}  grid={nx}x{ny}x{nz}')
" 2>/dev/null || echo "  PREFLIGHT_FAILED")

if echo "$DOF_CHECK" | grep -q "PREFLIGHT_FAILED"; then
    warn "DOF preflight check failed — review params before running"
else
    pass "DOF check: $DOF_CHECK"
fi

# Checkpoint warning
if [[ -f "$CHECKPOINT_BIN" ]]; then
    echo ""
    warn "Checkpoint file exists from a previous run."
    warn "The solver will RESUME from checkpoint, not start fresh."
    echo -e "\n  Delete checkpoint and start fresh? [y/N]"
    read -r -t 10 DEL_CKPT || DEL_CKPT="n"
    if [[ "${DEL_CKPT,,}" == "y" ]]; then
        rm -f "$CHECKPOINT_BIN" \
              "$REPO_ROOT/outputs/problem/result.json" \
              "$REPO_ROOT/outputs/problem/x_init.bin"
        pass "Checkpoint cleared — run will start from scratch"
    else
        info "Keeping checkpoint — run will resume from last saved iteration"
    fi
fi

# Final confirmation
echo ""
echo -e "  ${W}Ready to queue:${X}"
echo -e "  ${DIM}  bash scripts/simp_dashboard.sh auto ${FILTER_R} ${VF} ${MAX_ITER}${X}"
echo ""
echo -e "  Launch dashboard? [Y/n]"
printf "  > "
read -r -t 15 LAUNCH_ANS || LAUNCH_ANS="y"

if [[ "${LAUNCH_ANS,,}" == "n" ]]; then
    echo ""
    info "Run not started. Launch manually when ready:"
    info "  bash scripts/simp_dashboard.sh auto ${FILTER_R} ${VF} ${MAX_ITER}"
    echo ""
    exit 0
fi

echo ""
pass "Launching dashboard — detach with Ctrl+B, D"
info "Reattach any time: tmux attach -t simp_run"
echo ""

exec bash "$DASHBOARD_SCRIPT" auto "$FILTER_R" "$VF" "$MAX_ITER"