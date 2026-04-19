#!/usr/bin/env bash
# build_solver.sh — compile the Rust SIMP solver and copy binary to bin/
# Run this inside the Docker container:
#   docker-compose exec fenics-pipeline bash scripts/build_solver.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SOLVER_DIR="$REPO_ROOT/solver"
BIN_DIR="$REPO_ROOT/bin"

echo "Building SIMP solver (release)..."
cd "$SOLVER_DIR"
cargo build --release 2>&1

echo "Copying binary to bin/..."
mkdir -p "$BIN_DIR"
cp target/release/simp_solver "$BIN_DIR/simp_solver"
chmod +x "$BIN_DIR/simp_solver"

echo "Done: $BIN_DIR/simp_solver"
file "$BIN_DIR/simp_solver"