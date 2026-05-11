#!/usr/bin/env bash
# scripts/setup_amgcl.sh
#
# Fetches AMGCL header-only C++ library into solver/vendor/amgcl/.
# Run once before the first `cargo build --release` that includes the
# `amgcl` feature (which is in the default feature set).
#
# Requirements (WSL2 / Ubuntu):
#   git  — apt-get install git
#   g++  — apt-get install build-essential   (already needed for Rust crates)
#
# No Boost required — build uses -DAMGCL_NO_BOOST.
# No CUDA required for the OpenMP CPU path.
#
# Usage:
#   cd /path/to/fenics-pipeline
#   bash scripts/setup_amgcl.sh
#
# Idempotent: safe to run again — only re-clones if headers are missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENDOR_DIR="$REPO_ROOT/solver/vendor"
AMGCL_DIR="$VENDOR_DIR/amgcl"
AMGCL_SENTINEL="$AMGCL_DIR/make_solver.hpp"

# ── Already present? ──────────────────────────────────────────────────────────
if [[ -f "$AMGCL_SENTINEL" ]]; then
    echo "[setup_amgcl] AMGCL headers already present at $AMGCL_DIR"
    echo "[setup_amgcl] Delete that directory and re-run to force update."
    exit 0
fi

# ── Clone (shallow, headers only) ────────────────────────────────────────────
AMGCL_REPO="https://github.com/ddemidov/amgcl.git"
TMP_DIR="$(mktemp -d)"
echo "[setup_amgcl] Cloning AMGCL (shallow) into $TMP_DIR ..."

git clone \
    --depth=1 \
    --branch master \
    --filter=blob:none \
    --sparse \
    "$AMGCL_REPO" \
    "$TMP_DIR/amgcl_repo"

# Sparse checkout: only the header tree (amgcl/) — skips tests, docs, examples
cd "$TMP_DIR/amgcl_repo"
git sparse-checkout set amgcl

echo "[setup_amgcl] Copying headers to $AMGCL_DIR ..."
mkdir -p "$VENDOR_DIR"
cp -r "$TMP_DIR/amgcl_repo/amgcl" "$AMGCL_DIR"

rm -rf "$TMP_DIR"

# ── Verify ────────────────────────────────────────────────────────────────────
if [[ ! -f "$AMGCL_SENTINEL" ]]; then
    echo "[setup_amgcl] ERROR: expected $AMGCL_SENTINEL — something went wrong."
    echo "[setup_amgcl] Files found under $AMGCL_DIR:"
    ls "$AMGCL_DIR" 2>/dev/null | head -20 || echo "  (directory missing)"
    exit 1
fi

echo "[setup_amgcl] Done.  $(find "$AMGCL_DIR" -name '*.hpp' | wc -l) headers installed."
echo "[setup_amgcl] You can now run: cd solver && cargo build --release"
