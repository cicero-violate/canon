#!/usr/bin/env bash
set -euo pipefail

BIN_NAME="rtree"
WORKSPACE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$WORKSPACE_ROOT/target/debug/$BIN_NAME"
DST="$HOME/.local/bin/$BIN_NAME"

cargo build -p rtree

cp -f "$SRC" "$DST"
chmod +x "$DST"

echo "✓ installed $DST"

rtree --dot . > dep.dot
