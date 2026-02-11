#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
canon_root="$(cd "$script_dir/.." && pwd)"
default_bin="$canon_root/../target/debug/canon"
canon_bin="${CANON_BIN:-$default_bin}"
dsl_file="$script_dir/linter.canon"
input_ir="${1:-$canon_root/tests/data/valid_ir.json}"
output_ir="${2:-$script_dir/evolved.json}"
materialize_dir="${3:-$script_dir/out}"

if [[ ! -x "$canon_bin" ]]; then
  echo "Building canon CLI..."
  (cd "$canon_root" && cargo build -p canon --target-dir ../target >/dev/null)
fi

"$canon_bin" submit-dsl "$dsl_file" \
  --ir "$input_ir" \
  --output-ir "$output_ir" \
  --materialize-dir "$materialize_dir"
