#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# materialize sample IR into target/materialized_project
cargo run -- materialize \
  tests/data/valid_ir.json \
  target/materialized_project

# convert execution events into observe deltas
cargo run -- observe-events \
  tests/data/execution_record.json \
  proof.delta \
  target/observe_deltas.json

ls -R target/materialized_project | head -20
echo "Observe deltas:"
cat target/observe_deltas.json
