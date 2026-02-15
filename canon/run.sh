#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cargo run -p canon -- ingest \
  --src canon \
  --semantic-out canon/tests/data/generated_semantic.json \
  --layout-out canon/tests/data/generated_layout.json

cargo run -p canon -- materialize \
  canon/tests/data/generated_semantic.json \
  target/materialized_project \
  --layout canon/tests/data/generated_layout.json

# convert execution events into observe deltas
cargo run -- observe-events \
  tests/data/execution_record.json \
  proof.delta \
  target/observe_deltas.json

ls -R target/materialized_project | head -20
echo "Observe deltas:"
cat target/observe_deltas.json
