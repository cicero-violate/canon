cd /home/cicero-arch-omen/ai_sandbox/canon/canon
cargo build --bin canon 2>&1 | tail -3

# 1. Ingest a rust project's own source into an IR, and emit semantic and layout graphs
cargo run --bin canon -- ingest \
  --src /home/cicero-arch-omen/ai_sandbox/canon/canon \
  --semantic-out /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
  --layout-out /home/cicero-arch-omen/ai_sandbox/canon/canon.layout.json

# cargo run --bin canon -- ingest \
#   --src /home/cicero-arch-omen/ai_sandbox/canon/memory/database \
#   --semantic-out /home/cicero-arch-omen/ai_sandbox/canon/canon_store/database.ir.json \
#   --layout-out /home/cicero-arch-omen/ai_sandbox/canon/canon_store/database.layout.json



## INGEST AND DIAGNOSE
cargo run -p canon -- ingest \
  --src . \
  --semantic-out tests/data/generated_semantic.json \
  --layout-out tests/data/generated_layout.json \
  && cargo run -p canon -- diagnose tests/data/generated_semantic.json 2>&1

cargo run -p canon -- materialize \
  tests/data/generated_semantic.json \
  target/materialized_project \
  --layout tests/data/generated_layout.json


# 2. Bootstrap the capability graph and seed proposal
# cargo run --bin canon -- bootstrap-graph \
#   --ir /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
#   --graph-out /home/cicero-arch-omen/ai_sandbox/canon/canon.graph.json \
#   --proposal-out /home/cicero-arch-omen/ai_sandbox/canon/canon.proposal.json

/workspace/ai_sandbox/canon/target/debug/canon bootstrap-graph \
  --ir /workspace/ai_sandbox/canon_workspace/canon.ir.json \
  --graph-out /workspace/canon_workspace/canon.graph.json \
  --proposal-out /workspace/ai_sandbox/canon_workspace/canon.proposal.json

# 2. Start the chromium_messenger daemon (you do this manually)
# ./run_chromium_daemon.sh


# 3. Run the agent
cargo run -- run-agent \
  --ir /workspace/ai_sandbox/canon_workspace/canon.ir.json \
  --layout /workspace/ai_sandbox/canon_workspace/canon.semantic.layout.json \
  --graph /workspace/ai_sandbox/canon_workspace/canon.graph.json \
  --proposal /workspace/ai_sandbox/canon_workspace/canon.proposal.json \
  --ir-out /workspace/ai_sandbox/canon_workspace/canon.ir.out.json \
  --ledger-out /workspace/ai_sandbox/canon_workspace/canon.ledger.json \
  --graph-out /workspace/ai_sandbox/canon_workspace/canon.graph.out.json \
  --max-ticks 1
