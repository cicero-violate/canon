cd /home/cicero-arch-omen/ai_sandbox/canon/canon
cargo build --bin canon 2>&1 | tail -3

# 1. Ingest a rust project's own source into an IR, and emit semantic and layout graphs
cargo run --bin canon -- ingest \
  --src /home/cicero-arch-omen/ai_sandbox/canon/canon \
  --semantic-out /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
  --layout-out /home/cicero-arch-omen/ai_sandbox/canon/canon.layout.json

# cargo run --bin canon -- ingest \
#   --src /home/cicero-arch-omen/ai_sandbox/canon/memory/memory_engine \
#   --semantic-out /home/cicero-arch-omen/ai_sandbox/canon/canon_store/memory_engine.ir.json \
#   --layout-out /home/cicero-arch-omen/ai_sandbox/canon/canon_store/memory_engine.layout.json

cargo run -p canon -- materialize \
  /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
  target/materialized_project \
  --layout /home/cicero-arch-omen/ai_sandbox/canon/canon.layout.json

# 2. Bootstrap the capability graph and seed proposal
# cargo run --bin canon -- bootstrap-graph \
#   --ir /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
#   --graph-out /home/cicero-arch-omen/ai_sandbox/canon/canon.graph.json \
#   --proposal-out /home/cicero-arch-omen/ai_sandbox/canon/canon.proposal.json

/workspace/ai_sandbox/canon/target/debug/canon bootstrap-graph \
  --ir /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
  --graph-out /home/cicero-arch-omen/ai_sandbox/canon/canon.graph.json \
  --proposal-out /home/cicero-arch-omen/ai_sandbox/canon/canon.proposal.json

# 2. Start the chromium_messenger daemon (you do this manually)
# ./run_chromium_daemon.sh


# 3. Run the agent
cargo run -- run-agent \
  --ir /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.json \
  --layout /home/cicero-arch-omen/ai_sandbox/canon/canon.layout.json \
  --graph /home/cicero-arch-omen/ai_sandbox/canon/canon.graph.json \
  --proposal /home/cicero-arch-omen/ai_sandbox/canon/canon.proposal.json \
  --ir-out /home/cicero-arch-omen/ai_sandbox/canon/canon.ir.out.json \
  --ledger-out /home/cicero-arch-omen/ai_sandbox/canon/canon.ledger.json \
  --graph-out /home/cicero-arch-omen/ai_sandbox/canon/canon.graph.out.json \
  --max-ticks 1
