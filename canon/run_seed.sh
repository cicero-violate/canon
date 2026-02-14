# 1. Bootstrap the graph and seed proposal
canon bootstrap-graph --ir canon.ir.json \
  --graph-out graph.json --proposal-out proposal.json

# 2. Start the chromium_messenger daemon (you do this manually)
./run_chromium_daemon.sh

# 3. Run the agent loop
canon run-agent --ir canon.ir.json \
  --graph graph.json --proposal proposal.json \
  --ir-out out.ir.json --ledger-out ledger.json \
  --graph-out graph.json --max-ticks 10
