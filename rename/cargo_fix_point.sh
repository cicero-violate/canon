cd /workspace/ai_sandbox/canon_workspace/rename

rm -f ../.rename/state.tlog ../.rename/state.log 2>/dev/null
find .. -name "state.tlog" -o -name "state.log" -o -name "*.graph.log" | head -20

cargo run --example nodeop_movesymbol
