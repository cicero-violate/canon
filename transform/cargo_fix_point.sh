cd /workspace/ai_sandbox/canon_workspace/rename

rm -f .rename/* 2>/dev/null
rm -rf fix_point/src/*
find .. -name "state.tlog" -o -name "state.log" -o -name "*.graph.log" | head -20

# cargo run --example nodeop_movesymbol
RENAME_DEBUG_PLAN=1 cargo run --example nodeop_movesymbol 2>&1 \
  | grep "AliasGraph\|NodeId\|dedup\|top-level" \
  | head -40
