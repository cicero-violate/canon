cd /workspace/ai_sandbox/canon_workspace/rename

rm -f ../.rename/* 2>/dev/null
rm -f fix_point/src/*
find .. -name "state.tlog" -o -name "state.log" -o -name "*.graph.log" | head -20

cargo run --example nodeop_movesymbol
