cargo run --example list_symbols
cd /workspace/ai_sandbox/canon_workspace/rename/ 

# SYMBOLS.json uses "symbol" not "id" or "path"
# Dump struct + trait + type fully-qualified symbols
jq -r '
  .[]
  | select(.kind=="\"struct\"" or .kind=="\"trait\"" or .kind=="\"type\"")
  | "\(.kind)  \(.symbol)"
' SYMBOLS.json | sort
