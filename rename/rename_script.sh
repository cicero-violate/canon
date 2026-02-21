cargo run --example list_symbols
cd /workspace/ai_sandbox/canon_workspace/rename/ 

# SYMBOLS.json uses "symbol" not "id" or "path"
# Dump struct + trait + type fully-qualified symbols
jq -r '
  .[]
  | select(.kind=="\"struct\"" or .kind=="\"trait\"" or .kind=="\"type\"")
  | "\(.kind)  \(.symbol)"
' SYMBOLS.json | sort

jq -r '
  map(.kind |= gsub("\""; ""))
  | group_by(.kind)
  | map({kind: .[0].kind, count: length})
  | sort_by(.count)
  | reverse[]
  | "\(.kind)\t\(.count)"
' SYMBOLS.json
