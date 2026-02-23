cargo run --example list_symbols
cd /workspace/ai_sandbox/canon_workspace/rename/ 

# SYMBOLS.json uses "symbol" not "id" or "path"
# Dump struct + trait + type fully-qualified symbols
jq -r '
  map(.kind |= gsub("\""; "")) 
  | map(
      if .kind == "function" or .kind == "method"
      then .kind = "fn"
      else .
      end
    )
  | map(select(.kind | IN("struct","trait","type","fn")))
  | group_by(.kind)
  | sort_by(.[0].kind)
  | .[]
  | (.[0].kind | ascii_upcase),
    (map("  \(.symbol)") | sort[]),
    ""
' SYMBOLS.json

jq -r '
  map(.kind |= gsub("\""; ""))
  | group_by(.kind)
  | map({kind: .[0].kind, count: length})
  | sort_by(.count)
  | reverse[]
  | "\(.kind)\t\(.count)"
' SYMBOLS.json
