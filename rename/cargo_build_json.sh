cargo build --message-format=json 2>/dev/null \
| jq -c 'select(.reason=="compiler-message")
         | .message
         # | select(.level=="error" or .level=="warning")
         | select(.level=="error")
         | {level: .level,
            code: (.code.code // "unknown"),
            message: .message,
            spans: [.spans[]? | {file: .file_name, line: .line_start, col: .column_start}]}' \
> build_diagnostics.json

jq -r '.level + ":" + .code' build_diagnostics.json \
| sort | uniq -c | sort -k2

bat -n build_diagnostics.json

