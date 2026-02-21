cargo build --message-format=json 2>/dev/null \
| jq -c 'select(.reason=="compiler-message")
         | .message
         | select(.level=="errors")
         | {code: (.code.code // "unknown"),
            message: .message,
            spans: [.spans[]? | {file: .file_name, line: .line_start, col: .column_start}]}' \
> build_errors.json

jq -r '.code' build_errors.json | sort | uniq -c | sort -k2
bat -n build_errors.json 
