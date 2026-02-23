# Diagnose build_diagnostics.json
# Goal: For each error code, show up to 5 example messages with file:line

import json
from collections import defaultdict

# Load diagnostics
with open("build_diagnostics.json", "r") as f:
    diagnostics = [json.loads(line) for line in f if line.strip()]

# Group by error code (errors only)
errors = defaultdict(list)

for d in diagnostics:
    if d.get("level") == "error":
        code = d.get("code", "unknown")
        errors[code].append(d)

# Print summary and up to 5 examples per error code
for code, entries in sorted(errors.items()):
    print(f"\n=== {code} (total: {len(entries)}) ===")
    for e in entries[:5]:
        message = e.get("message", "")
        spans = e.get("spans", [])
        if spans:
            span = spans[0]
            file = span.get("file", "?")
            line = span.get("line", "?")
            print(f"- {file}:{line} -> {message}")
        else:
            print(f"- {message}")
