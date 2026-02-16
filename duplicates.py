#!/usr/bin/env python3
import json
import sys
from collections import Counter

ir = json.load(open(sys.argv[1]))

def check(kind):
    ids = [x["id"] for x in ir.get(kind, [])]
    counts = Counter(ids)
    dups = [k for k, v in counts.items() if v > 1]
    if dups:
        print(f"\n{kind.upper()} DUPLICATES:")
        for d in dups[:10]:
            print(" ", d, "x", counts[d])
    else:
        print(f"\n{kind.upper()}: no duplicates")

for kind in [
    "modules",
    "structs",
    "traits",
    "impls",
    "functions",
    "deltas",
    "proofs",
    "judgments",
    "proposals",
]:
    check(kind)
