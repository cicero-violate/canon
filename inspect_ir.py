#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


def infer_schema(obj, depth=0):
    if isinstance(obj, dict):
        return {
            "type": "object",
            "fields": {k: infer_schema(v, depth + 1) for k, v in obj.items()}
        }
    elif isinstance(obj, list):
        if not obj:
            return {"type": "array", "items": "unknown"}
        return {
            "type": "array",
            "items": infer_schema(obj[0], depth + 1)
        }
    else:
        return {"type": type(obj).__name__}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main(semantic_path, layout_path):
    semantic = load_json(semantic_path)
    layout = load_json(layout_path)

    print_section("Top-Level Schema (semantic.json)")
    print(json.dumps(infer_schema(semantic), indent=2))

    print_section("Top-Level Schema (layout.json)")
    print(json.dumps(infer_schema(layout), indent=2))

    impl_blocks = semantic.get("impls", [])
    functions = semantic.get("functions", [])

    print_section("Impl Block Schema")
    if impl_blocks:
        print(json.dumps(infer_schema(impl_blocks[0]), indent=2))
    else:
        print("No impl_blocks found.")

    print_section("Function Schema")
    if functions:
        print(json.dumps(infer_schema(functions[0]), indent=2))
    else:
        print("No functions found.")

    # ===============================
    # ID Mismatch Analysis
    # ===============================

    impl_ids = {block["id"] for block in impl_blocks}
    function_impl_ids = {
        f["impl_id"]
        for f in functions
        if f.get("impl_id") and f["impl_id"] != ""
    }

    missing_impls = function_impl_ids - impl_ids

    print_section("Impl ID Mismatch Report")
    print(f"Total impl_blocks: {len(impl_ids)}")
    print(f"Functions referencing impl_id: {len(function_impl_ids)}")
    print(f"Unresolved impl_id count: {len(missing_impls)}")

    if missing_impls:
        print("\nSample unresolved impl_ids:")
        for x in list(missing_impls)[:10]:
            print("  ", x)

    # ===============================
    # Trait Resolution Analysis
    # ===============================

    trait_ids = {t["id"] for t in semantic.get("traits", [])}

    broken_traits = [
        block for block in impl_blocks
        if block.get("trait_id") and block["trait_id"] not in trait_ids
    ]

    print_section("Trait Resolution Report")
    print(f"Total traits: {len(trait_ids)}")
    print(f"Impl blocks with trait_id: {len([b for b in impl_blocks if b.get('trait_id')])}")
    print(f"Broken trait references: {len(broken_traits)}")

    for b in broken_traits[:10]:
        print("  Impl:", b["id"])
        print("    trait_id:", b.get("trait_id"))

    print_section("DONE")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inspect_ir.py <semantic.json> <layout.json>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
