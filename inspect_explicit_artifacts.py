#!/usr/bin/env python3

import json
import sys
from collections import defaultdict


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def index_by_id(items):
    return {item["id"]: item for item in items}


def main(path):
    ir = load(path)

    modules = index_by_id(ir.get("modules", []))
    structs = index_by_id(ir.get("structs", []))
    enums = index_by_id(ir.get("enums", []))
    traits = index_by_id(ir.get("traits", []))
    impls = index_by_id(ir.get("impls", []))
    deltas = index_by_id(ir.get("deltas", [])) if "deltas" in ir else {}

    unresolved = defaultdict(list)

    # ---- Function checks ----
    for f in ir.get("functions", []):
        # impl_id
        if f.get("impl_id"):
            if f["impl_id"] not in impls:
                unresolved["function.impl_id"].append(f["impl_id"])

        # trait_function
        if f.get("trait_function"):
            trait_fn_ids = {
                tf["id"]
                for t in traits.values()
                for tf in t.get("functions", [])
            }
            if f["trait_function"] not in trait_fn_ids:
                unresolved["function.trait_function"].append(f["trait_function"])

        # deltas
        for d in f.get("deltas", []):
            delta_id = d.get("delta")
            if delta_id and delta_id not in deltas:
                unresolved["function.delta"].append(delta_id)

    # ---- Impl checks ----
    for block in ir.get("impls", []):
        if block.get("struct_id"):
            if (
                block["struct_id"] not in structs
                and block["struct_id"] not in enums
            ):
                unresolved["impl.struct_id"].append(block["struct_id"])

        if block.get("trait_id"):
            if block["trait_id"] not in traits:
                unresolved["impl.trait_id"].append(block["trait_id"])

        for binding in block.get("functions", []):
            if binding.get("function"):
                if binding["function"] not in {
                    f["id"] for f in ir.get("functions", [])
                }:
                    unresolved["impl.binding.function"].append(
                        binding["function"]
                    )

    # ---- Report ----
    print("\n" + "=" * 70)
    print("ExplicitArtifacts Resolution Report")
    print("=" * 70)

    total = 0
    for field, values in unresolved.items():
        unique_vals = sorted(set(values))
        count = len(values)
        total += count
        print(f"\nField: {field}")
        print(f"  Count: {count}")
        print("  Sample:")
        for v in unique_vals[:5]:
            print("   -", v)

    if total == 0:
        print("\nNo unresolved artifact references detected.")
    else:
        print(f"\nTOTAL unresolved references: {total}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_explicit_artifacts.py canon.semantic.json")
        sys.exit(1)

    main(sys.argv[1])
