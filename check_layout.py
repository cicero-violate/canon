#!/usr/bin/env python3

import json
import sys

LAYOUT_FILE = "canon.semantic.layout.json"


def fail(msg):
    print(f"[ERROR] {msg}")
    sys.exit(1)


def normalize_node(node_dict):
    if not isinstance(node_dict, dict):
        return str(node_dict)

    if len(node_dict) != 1:
        raise ValueError(f"Unexpected node shape: {node_dict}")

    kind, value = next(iter(node_dict.items()))
    return f"{kind}:{value}"


def main():
    with open(LAYOUT_FILE) as f:
        data = json.load(f)

    modules = data.get("modules", [])
    routing = data.get("routing", [])

    print(f"Total modules: {len(modules)}")
    print(f"Total routing entries: {len(routing)}")

    module_ids = set()
    file_ids = set()

    # --- Validate modules + files ---
    for module in modules:
        mid = module["id"]
        if mid in module_ids:
            fail(f"Duplicate module id: {mid}")
        module_ids.add(mid)

        for file in module.get("files", []):
            fid = file["id"]
            if fid in file_ids:
                fail(f"Duplicate file id: {fid}")
            file_ids.add(fid)

    print("✔ Modules and files validated")

    # --- Validate routing ---
    routed_nodes = set()

    for entry in routing:
        node = normalize_node(entry["node"])
        file_id = entry["file_id"]

        if file_id not in file_ids:
            fail(f"Routing references unknown file_id: {file_id}")

        if node in routed_nodes:
            fail(f"Duplicate routing for node: {node}")

        routed_nodes.add(node)

    print("✔ Routing nodes are unique")
    print("✔ All routing file_ids are valid")

    print("Check complete.")


if __name__ == "__main__":
    main()
