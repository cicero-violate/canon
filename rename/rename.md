Looking at the current canon symbol map (document 3), here are 10 that still need renaming:

| Old Name                                  | New Name                          |
|-------------------------------------------+-----------------------------------|
| `assign_manifest_slots`                   | `build_manifest_entries`          |
| `load_ir`                                 | `load_ir_from_file`               |
| `load_ir_or_semantic`                     | `load_ir_or_semantic_graph`       |
| `apply_deltas`                            | `apply_admitted_deltas`           |
| `enforce_delta_application`               | `assert_delta_is_applicable`      |
| `compute_reward`                          | `compute_execution_reward`        |
| `check` (validate/check_artifacts/mod.rs) | `check_artifacts`                 |
| `check` (validate/check_graphs.rs)        | `check_graphs`                    |
| `check` (validate/check_deltas.rs)        | `check_deltas_top`                |
| `execution_events_to_deltas`              | `wrap_execution_events_as_deltas` |

The `check` naming is the most urgent — three separate files all export a function called `check` with identical signatures. When anything imports or references them by name in a flat context the names are completely ambiguous. The `load_ir` vs `load_ir_from_file` vs `load_ir_from_path` cluster is the second biggest problem since all three do slightly different things but their names suggest they are the same operation.
