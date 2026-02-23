Great, glad it worked! Here is a full sweep of everything still worth renaming:

| Old Name                                   | New Name                            |
|--------------------------------------------+-------------------------------------|
| `check` (validate/check_proposals.rs)      | `check_proposals_top`               |
| `check` (validate/check_project.rs)        | `check_project`                     |
| `check` (validate/check_execution.rs)      | `check_execution_top`               |
| `run` (main.rs)                            | `run_application`                   |
| `slim_function`                            | `slim_function_for_slice`           |
| `truncate_list`                            | `truncate_ir_list`                  |
| `extract_field`                            | `extract_ir_field_value`            |
| `finalize_file`                            | `finalize_rendered_file`            |
| `hash_contents`                            | `hash_file_contents`                |
| `apply_preserve_regions`                   | `apply_preserve_region_splices`     |
| `splice_preserve_blocks`                   | `splice_preserved_content_blocks`   |
| `extract_preserve_blocks`                  | `extract_named_preserve_blocks`     |
| `split_lines_with_endings`                 | `split_content_lines_with_endings`  |
| `write_file_tree`                          | `write_file_tree_to_disk`           |
| `file_stem` (materialize/mod.rs)           | `extract_file_stem`                 |
| `file_stem` (materialize/render_module.rs) | `extract_module_file_stem`          |
| `world_model_bonus`                        | `compute_world_model_reward_bonus`  |
| `finalize_execution`                       | `finalize_tick_execution`           |
| `register_plan_and_execution`              | `persist_plan_and_execution_record` |
| `ensure_planning_judgment`                 | `ensure_tick_planning_judgment`     |
| `default_predicted_snapshot`               | `build_default_predicted_snapshot`  |
| `reconcile_prediction`                     | `reconcile_tick_prediction`         |
| `index_segment`                            | `index_manifest_segment`            |
| `encode` (storage/builder.rs)              | `encode_artifact`                   |
| `from_ir` (ArtifactManifest)               | `build_manifest_from_ir`            |
| `describe_event`                           | `describe_execution_event`          |
| `compute_patch_id`                         | `hash_patch_id`                     |
| `is_structured_patch`                      | `patch_has_structured_format`       |
| `entry_for` (diagnose/pipeline.rs)         | `pipeline_entry_for_field`          |
| `lookup` (diagnose/predicate.rs)           | `lookup_rule_predicate`             |
| `fallback_predicate`                       | `build_fallback_rule_predicate`     |
| `render_tick_executor_edges`               | `render_tick_graph_edge_summary`    |
| `compute_module_cycle`                     | `detect_module_cycle`               |
| `trace_cluster`                            | `trace_violation_cluster`           |
| `cluster_by_rule`                          | `group_violations_by_rule`          |
| `apply_mutation` (agent/meta.rs)           | `apply_graph_mutation`              |
| `propose_mutations`                        | `propose_graph_mutations`           |
| `find_disconnected`                        | `find_disconnected_graph_node`      |
| `visit` (agent/dispatcher.rs)              | `visit_dispatcher_node`             |
| `required_fields_for_stage`                | `required_llm_fields_for_stage`     |
| `require_stage`                            | `require_pipeline_stage_output`     |
| `extract_str_field`                        | `extract_payload_str_field`         |
| `spawn` (agent/ws_server.rs)               | `spawn_ws_bridge`                   |
| `accept_loop`                              | `run_ws_accept_loop`                |
| `handle_connection`                        | `handle_ws_connection`              |
| `handle_inbound`                           | `handle_ws_inbound_message`         |
| `is_done` (agent/sse.rs)                   | `sse_stream_is_done`                |
| `build_file_lookup` (layout/validation.rs) | `build_layout_file_lookup`          |
| `build_module_maps`                        | `build_layout_module_maps`          |
| `normalize_layout` (layout/mod.rs)         | `normalize_layout_graph_in_place`   |
| `layout_node_key`                          | `layout_node_type_key`              |
| `default_file_id` (layout/strategies.rs)   | `default_module_file_id`            |
| `sanitize_module`                          | `sanitize_module_id_segment`        |
| `route_all_nodes`                          | `route_all_semantic_nodes`          |
| `ensure_named_file`                        | `ensure_layout_named_file`          |
| `normalize_layout_graph`                   | `normalize_and_dedupe_layout_graph` |
| `slugify` (dot_import.rs)                  | `slugify_dot_label`                 |
| `file_stem` (dot_import.rs)                | `dot_label_file_stem`               |
| `cluster_label`                            | `find_cluster_label`                |
| `cluster_label_to_word`                    | `parse_cluster_label_as_word`       |
| `edge_color`                               | `dot_edge_color_for_module`         |
| `lib_node`                                 | `dot_lib_node_line`                 |
| `entry_node`                               | `dot_entry_node_line`               |
| `cluster_id_of`                            | `dot_cluster_id_for_module`         |
| `sanitize_node_id`                         | `dot_sanitize_node_id`              |
| `slugify` (dot_export.rs)                  | `slugify_dot_id`                    |
| `initial_state_seed`                       | `compute_initial_state_seed`        |
| `payload_hash`                             | `compute_delta_payload_hash`        |
| `kernel_delta`                             | `delta_to_kernel_delta`             |
| `map_proof_scope`                          | `map_ir_proof_scope_to_kernel`      |
