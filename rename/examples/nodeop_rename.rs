#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::rename::core::{NullOracle, project_editor::ProjectEditor};
use rename::rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/canon");
    let mut editor = ProjectEditor::load(project_path, Box::new(NullOracle))?;

    let renames = [
        (
            "crate::validate::check_proposals::check",
            "check_proposals_top",
        ),
        (
            "crate::validate::check_project::check",
            "check_project",
        ),
        (
            "crate::validate::check_execution::check",
            "check_execution_top",
        ),
        (
            "crate::run",
            "run_application",
        ),
        (
            "crate::agent::slice::slim_function",
            "slim_function_for_slice",
        ),
        (
            "crate::agent::slice::truncate_list",
            "truncate_ir_list",
        ),
        (
            "crate::agent::slice::extract_field",
            "extract_ir_field_value",
        ),
        (
            "crate::materialize::finalize_file",
            "finalize_rendered_file",
        ),
        (
            "crate::materialize::hash_contents",
            "hash_file_contents",
        ),
        (
            "crate::materialize::apply_preserve_regions",
            "apply_preserve_region_splices",
        ),
        (
            "crate::materialize::splice_preserve_blocks",
            "splice_preserved_content_blocks",
        ),
        (
            "crate::materialize::extract_preserve_blocks",
            "extract_named_preserve_blocks",
        ),
        (
            "crate::materialize::split_lines_with_endings",
            "split_content_lines_with_endings",
        ),
        (
            "crate::materialize::write_file_tree",
            "write_file_tree_to_disk",
        ),
        (
            "crate::materialize::file_stem",
            "extract_file_stem",
        ),
        (
            "crate::materialize::render_module::file_stem",
            "extract_module_file_stem",
        ),
        (
            "crate::runtime::tick_executor::planning::world_model_bonus",
            "compute_world_model_reward_bonus",
        ),
        (
            "crate::runtime::tick_executor::planning::finalize_execution",
            "finalize_tick_execution",
        ),
        (
            "crate::runtime::tick_executor::planning::register_plan_and_execution",
            "persist_plan_and_execution_record",
        ),
        (
            "crate::runtime::tick_executor::planning::ensure_planning_judgment",
            "ensure_tick_planning_judgment",
        ),
        (
            "crate::runtime::tick_executor::types::default_predicted_snapshot",
            "build_default_predicted_snapshot",
        ),
        (
            "crate::runtime::tick_executor::reconcile::reconcile_prediction",
            "reconcile_tick_prediction",
        ),
        (
            "crate::storage::builder::ManifestSlotLookup::index_segment",
            "index_manifest_segment",
        ),
        (
            "crate::storage::builder::MemoryIrBuilder::encode",
            "encode_artifact",
        ),
        (
            "crate::storage::manifest::ArtifactManifest::from_ir",
            "build_manifest_from_ir",
        ),
        (
            "crate::observe::describe_event",
            "describe_execution_event",
        ),
        (
            "crate::patch_protocol::compute_patch_id",
            "hash_patch_id",
        ),
        (
            "crate::patch_protocol::is_structured_patch",
            "patch_has_structured_format",
        ),
        (
            "crate::diagnose::pipeline::entry_for",
            "pipeline_entry_for_field",
        ),
        (
            "crate::diagnose::predicate::lookup",
            "lookup_rule_predicate",
        ),
        (
            "crate::diagnose::fallback_predicate",
            "build_fallback_rule_predicate",
        ),
        (
            "crate::diagnose::render_tick_executor_edges",
            "render_tick_graph_edge_summary",
        ),
        (
            "crate::diagnose::compute_module_cycle",
            "detect_module_cycle",
        ),
        (
            "crate::diagnose::trace_cluster",
            "trace_violation_cluster",
        ),
        (
            "crate::diagnose::cluster_by_rule",
            "group_violations_by_rule",
        ),
        (
            "crate::agent::meta::apply_mutation",
            "apply_graph_mutation",
        ),
        (
            "crate::agent::meta::propose_mutations",
            "propose_graph_mutations",
        ),
        (
            "crate::agent::meta::find_disconnected",
            "find_disconnected_graph_node",
        ),
        (
            "crate::agent::dispatcher::CapabilityNodeDispatcher::visit",
            "visit_dispatcher_node",
        ),
        (
            "crate::agent::llm_provider::required_fields_for_stage",
            "required_llm_fields_for_stage",
        ),
        (
            "crate::agent::pipeline::require_stage",
            "require_pipeline_stage_output",
        ),
        (
            "crate::agent::pipeline::extract_str_field",
            "extract_payload_str_field",
        ),
        (
            "crate::agent::ws_server::spawn",
            "spawn_ws_bridge",
        ),
        (
            "crate::agent::ws_server::accept_loop",
            "run_ws_accept_loop",
        ),
        (
            "crate::agent::ws_server::handle_connection",
            "handle_ws_connection",
        ),
        (
            "crate::agent::ws_server::handle_inbound",
            "handle_ws_inbound_message",
        ),
        (
            "crate::agent::sse::is_done",
            "sse_stream_is_done",
        ),
        (
            "crate::layout::validation::build_file_lookup",
            "build_layout_file_lookup",
        ),
        (
            "crate::layout::validation::build_module_maps",
            "build_layout_module_maps",
        ),
        (
            "crate::layout::normalize_layout",
            "normalize_layout_graph_in_place",
        ),
        (
            "crate::layout::layout_node_key",
            "layout_node_type_key",
        ),
        (
            "crate::layout::strategies::default_file_id",
            "default_module_file_id",
        ),
        (
            "crate::layout::strategies::sanitize_module",
            "sanitize_module_id_segment",
        ),
        (
            "crate::layout::strategies::route_all_nodes",
            "route_all_semantic_nodes",
        ),
        (
            "crate::layout::strategies::ensure_named_file",
            "ensure_layout_named_file",
        ),
        (
            "crate::layout::strategies::normalize_layout_graph",
            "normalize_and_dedupe_layout_graph",
        ),
        (
            "crate::dot_import::slugify",
            "slugify_dot_label",
        ),
        (
            "crate::dot_import::file_stem",
            "dot_label_file_stem",
        ),
        (
            "crate::dot_import::cluster_label",
            "find_cluster_label",
        ),
        (
            "crate::dot_import::cluster_label_to_word",
            "parse_cluster_label_as_word",
        ),
        (
            "crate::dot_export::edge_color",
            "dot_edge_color_for_module",
        ),
        (
            "crate::dot_export::lib_node",
            "dot_lib_node_line",
        ),
        (
            "crate::dot_export::entry_node",
            "dot_entry_node_line",
        ),
        (
            "crate::dot_export::cluster_id_of",
            "dot_cluster_id_for_module",
        ),
        (
            "crate::dot_export::sanitize_node_id",
            "dot_sanitize_node_id",
        ),
        (
            "crate::dot_export::slugify",
            "slugify_dot_id",
        ),
        (
            "crate::evolution::kernel_bridge::initial_state_seed",
            "compute_initial_state_seed",
        ),
        (
            "crate::evolution::kernel_bridge::payload_hash",
            "compute_delta_payload_hash",
        ),
        (
            "crate::evolution::kernel_bridge::kernel_delta",
            "delta_to_kernel_delta",
        ),
        (
            "crate::evolution::kernel_bridge::map_proof_scope",
            "map_ir_proof_scope_to_kernel",
        ),
    ];


    for (symbol_id, new_name) in renames {
        editor.queue_by_id(symbol_id, FieldMutation::RenameIdent(new_name.to_string()))?;
    }

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched: {:?}", report.touched_files);

    let preview = editor.preview()?;
    println!("preview: {preview}");

    let written = editor.commit()?;
    println!("written: {:?}", written);

    Ok(())
}
