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
            "crate::agent::observe::observe_ir",
            "analyze_ir",
        ),
        (
            "crate::agent::observe::observation_to_payload",
            "ir_observation_to_json",
        ),
        (
            "crate::agent::observe::IrObservation",
            "IrAnalysisReport",
        ),
        (
            "crate::agent::pipeline::run_pipeline",
            "run_refactor_pipeline",
        ),
        (
            "crate::agent::pipeline::record_pipeline_outcome",
            "record_refactor_reward",
        ),
        (
            "crate::agent::pipeline::PipelineResult",
            "RefactorResult",
        ),
        (
            "crate::agent::pipeline::PipelineError",
            "RefactorError",
        ),
        (
            "crate::agent::meta::run_meta_tick",
            "evolve_capability_graph",
        ),
        (
            "crate::agent::meta::MetaTickResult",
            "GraphEvolutionResult",
        ),
        (
            "crate::agent::meta::MetaTickError",
            "GraphEvolutionError",
        ),
        (
            "crate::agent::runner::persist_ir",
            "write_ir_to_disk",
        ),
        (
            "crate::agent::runner::persist_ledger",
            "write_ledger_to_disk",
        ),
        (
            "crate::storage::builder::MemoryIrBuilder::persist_segment",
            "write_artifact_page",
        ),
        (
            "crate::storage::builder::MemoryIrBuilder::synthetic_judgment",
            "build_builder_judgment_proof",
        ),
        (
            "crate::storage::builder::MemoryIrBuilder::commit_deltas",
            "admit_and_commit_pages",
        ),
        (
            "crate::storage::reader::MemoryIrReader::from_checkpoint",
            "read_ir_from_checkpoint",
        ),
        (
            "crate::io_utils::load_ir_any",
            "load_ir_from_path",
        ),
        (
            "crate::storage::builder::ManifestSlotLookup::slot_or_err",
            "require_slot",
        ),
        (
            "crate::storage::manifest::assign_slots",
            "assign_manifest_slots",
        ),
        (
            "crate::agent::slice::build_ir_slice",
            "slice_ir_fields",
        ),
        (
            "crate::observe::execution_events_to_observe_deltas",
            "execution_events_to_deltas",
        ),
        (
            "crate::evolution::lyapunov::check_topology_drift",
            "enforce_lyapunov_bound",
        ),
        (
            "crate::agent::bootstrap::bootstrap_proposal",
            "seed_refactor_proposal",
        ),
        (
            "crate::agent::bootstrap::bootstrap_graph",
            "seed_capability_graph",
        ),
        (
            "crate::decision::auto_dsl::auto_accept_dsl_proposal",
            "apply_dsl_proposal",
        ),
        (
            "crate::agent::reward::NodeOutcome",
            "PipelineNodeOutcome",
        ),
        (
            "crate::agent::reward::RewardLedger",
            "NodeRewardLedger",
        ),
        (
            "crate::agent::dispatcher::AgentCallDispatcher",
            "CapabilityNodeDispatcher",
        ),
        (
            "crate::agent::dispatcher::AgentCallDispatcher::dispatch_order",
            "topological_call_order",
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
