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
            "crate::storage::manifest::assign_manifest_slots",
            "build_manifest_entries",
        ),
        (
            "crate::io_utils::load_ir",
            "load_ir_from_file",
        ),
        (
            "crate::io_utils::load_ir_or_semantic",
            "load_ir_or_semantic_graph",
        ),
        (
            "crate::evolution::apply_deltas",
            "apply_admitted_deltas",
        ),
        (
            "crate::evolution::enforce_delta_application",
            "assert_delta_is_applicable",
        ),
        (
            "crate::runtime::reward::compute_reward",
            "compute_execution_reward",
        ),
        (
            "crate::validate::check_artifacts::check",
            "check_artifacts",
        ),
        (
            "crate::validate::check_graphs::check",
            "check_graphs",
        ),
        (
            "crate::validate::check_deltas::check",
            "check_deltas_top",
        ),
        (
            "crate::observe::execution_events_to_deltas",
            "wrap_execution_events_as_deltas",
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
