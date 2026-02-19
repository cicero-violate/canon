#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::rename::core::project_editor::ProjectEditor;
use rename::rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    let renames = [
        (
            "crate::rustc_integration::project::CargoProject::build_dependencies",
            "ensure_dependencies_built",
        ),
        (
            "crate::rename::api::UpsertRequest::node_op",
            "push_node_op",
        ),
        (
            "crate::rename::macros::extract_derive_identifiers",
            "extract_derive_idents",
        ),
        (
            "crate::rename::macros::extract_proc_macro_identifiers",
            "extract_proc_macro_idents",
        ),
        (
            "crate::rename::macros::MacroInvocationAnalyzer::predict_generated_identifiers",
            "predict_generated_idents",
        ),
        (
            "crate::rename::macros::MacroHandlingReport::add_flagged",
            "add_flagged_reason",
        ),
        (
            "crate::rename::macros::MacroIdentifierCollector::process_macro_rules",
            "process_macro_rules_def",
        ),
        (
            "crate::rename::alias::graph::AliasGraph::resolve_chain",
            "resolve_alias_chain",
        ),
        (
            "crate::rename::structured::ast_render::render_function",
            "render_fn_item",
        ),
        (
            "crate::rename::structured::ast_render::render_impl",
            "render_impl_item",
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
