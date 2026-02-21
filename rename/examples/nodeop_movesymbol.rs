#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::NodeOp;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // Test: move NullOracle from crate::core::oracle into crate::core::project_editor
    // NullOracle is only used inside project_editor — this is a cohesion improvement.
    let symbol_id = "crate::core::oracle::NullOracle";
    let new_module = "crate::core::project_editor";

    let handle = editor
        .registry
        .handles
        .get(symbol_id)
        .cloned()
        .ok_or_else(|| format!("symbol not found: {symbol_id}"))?;

    println!("found handle: {:?}", handle);

    // Debug: print all handles containing "oracle" to find the impl block ID
    for id in editor.debug_list_symbol_ids() {
        if id.contains("oracle") || id.contains("NullOracle") {
            println!("handle: {id}");
        }
    }

    editor.queue(symbol_id, NodeOp::MoveSymbol {
        handle,
        new_module_path: new_module.to_string(),
        new_crate: None,
    })?;

    // Debug: print all handles containing "oracle" to find the impl block ID
    for id in editor.debug_list_symbol_ids() {
        if id.contains("oracle") || id.contains("NullOracle") {
            println!("handle: {id}");
        }
    }

    // Also move the impl block — it's a separate item that must travel with the struct
    let impl_id = "crate::core::oracle::NullOracle as crate::core::oracle::StructuralEditOracle";
    if let Some(impl_handle) = editor.registry.handles.get(impl_id).cloned() {
        println!("found impl handle: {:?}", impl_handle);
        editor.queue(impl_id, NodeOp::MoveSymbol {
            handle: impl_handle,
            new_module_path: new_module.to_string(),
            new_crate: None,
        })?;
    } else {
        println!("impl handle not found, skipping");
    }
    // The impl block has no direct handle — derive it from a method handle.
    // All methods share the same item_index which points to the impl block.
    let method_id = "crate::core::oracle::NullOracle as crate::core::oracle::StructuralEditOracle::impact_of";
    if let Some(method_handle) = editor.registry.handles.get(method_id).cloned() {
        // Build an impl-level handle: same file + item_index, no nested_path
        let impl_handle = rename::structured::node_handle(
            method_handle.file.clone(),
            method_handle.item_index,
            vec![],
            rename::state::NodeKind::Impl,
        );
        println!("derived impl handle: {:?}", impl_handle);
        editor.queue("crate::core::oracle::NullOracle::impl_StructuralEditOracle", NodeOp::MoveSymbol {
            handle: impl_handle,
            new_module_path: new_module.to_string(),
            new_crate: None,
        })?;
    }

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched:    {:?}", report.touched_files);
    println!("file_moves: {:?}", report.file_moves);

    // Dry-run by default — pass --commit to write to disk
    if std::env::args().any(|a| a == "--commit") {
        let written = editor.commit()?;
        println!("written: {:?}", written);
    } else {
        let preview = editor.preview()?;
        println!("preview:\n{preview}");
        println!("\n(dry-run — pass --commit to apply)");
    }

    Ok(())
}
