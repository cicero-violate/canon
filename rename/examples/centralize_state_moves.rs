#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::NodeOp;
use std::path::Path;

/// Centralize all `state::*` modules under `crate::state`
/// using MoveSymbol operations (file-backed modules).
///
/// This builds on nodeop_movesymbol.rs but batches moves
/// to enforce a single state root.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path =
        Path::new("/workspace/ai_sandbox/canon_workspace/rename/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    let targets = [
        "crate::state::builder",
        "crate::state::graph",
        "crate::state::ids",
        "crate::state::node",
        "crate::state::workspace",
        "crate::state::capability",
    ];

    for symbol_id in targets {
        let new_module = "crate::state";

        let handle = editor
            .registry
            .handles
            .get(symbol_id)
            .cloned()
            .ok_or_else(|| format!("symbol not found: {symbol_id}"))?;

        println!("centralizing: {symbol_id}");

        editor.queue(
            symbol_id,
            NodeOp::MoveSymbol {
                handle,
                new_module_path: new_module.to_string(),
                new_crate: None,
            },
        )?;
    }

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched:    {:?}", report.touched_files);
    println!("file_moves: {:?}", report.file_moves);

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

