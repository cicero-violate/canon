#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::NodeOp;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // Move IncludeVisitor from crate::fs into crate::fs::include_visitor.
    // Tests: Gap 6 visibility promotion, Gap 7 super:: rewrite, impl co-movement.
    let symbol_id = "crate::fs::IncludeVisitor";
    let new_module = "crate::fs::include_visitor";

    let handle = editor
        .registry
        .handles
        .get(symbol_id)
        .cloned()
        .ok_or_else(|| format!("symbol not found: {symbol_id}"))?;

    println!("found handle: {:?}", handle);

    editor.queue(symbol_id, NodeOp::MoveSymbol {
        handle,
        new_module_path: new_module.to_string(),
        new_crate: None,
    })?;

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
