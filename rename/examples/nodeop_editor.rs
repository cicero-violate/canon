#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use anyhow::Result;
use rename::rename::core::ProjectEditor;
use rename::rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<()> {
    let project_root = Path::new("/workspace/ai_sandbox/canon_workspace/rename");

    let mut editor = ProjectEditor::load_with_rustc(project_root)?;

    let symbol_id = "crate::rename::core::project_editor::ProjectEditor";
    editor.queue_by_id(symbol_id, FieldMutation::RenameIdent("ProjectAstEditor".to_string()))?;

    let conflicts = editor.validate()?;
    if !conflicts.is_empty() {
        eprintln!("Conflicts: {conflicts:?}");
    }

    // apply/commit are currently stubs; this demonstrates the orchestration flow.
    let report = editor.apply()?;
    println!("Queued edits touching: {:?}", report.touched_files);
    if !report.conflicts.is_empty() {
        println!("Conflicts: {:?}", report.conflicts);
    }
    println!("Preview: {}", editor.preview()?);

    Ok(())
}
