#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");

    let mut editor = ProjectEditor::load_with_rustc(project_path)?;
    editor.queue_by_id("crate::rename::core::project_editor::ProjectEditor", FieldMutation::RenameIdent("ProjectEditor2".to_string()))?;

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched: {:?}", report.touched_files);

    let preview = editor.preview()?;
    println!("preview: {preview}");

    Ok(())
}
