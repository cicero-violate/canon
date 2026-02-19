#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::rename::core::project_editor::ProjectEditor;
use rename::rename::structured::NodeOp;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    let symbol_id = "crate::rename::core::project_editor::ProjectEditor";
    let handle = editor
        .registry
        .handles
        .get(symbol_id)
        .cloned()
        .ok_or("symbol handle not found")?;

    editor.queue(
        symbol_id,
        NodeOp::MoveSymbol {
            handle,
            new_module_path: "crate::rename::core::project_editor".to_string(),
            new_crate: None,
        },
    )?;

    let conflicts = editor.validate()?;
    println!("conflicts: {conflicts:?}");

    let report = editor.apply()?;
    println!("touched: {:?}", report.touched_files);
    println!("file moves: {:?}", report.file_moves);

    let preview = editor.preview()?;
    println!("preview:\n{preview}");

    // Uncomment to persist changes on disk.
    // let written = editor.commit()?;
    // println!("written: {:?}", written);

    Ok(())
}
