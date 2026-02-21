#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // 3 targeted renames pulled from SYMBOLS.json
    let renames = [
        ("crate::core::project_editor::cross_file::normalize_snippet", "normalize_insert_snippet"),
        ("crate::core::project_editor::cross_file::parse_single_item", "parse_single_item_snippet"),
        ("crate::core::project_editor::cross_file::validate_item_matches", "validate_moved_item"),
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
