#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

// rename::* is redundant/confusing.
// Crate name is already `rename`.
use rename::core::project_editor::ProjectEditor;
use rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target memory/database crate
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/memory/database/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // Use ONLY fully-qualified symbols confirmed in memory/database SYMBOLS.json
    let renames = [
        // struct crate::tlog::tlog::TransactionLog
        ("crate::tlog::tlog::TransactionLog", "DeltaLog"),

        // trait crate::engine::Engine
        ("crate::engine::Engine", "DeltaExecutionEngine"),

        // type crate::primitives::Hash
        ("crate::primitives::Hash", "StateHash"),
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
