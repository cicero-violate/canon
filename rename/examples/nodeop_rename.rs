#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::core::project_editor::ProjectEditor;
use rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target: rename crate itself
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // 5 confusing names in the rename crate, confirmed from SYMBOLS.json
    let renames = [
        // "Enhanced" is meaningless — no non-enhanced version exists
        ("crate::occurrence::EnhancedOccurrenceVisitor", "OccurrenceVisitor"),

        // "Symbol" is overloaded — this walks AST items, not symbols
        ("crate::core::collect::collector::SymbolCollector", "ItemCollector"),

        // "Enhanced" prefix on ImplContext inside occurrence is redundant noise
        ("crate::occurrence::ImplContext", "OccurrenceImplContext"),

        // "Structured" prefix is vague — this tracks an edit session
        ("crate::core::structured::StructuredEditTracker", "EditSessionTracker"),

        // Ambiguous with GraphSnapshot in state — needs Graph prefix
        ("crate::core::project_editor::SnapshotOracle", "GraphSnapshotOracle"),
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
