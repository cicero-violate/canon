use rename::core::project_editor::ProjectEditor;
use rename::core::StructuralEditOracle;
use rename::structured::NodeOp;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");
    let oracle = Box::new(StructuralEditOracle::default());
    let mut editor = ProjectEditor::load(project_path, oracle)?;

    // Cross-file MoveSymbol test: move three utils functions into editor.rs.
    let moves = [
        ("crate::core::project_editor::utils::build_symbol_index", "crate::core::project_editor::editor"),
        ("crate::core::project_editor::utils::find_project_root", "crate::core::project_editor::editor"),
        ("crate::core::project_editor::utils::find_project_root_sync", "crate::core::project_editor::editor"),
    ];

    for (symbol_id, new_module) in moves {
        let handle = editor
            .registry
            .handles
            .get(symbol_id)
            .cloned()
            .ok_or_else(|| format!("symbol not found: {symbol_id}"))?;

        println!("queue move: {symbol_id} -> {new_module}");
        editor.queue(symbol_id, NodeOp::MoveSymbol {
            handle,
            symbol_id: symbol_id.to_string(),
            new_module_path: new_module.to_string(),
            new_crate: None,
        })?;
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
        println!("preview:
{preview}");
        println!("
(dry-run — pass --commit to apply)");
    }

    Ok(())
}
