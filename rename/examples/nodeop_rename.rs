#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

// rename::* is redundant/confusing.
// Crate name is already `rename`.
use rename::core::project_editor::ProjectEditor;
use rename::structured::FieldMutation;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target external project to refactor (example: canon crate)
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/canon/src");
    let mut editor = ProjectEditor::load_with_rustc(project_path)?;

    // NOTE:
    // Ensure these symbol IDs actually exist in the loaded Canon project.
    // Remove or adjust entries if "no handle found" occurs.

    // NOTE:
    // Symbol IDs must exist in the loaded project.
    // Use symbol index inspection to discover valid IDs.
    //
    // Example valid rename (must exist in loaded project):
    let renames = [
        // IR → Canonical State
        ("crate::ir::core::SystemState", "CanonicalState"),
        ("crate::ir::delta::Delta", "CanonicalDelta"),
        ("crate::ir::delta::DeltaPayload", "CanonicalDeltaPayload"),
        // Execution terminology
        ("crate::ir::graphs::TickGraph", "ExecutionPlanGraph"),
        ("crate::runtime::tick_executor::TickExecutor", "ExecutionEngine"),
        // Judgment terminology
        ("crate::ir::judgment::JudgmentPredicate", "JudgmentRule"),
        // Layout terminology
        ("crate::layout::LayoutGraph", "SourceTopology"),
        // Agent terminology
        ("crate::agent::capability::CapabilityGraph", "AgentTopology"),
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
