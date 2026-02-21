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
    // Symbol IDs must exist in the loaded project.
    // Use symbol index inspection to discover valid IDs.
    //
    // Example valid rename (must exist in loaded project):
    let renames = [
        // IR → State
        ("crate::ir::core::CanonicalIr", "SystemState"),
        ("crate::storage::reader::MemoryIrReader", "StateReader"),
        ("crate::storage::builder::MemoryIrBuilder", "StateWriter"),
        // Tick → Execution
        ("crate::ir::graphs::TickGraph", "ExecutionGraph"),
        ("crate::runtime::tick_executor::TickExecutor", "ExecutionEngine"),
        ("crate::ir::timeline::TickEpoch", "ExecutionEpoch"),
        // Delta → StateChange
        ("crate::ir::delta::Delta", "StateChange"),
        ("crate::ir::admission::DeltaAdmission", "ChangeAdmission"),
        ("crate::runtime::delta_verifier::DeltaVerifier", "ChangeVerifier"),
        ("crate::ir::delta::DeltaPayload", "ChangePayload"),
        // Judgment → Decision / Rule
        ("crate::ir::judgment::Judgment", "Decision"),
        ("crate::ir::judgment::JudgmentPredicate", "Rule"),
        // Layout → FileTopology
        ("crate::layout::LayoutGraph", "FileTopology"),
        ("crate::layout::LayoutAssignment", "FileBinding"),
        ("crate::layout::SemanticGraph", "ParsedModel"),
        // Capability → Agent
        ("crate::agent::capability::CapabilityNode", "AgentNode"),
        ("crate::agent::capability::CapabilityGraph", "AgentGraph"),
        ("crate::agent::dispatcher::CapabilityNodeDispatcher", "AgentScheduler"),
        // Evolution / Drift
        ("crate::evolution::lyapunov::TopologyFingerprint", "StructureMetrics"),
        ("crate::evolution::lyapunov::LyapunovError", "StructureDriftError"),
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
