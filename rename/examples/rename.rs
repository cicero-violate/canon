use rename::apply_rename_with_map;
use std::collections::HashMap;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Target project root (crate to refactor)
    let project_path =
        Path::new("/workspace/ai_sandbox/canon_workspace/canon");

    // IMPORTANT:
    // Always use fully-qualified paths to avoid global identifier corruption.
    // This prevents doc strings, comments, and unrelated modules from breaking.
    let mut map = HashMap::new();

    map.insert(
        "crate::runtime::value::Value".to_string(),
        "RuntimeValue".to_string(),
    );

    map.insert(
        "crate::runtime::context::ExecutionContext".to_string(),
        "RuntimeContext".to_string(),
    );

    map.insert(
        "crate::ir::delta::Delta".to_string(),
        "IrDelta".to_string(),
    );

    map.insert(
        "crate::ir::proofs::Proof".to_string(),
        "IrProof".to_string(),
    );

    map.insert(
        "crate::ir::timeline::Plan".to_string(),
        "ExecutionPlan".to_string(),
    );

    // false  -> do not include tests
    // None   -> no file filter
    apply_rename_with_map(project_path, &map, false, None)?;

    println!("Scoped renames applied successfully.");
    Ok(())
}

