#![feature(rustc_private)]

use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    let root = std::env::args().nth(1).map(PathBuf::from).expect("workspace path required");
    run_pipeline(root)
}

fn run_pipeline(root: PathBuf) -> Result<()> {
    println!("Starting pipeline on {:?}", root);

    // Stage 1: construct ModelIR (capture_rustc will populate nodes later).
    let mut ir = model::ir::model_ir::ModelIR::new();

    // Stage 2: derive constraint graphs + solve.
    analyzer::analyze(&mut ir)
        .map_err(|e| anyhow::anyhow!("analysis failed: {e}"))?;

    // Stage 3: project ModelIR -> file plan -> emit to disk.
    let plan = projection::project(&ir)
        .map_err(|e| anyhow::anyhow!("project failed: {e}"))?;
    projection::emit_to_disk(&plan, &root)
        .map_err(|e| anyhow::anyhow!("emit failed: {e}"))?;

    // Stage 4: JSON snapshot for inspection.
    let out_path = root.join("model_ir.json");
    let json = serde_json::to_string_pretty(&ir)
        .map_err(|e| anyhow::anyhow!("json serialize failed: {e}"))?;
    std::fs::write(&out_path, &json)
        .map_err(|e| anyhow::anyhow!("json write failed: {e}"))?;

    println!("Emitted {} files.", plan.files.len());
    println!("Pipeline complete.");
    Ok(())
}
