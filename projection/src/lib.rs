use anyhow::Result;
use model::ir::model_ir::ModelIR;
use std::path::{Path, PathBuf};
use std::fs;

pub mod emit;

#[derive(Debug, Clone)]
pub struct Plan {
    pub files: Vec<(PathBuf, String)>,
}

/// Walk ModelIR and produce a Plan: one entry per Module node.
pub fn project(ir: &ModelIR) -> Result<Plan> {
    let mut files: Vec<(PathBuf, String)> = emit::emit_files(ir)
        .into_iter()
        .map(|(path, src)| (PathBuf::from(path), src))
        .collect();
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Plan { files })
}

/// Write each file in the plan to disk under `root`.
pub fn emit_to_disk(plan: &Plan, root: &Path) -> Result<()> {
    for (path, content) in &plan.files {
        let full = root.join(path);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(full, content)?;
    }
    Ok(())
}
