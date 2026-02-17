use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use crate::{
    CanonicalIr,
    layout::{LayoutGraph, SemanticGraph},
    semantic_builder::SemanticIrBuilder,
    storage::reader::MemoryIrReader,
};

fn load_ir_any(path: &Path) -> Result<CanonicalIr, Box<dyn std::error::Error>> {
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("bin"))
        .unwrap_or(false)
    {
        return Ok(MemoryIrReader::from_checkpoint(path)?);
    }
    let data = fs::read(path)?;
    // Try full CanonicalIr first (has `meta` field).
    if let Ok(ir) = serde_json::from_slice::<CanonicalIr>(&data) {
        return Ok(ir);
    }
    // Fall back to SemanticGraph (produced by canon ingest).
    let semantic: SemanticGraph = serde_json::from_slice(&data)?;
    let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("canon");
    // Strip everything from the first dot so "canon.semantic" → "canon",
    // since Word rejects strings containing punctuation.
    let base_name = name.split('.').next().unwrap_or("canon");
    Ok(SemanticIrBuilder::new(base_name).build(semantic))
}

pub fn load_ir(path: &Path) -> Result<CanonicalIr, Box<dyn std::error::Error>> {
    load_ir_any(path)
}

/// Load either a full `CanonicalIr` or a `SemanticGraph` produced by
/// `canon ingest`. When a semantic graph is detected it is promoted to
/// a `CanonicalIr` via `SemanticIrBuilder`.
pub fn load_ir_or_semantic(path: &Path) -> Result<CanonicalIr, Box<dyn std::error::Error>> {
    load_ir_any(path)
}

pub fn load_layout(path: PathBuf) -> Result<LayoutGraph, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    Ok(serde_json::from_slice(&data)?)
}

pub fn resolve_layout(arg: Option<PathBuf>, ir: &Path) -> PathBuf {
    arg.unwrap_or_else(|| default_layout_path_for(ir))
}

pub fn default_layout_path_for(ir_path: &Path) -> PathBuf {
    let stem = ir_path
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| OsString::from("canonical"));
    let mut layout_name = stem;
    layout_name.push(".layout.json");
    ir_path.with_file_name(layout_name)
}

/// Derives the default capability graph path for a given IR path.
/// e.g. `canon.ir.json` → `canon.graph.json`
pub fn default_graph_path_for(ir_path: &Path) -> PathBuf {
    let stem = ir_path
        .file_stem()
        .map(|s| s.to_os_string())
        .unwrap_or_else(|| OsString::from("canonical"));
    let mut graph_name = stem;
    graph_name.push(".graph.json");
    ir_path.with_file_name(graph_name)
}
