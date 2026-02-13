use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

use canon::{layout::LayoutGraph, CanonicalIr};

pub fn load_ir(path: &Path) -> Result<CanonicalIr, Box<dyn std::error::Error>> {
    let data = fs::read(path)?;
    Ok(serde_json::from_slice(&data)?)
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

