//! Source ingestor scaffolding (ING-001).
//!
//! This module will ingest an existing Rust workspace into `CanonicalIr`.
//! The actual implementation will land once `TODO_self_replicate_DAG.md` node
//! `ING-001` is addressed.

mod builder;
mod fs_walk;
mod parser;

use std::path::{Path, PathBuf};

use crate::layout::LayoutMap;

/// Options that control workspace ingestion.
pub struct IngestOptions {
    pub root: PathBuf,
}

impl IngestOptions {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

/// Placeholder error surface for the future ingestor.
#[derive(Debug)]
pub enum IngestError {
    Io(std::io::Error),
    Parse(String),
    UnsupportedFeature(String),
}

impl From<std::io::Error> for IngestError {
    fn from(err: std::io::Error) -> Self {
        IngestError::Io(err)
    }
}

/// Entry point for converting an existing workspace into semantic + layout graphs.
pub fn ingest_workspace(opts: &IngestOptions) -> Result<LayoutMap, IngestError> {
    let files = fs_walk::discover_source_files(&opts.root)?;
    let parsed = parser::parse_workspace(&opts.root, &files)?;
    let layout_map = builder::build_layout_map(&opts.root, parsed)?;
    Ok(layout_map)
}

fn _ensure_path_is_dir(path: &Path) -> Result<(), IngestError> {
    if path.is_dir() {
        Ok(())
    } else {
        Err(IngestError::UnsupportedFeature(format!(
            "ingest requires a directory root: {}",
            path.display()
        )))
    }
}
