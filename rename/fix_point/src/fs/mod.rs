mod include_visitor;

use crate::fs::include_visitor::IncludeVisitor;


use anyhow::Result;


use std::path::{Path, PathBuf};


use syn::visit::{self, Visit};


use walkdir::WalkDir;


#[derive(Debug, Clone)]
pub struct AuxiliaryFile {
    pub path: PathBuf,
    pub kind: AuxiliaryKind,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone)]
pub struct DiscoveredFiles {
    pub rust_files: Vec<PathBuf>,
    pub auxiliary_files: Vec<AuxiliaryFile>,
}
