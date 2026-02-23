use anyhow::{Context, Result};


use std::path::{Path, PathBuf};


#[derive(Debug, Clone)]
pub enum LayoutChange {
    /// Convert inline module to file
    InlineToFile,
    /// Convert file module to inline
    FileToInline,
    /// Convert between directory layouts
    DirectoryLayoutChange { from: ModuleLayout, to: ModuleLayout },
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone)]
pub struct ModuleMovePlan {
    /// Original module path
    pub from_path: ModulePath,
    /// New module path
    pub to_path: ModulePath,
    /// Original file location
    pub from_file: PathBuf,
    /// New file location
    pub to_file: PathBuf,
    /// Whether this requires creating a new directory
    pub create_directory: bool,
    /// Whether this converts between inline and file module
    pub layout_change: Option<LayoutChange>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}
