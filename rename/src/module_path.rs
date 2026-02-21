//! Module path computation and file/module move planning
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
/// Represents a module in the Rust module hierarchy
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}
impl ModulePath {
    pub fn new(segments: Vec<String>) -> Self {
        Self { segments, is_inline: false }
    }
    pub fn from_string(path: &str) -> Self {
        let segments = path.split("::").map(String::from).collect();
        Self::new(segments)
    }
    pub fn to_string(&self) -> String {
        self.segments.join("::")
    }
    pub fn parent(&self) -> Option<ModulePath> {
        if self.segments.len() <= 1 {
            return None;
        }
        Some(ModulePath::new(self.segments[..self.segments.len() - 1].to_vec()))
    }
    pub fn last_segment(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }
}
/// Represents the physical layout of a module
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
impl ModuleLayout {
    /// Get the file path for this layout
    pub fn file_path(&self) -> Option<&Path> {
        match self {
            ModuleLayout::Inline => None,
            ModuleLayout::File(p) | ModuleLayout::DirectoryModRs(p) | ModuleLayout::DirectoryNamed(p) => Some(p),
        }
    }
    /// Check if this is a directory-based layout
    pub fn is_dir_path(&self) -> bool {
        matches!(self, ModuleLayout::DirectoryModRs(_) | ModuleLayout::DirectoryNamed(_))
    }
}
/// Normalize module path computation for different file layouts
pub fn compute_module_path(project_root: &Path, file_path: &Path) -> Result<String> {
    let rel_path = file_path.strip_prefix(project_root).context("File not in project")?;
    let mut segments = Vec::new();
    let mut components: Vec<_> = rel_path.components().collect();
    if components.first().and_then(|c| c.as_os_str().to_str()) == Some("src") {
        components.remove(0);
    }
    for (i, component) in components.iter().enumerate() {
        let comp_str = component.as_os_str().to_str().context("Invalid UTF-8 in path")?;
        let is_last = i == components.len() - 1;
        if is_last {
            if comp_str == "mod.rs" {
            } else if comp_str == "lib.rs" || comp_str == "main.rs" {
            } else if let Some(stem) = Path::new(comp_str).file_stem() {
                segments.push(stem.to_str().unwrap().to_string());
            }
        } else {
            segments.push(comp_str.to_string());
        }
    }
    if segments.is_empty() {
        Ok("crate".to_string())
    } else {
        Ok(format!("crate::{}", segments.join("::")))
    }
}
/// Determine the layout of a module given its path
pub fn detect_module_layout(file_path: &Path) -> ModuleLayout {
    let file_name = file_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if file_name == "mod.rs" {
        ModuleLayout::DirectoryModRs(file_path.to_path_buf())
    } else {
        if let Some(stem) = file_path.file_stem() {
            let dir_path = file_path.parent().unwrap().join(stem);
            if dir_path.is_dir() {
                ModuleLayout::DirectoryNamed(file_path.to_path_buf())
            } else {
                ModuleLayout::File(file_path.to_path_buf())
            }
        } else {
            ModuleLayout::File(file_path.to_path_buf())
        }
    }
}
/// Plan for moving a module
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
#[derive(Debug, Clone)]
pub enum LayoutChange {
    /// Convert inline module to file
    InlineToFile,
    /// Convert file module to inline
    FileToInline,
    /// Convert between directory layouts
    DirectoryLayoutChange { from: ModuleLayout, to: ModuleLayout },
}
impl ModuleMovePlan {
    /// Create a plan to move a module
    pub fn new(from_path: ModulePath, to_path: ModulePath, from_file: PathBuf, project_root: &Path) -> Result<Self> {
        let to_file = compute_new_file_path(&to_path, project_root)?;
        let create_directory = if let Some(parent) = to_file.parent() { !parent.exists() } else { false };
        Ok(Self { from_path, to_path, from_file, to_file, create_directory, layout_change: None })
    }
}
/// Compute the new file path for a module given its new module path
pub(crate) fn compute_new_file_path(module_path: &ModulePath, project_root: &Path) -> Result<PathBuf> {
    let mut path = project_root.join("src");
    let segments: Vec<_> = module_path.segments.iter().skip_while(|s| *s == "crate").collect();
    if segments.is_empty() {
        return Ok(path.join("lib.rs"));
    }
    for segment in &segments[..segments.len().saturating_sub(1)] {
        path = path.join(segment);
    }
    if let Some(last) = segments.last() {
        path = path.join(format!("{}.rs", last));
    }
    Ok(path)
}
/// Convert an inline module to a file module
pub fn inline_mod_to_file_plan(module_name: &str, parent_file: &Path, project_root: &Path) -> Result<ModuleMovePlan> {
    let parent_module = compute_module_path(project_root, parent_file)?;
    let from_path = ModulePath::from_string(&format!("{}::{}", parent_module, module_name));
    let parent_dir = parent_file.parent().context("Parent file has no parent directory")?;
    let new_file = parent_dir.join(format!("{}.rs", module_name));
    let to_path = from_path.clone();
    let plan = ModuleMovePlan { from_path, to_path, from_file: parent_file.to_path_buf(), to_file: new_file, create_directory: false, layout_change: Some(LayoutChange::InlineToFile) };
    Ok(plan)
}
/// Convert a file module to an inline module
pub fn file_mod_to_inline_plan(file_path: &Path, target_parent_file: &Path, project_root: &Path) -> Result<ModuleMovePlan> {
    let from_path = ModulePath::from_string(&compute_module_path(project_root, file_path)?);
    let to_path = from_path.clone();
    let plan =
        ModuleMovePlan { from_path, to_path, from_file: file_path.to_path_buf(), to_file: target_parent_file.to_path_buf(), create_directory: false, layout_change: Some(LayoutChange::FileToInline) };
    Ok(plan)
}
