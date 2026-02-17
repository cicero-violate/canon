use std::fs;
use std::path::{Path, PathBuf};

use super::IngestError;
use super::fs_walk::DiscoveredFile;

use crate::ingest::_ensure_path_is_dir;

/// Placeholder representation of parsed items.
pub(crate) struct ParsedWorkspace {
    pub files: Vec<ParsedFile>,
}

pub(crate) struct ParsedFile {
    pub path: PathBuf,
    pub module_path: Vec<String>,
    pub ast: syn::File,
}

impl ParsedFile {
    pub fn path_string(&self) -> String {
        self.path.to_string_lossy().to_string()
    }
}

/// Parse the workspace into intermediate AST structures.
///
/// TODO(ING-001): Extend to capture items, attributes, and macro invocations.
pub(crate) fn parse_workspace(
    root: &Path,
    discovered: &[DiscoveredFile],
) -> Result<ParsedWorkspace, IngestError> {
    _ensure_path_is_dir(root)?;
    let mut files = Vec::new();
    for file in discovered {
        let source = fs::read_to_string(&file.absolute)?;
        let ast = syn::parse_file(&source).map_err(|err| IngestError::Parse(err.to_string()))?;
        let module_path = infer_module_path(&file.relative);
        files.push(ParsedFile {
            path: file.relative.clone(),
            module_path,
            ast,
        });
    }
    Ok(ParsedWorkspace { files })
}

fn infer_module_path(relative: &Path) -> Vec<String> {
    let mut parts = Vec::new();
    for component in relative.components() {
        let std::path::Component::Normal(seg) = component else {
            continue;
        };
        let Some(name) = seg.to_str() else {
            continue;
        };
        if name == "src" {
            continue;
        }
        if name == "mod.rs" || name == "lib.rs" {
            continue;
        }
        if name.ends_with(".rs") {
            parts.push(name.trim_end_matches(".rs").replace('-', "_").to_string());
        } else {
            parts.push(name.replace('-', "_"));
        }
    }
    parts
}
