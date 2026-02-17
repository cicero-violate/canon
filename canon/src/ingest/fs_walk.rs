use std::path::{Component, Path, PathBuf};

use walkdir::WalkDir;

use super::IngestError;

#[derive(Debug, Clone)]
pub(crate) struct DiscoveredFile {
    pub absolute: PathBuf,
    pub relative: PathBuf,
}

/// Discover candidate Rust source files under the workspace root.
///
/// TODO(ING-001): Respect `.gitignore`/`Cargo.toml` module declarations.
pub(crate) fn discover_source_files(root: &Path) -> Result<Vec<DiscoveredFile>, IngestError> {
    super::_ensure_path_is_dir(root)?;

    eprintln!("INGEST DEBUG: root = {}", root.display());
    eprintln!("INGEST DEBUG: root.exists() = {}", root.exists());
    eprintln!("INGEST DEBUG: root.is_dir() = {}", root.is_dir());

    let src_root = resolve_src_root(root)?;

    eprintln!("INGEST DEBUG: final src_root = {}", src_root.display());
    eprintln!(
        "INGEST DEBUG: final src_root.is_dir() = {}",
        src_root.is_dir()
    );

    let mut files = Vec::new();
    for entry in WalkDir::new(&src_root)
        .into_iter()
        .filter_entry(|e| !is_ignored(e.path()))
    {
        let entry = entry.map_err(map_walkdir_error)?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .or_else(|_| path.strip_prefix(&src_root))
            .unwrap_or(path)
            .to_path_buf();
        files.push(DiscoveredFile {
            absolute: path.to_path_buf(),
            relative,
        });
    }
    // Allow empty src directories (needed for materialize → ingest roundtrip tests)
    if files.is_empty() {
        return Ok(Vec::new());
    }
    Ok(files)
}

fn is_ignored(path: &Path) -> bool {
    for component in path.components() {
        if let Component::Normal(name) = component {
            if let Some(seg) = name.to_str() {
                if matches!(seg, "target" | "tests" | "examples") {
                    return true;
                }
            }
        }
    }
    false
}

fn resolve_src_root(root: &Path) -> Result<PathBuf, IngestError> {
    if root
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| name == "src")
        .unwrap_or(false)
    {
        return Ok(root.to_path_buf());
    }

    let candidate = root.join("src");
    if candidate.is_dir() {
        return Ok(candidate);
    }

    Err(IngestError::UnsupportedFeature(format!(
        "ING-001: ingest expects a crate directory containing `src/` or the `src/` directory itself. Provided root `{}` does not contain readable sources.",
        root.display()
    )))
}

fn map_walkdir_error(err: walkdir::Error) -> IngestError {
    if let Some(io_err) = err.io_error() {
        IngestError::Io(std::io::Error::new(io_err.kind(), io_err.to_string()))
    } else {
        IngestError::UnsupportedFeature(err.to_string())
    }
}
