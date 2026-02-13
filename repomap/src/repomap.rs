use crate::extractor::extract_symbols;
use crate::symbol::Symbol;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// One file's worth of extracted symbols.
pub struct FileMap {
    pub path: PathBuf,
    pub symbols: Vec<Symbol>,
}

/// Walk `root_dir`, parse every `.rs` file, return a list of FileMaps.
pub fn build_repomap(root_dir: &Path) -> Vec<FileMap> {
    let mut result = Vec::new();

    for entry in WalkDir::new(root_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "rs").unwrap_or(false))
    {
        let path = entry.path().to_path_buf();
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let symbols = extract_symbols(&src);
        if !symbols.is_empty() {
            result.push(FileMap { path, symbols });
        }
    }

    result
}

/// Render the full repo map as a compact string ready to inject into
/// a planner LLM prompt.
pub fn render_repomap(maps: &[FileMap], root_dir: &Path) -> String {
    let mut out = String::new();

    for fm in maps {
        // Show path relative to root so the LLM sees `src/graph.rs` not abs path
        let rel = fm.path.strip_prefix(root_dir).unwrap_or(&fm.path);
        out.push_str(&format!("{}:\n", rel.display()));

        for sym in &fm.symbols {
            out.push_str(&sym.render());
            out.push('\n');
        }
        out.push('\n');
    }

    out
}

/// Count approximate tokens (rough: 1 token ≈ 4 chars).
pub fn estimate_tokens(s: &str) -> usize {
    s.len() / 4
}
