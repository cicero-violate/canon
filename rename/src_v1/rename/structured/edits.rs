use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::ast_render;
use crate::rename::rewrite::RewriteBufferSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}

impl AstEdit {
    pub fn replace<T: quote::ToTokens>(
        file: impl Into<PathBuf>,
        start: usize,
        end: usize,
        node: &T,
    ) -> Self {
        Self {
            file: file.into(),
            start,
            end,
            replacement: ast_render::render_node(node),
        }
    }

    pub fn insert<T: quote::ToTokens>(file: impl Into<PathBuf>, offset: usize, node: &T) -> Self {
        Self {
            file: file.into(),
            start: offset,
            end: offset,
            replacement: ast_render::render_node(node),
        }
    }
}

pub fn apply_ast_rewrites(edits: &[AstEdit], format: bool) -> Result<Vec<PathBuf>> {
    let mut buffers = RewriteBufferSet::new();
    let mut file_contents: HashMap<PathBuf, String> = HashMap::new();

    for edit in edits {
        if !file_contents.contains_key(&edit.file) {
            let content = std::fs::read_to_string(&edit.file)?;
            file_contents.insert(edit.file.clone(), content);
        }
    }

    for edit in edits {
        let content = file_contents.get(&edit.file).unwrap();
        let buffer = buffers.ensure_buffer(&edit.file, content);
        buffer.replace(edit.start, edit.end, &edit.replacement)?;
    }

    let touched = buffers.flush()?;

    if format && !touched.is_empty() {
        for file in &touched {
            if file.exists() {
                let _ = std::process::Command::new("rustfmt")
                    .arg("--edition")
                    .arg("2021")
                    .arg(file)
                    .status();
            }
        }
    }

    Ok(touched)
}
