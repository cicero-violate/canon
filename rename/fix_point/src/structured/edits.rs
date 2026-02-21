use anyhow::{bail, Result};


use serde::{Deserialize, Serialize};


use std::collections::HashMap;


use std::path::PathBuf;


use syn::spanned::Spanned;


use super::ast_render;


use crate::model::core_span::span_to_offsets;


use crate::model::span::LineColumn;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}


pub fn apply_ast_rewrites(edits: &[AstEdit], format: bool) -> Result<Vec<PathBuf>> {
    let mut file_contents: HashMap<PathBuf, String> = HashMap::new();
    let mut touched: Vec<PathBuf> = Vec::new();
    for edit in edits {
        if !file_contents.contains_key(&edit.file) {
            let content = std::fs::read_to_string(&edit.file)?;
            file_contents.insert(edit.file.clone(), content);
        }
    }
    let mut edits_by_file: HashMap<PathBuf, Vec<&AstEdit>> = HashMap::new();
    for edit in edits {
        edits_by_file.entry(edit.file.clone()).or_default().push(edit);
    }
    for (file, file_edits) in edits_by_file {
        let content = file_contents.get(&file).unwrap();
        let mut ast = syn::parse_file(content)?;
        let mut changed = false;
        for edit in file_edits {
            let replacement_file = syn::parse_file(&edit.replacement)
                .or_else(|_| syn::parse_file(&format!("{}\n", edit.replacement)))?;
            let replacement_items = replacement_file.items;
            if edit.start == edit.end {
                if insert_items_at_offset(
                    &mut ast,
                    content,
                    edit.start,
                    replacement_items,
                )? {
                    changed = true;
                }
            } else {
                if replace_items_in_range(
                    &mut ast,
                    content,
                    edit.start,
                    edit.end,
                    replacement_items,
                )? {
                    changed = true;
                }
            }
        }
        if changed {
            let rendered = prettyplease::unparse(&ast);
            if rendered != *content {
                std::fs::write(&file, rendered)?;
            }
            touched.push(file);
        }
    }
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


fn find_insert_index(ast: &syn::File, content: &str, offset: usize) -> usize {
    for (index, item) in ast.items.iter().enumerate() {
        let start = span_to_offsets(
                content,
                &span_to_line_column(item.span().start()),
                &span_to_line_column(item.span().end()),
            )
            .0;
        if start >= offset {
            return index;
        }
    }
    ast.items.len()
}


fn insert_items_at_offset(
    ast: &mut syn::File,
    content: &str,
    offset: usize,
    mut items: Vec<syn::Item>,
) -> Result<bool> {
    if items.is_empty() {
        return Ok(false);
    }
    let index = find_insert_index(ast, content, offset);
    for (i, item) in items.drain(..).enumerate() {
        ast.items.insert(index + i, item);
    }
    Ok(true)
}


fn replace_items_in_range(
    ast: &mut syn::File,
    content: &str,
    start: usize,
    end: usize,
    items: Vec<syn::Item>,
) -> Result<bool> {
    let mut ranges = Vec::new();
    for (index, item) in ast.items.iter().enumerate() {
        let span = span_to_offsets(
            content,
            &span_to_line_column(item.span().start()),
            &span_to_line_column(item.span().end()),
        );
        ranges.push((index, span.0, span.1));
    }
    let mut target_indices: Vec<usize> = ranges
        .iter()
        .filter(|(_, item_start, item_end)| *item_start <= end && *item_end >= start)
        .map(|(index, _, _)| *index)
        .collect();
    if target_indices.is_empty() {
        bail!("no AST items overlap edit range {start}..{end}");
    }
    target_indices.sort();
    let first = target_indices[0];
    let last = *target_indices.last().unwrap();
    ast.items.splice(first..=last, items);
    Ok(true)
}


fn span_to_line_column(point: proc_macro2::LineColumn) -> LineColumn {
    LineColumn {
        line: point.line as i64,
        column: point.column as i64 + 1,
    }
}
