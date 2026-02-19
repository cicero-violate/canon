use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use syn::visit::{self, Visit};

use crate::fs;

use super::paths::module_path_for_file;
use super::span::span_to_range;
use super::types::{SymbolEdit, SymbolIndex};
use super::use_map::normalize_use_prefix;

/// Track and rename use statement aliases
pub(crate) fn collect_and_rename_aliases(
    project: &Path,
    symbol_table: &SymbolIndex,
    mapping: &HashMap<String, String>,
) -> Result<Vec<SymbolEdit>> {
    let mut alias_edits = Vec::new();
    let files = fs::collect_rs_files(project)?;

    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;

        // Process use statements to find aliases
        for item in &ast.items {
            if let syn::Item::Use(use_item) = item {
                collect_alias_edits(
                    &use_item.tree,
                    use_item.leading_colon.is_some(),
                    &module_path,
                    file,
                    symbol_table,
                    mapping,
                    &mut alias_edits,
                );
            }
        }
    }

    Ok(alias_edits)
}

fn collect_alias_edits(
    tree: &syn::UseTree,
    has_leading_colon: bool,
    module_path: &str,
    file: &Path,
    symbol_table: &SymbolIndex,
    mapping: &HashMap<String, String>,
    edits: &mut Vec<SymbolEdit>,
) {
    let mut prefix = Vec::new();
    if has_leading_colon {
        prefix.push("crate".to_string());
    }

    collect_alias_edits_recursive(
        tree,
        &mut prefix,
        module_path,
        file,
        symbol_table,
        mapping,
        edits,
    );
}

fn collect_alias_edits_recursive(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    module_path: &str,
    file: &Path,
    symbol_table: &SymbolIndex,
    mapping: &HashMap<String, String>,
    edits: &mut Vec<SymbolEdit>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            prefix.push(path.ident.to_string());
            collect_alias_edits_recursive(
                &path.tree,
                prefix,
                module_path,
                file,
                symbol_table,
                mapping,
                edits,
            );
            prefix.pop();
        }
        syn::UseTree::Rename(rename) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(rename.ident.to_string());
            let target_id = full.join("::");

            // Check if target symbol is being renamed
            if let Some(new_name) = mapping.get(&target_id) {
                // Rename the target path (left side of 'as')
                edits.push(SymbolEdit {
                    id: target_id.clone(),
                    file: file.to_string_lossy().to_string(),
                    kind: "use_alias_target".to_string(),
                    start: span_to_range(rename.ident.span()).start,
                    end: span_to_range(rename.ident.span()).end,
                    new_name: new_name.clone(),
                });
            }

            // Check if alias itself should be renamed
            let alias_id = format!("{}@alias:{}", target_id, rename.rename);
            if let Some(new_alias) = mapping.get(&alias_id) {
                // Rename the alias (right side of 'as')
                edits.push(SymbolEdit {
                    id: alias_id.clone(),
                    file: file.to_string_lossy().to_string(),
                    kind: "use_alias_name".to_string(),
                    start: span_to_range(rename.rename.span()).start,
                    end: span_to_range(rename.rename.span()).end,
                    new_name: new_alias.clone(),
                });
            }

            // Also need to update usages of the alias in this file
            let alias_name = rename.rename.to_string();
            if let Some(new_alias) = mapping.get(&alias_id) {
                collect_alias_usage_edits(file, &alias_name, new_alias, &target_id, edits);
            }
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_alias_edits_recursive(
                    item,
                    prefix,
                    module_path,
                    file,
                    symbol_table,
                    mapping,
                    edits,
                );
            }
        }
        _ => {}
    }
}

fn collect_alias_usage_edits(
    file: &Path,
    alias_name: &str,
    new_alias: &str,
    target_id: &str,
    edits: &mut Vec<SymbolEdit>,
) {
    // Re-parse file to find alias usages
    if let Ok(content) = std::fs::read_to_string(file) {
        if let Ok(ast) = syn::parse_file(&content) {
            let mut visitor = AliasUsageVisitor {
                alias_name: alias_name.to_string(),
                new_alias: new_alias.to_string(),
                target_id: target_id.to_string(),
                file,
                edits,
            };
            visitor.visit_file(&ast);
        }
    }
}

struct AliasUsageVisitor<'a> {
    alias_name: String,
    new_alias: String,
    target_id: String,
    file: &'a Path,
    edits: &'a mut Vec<SymbolEdit>,
}

impl<'ast> Visit<'ast> for AliasUsageVisitor<'_> {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        // Check if first segment matches the alias
        if let Some(first_seg) = path.segments.first() {
            if first_seg.ident.to_string() == self.alias_name {
                self.edits.push(SymbolEdit {
                    id: format!("{}@alias_use", self.target_id),
                    file: self.file.to_string_lossy().to_string(),
                    kind: "alias_usage".to_string(),
                    start: span_to_range(first_seg.ident.span()).start,
                    end: span_to_range(first_seg.ident.span()).end,
                    new_name: self.new_alias.clone(),
                });
            }
        }
        visit::visit_path(self, path);
    }
}
