use crate::core::paths::module_path_for_file;
use crate::core::project_editor::propagate::build_symbol_index_and_occurrences;
use crate::core::symbol_id::normalize_symbol_id;
use crate::model::types::SymbolIndex;
use crate::resolve::Resolver;
use crate::state::NodeRegistry;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use syn::visit_mut::VisitMut;

#[derive(Debug, Default, Clone)]
pub(crate) struct MoveSet {
    pub entries: HashMap<String, (String, String)>,
}

pub(crate) fn run_pass1_canonical_rewrite(registry: &mut NodeRegistry, moveset: &MoveSet) -> Result<HashSet<PathBuf>> {
    if moveset.entries.is_empty() {
        return Ok(HashSet::new());
    }
    let (symbol_table, _occurrences, alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let project_root = registry
        .asts
        .keys()
        .next()
        .and_then(|f| f.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let mut touched = HashSet::new();
    for (file, ast) in registry.asts.iter_mut() {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        let resolver = Resolver::new(&module_path, &alias_graph, &symbol_table);
        let mut rewriter = CanonicalRewriteVisitor {
            resolver,
            moveset,
            changed: false,
        };
        rewriter.visit_file_mut(ast);
        if rewriter.changed {
            let rendered = crate::structured::render_file(ast);
            registry.sources.insert(file.clone(), std::sync::Arc::new(rendered));
            touched.insert(file.clone());
        }
    }
    Ok(touched)
}

struct CanonicalRewriteVisitor<'a> {
    resolver: Resolver<'a>,
    moveset: &'a MoveSet,
    changed: bool,
}

impl<'a> CanonicalRewriteVisitor<'a> {
    fn rewrite_path(&mut self, path: &mut syn::Path) {
        let mut segments: Vec<String> = Vec::new();
        for seg in &path.segments {
            segments.push(seg.ident.to_string());
        }
        if segments.is_empty() {
            return;
        }
        let resolved = self.resolver.resolve_path_segments(&segments);
        let Some(resolved) = resolved else { return };
        let Some((_, new_module)) = self.moveset.entries.get(&resolved) else {
            return;
        };
        let name = resolved.rsplit("::").next().unwrap_or(&resolved);
        let mut new_segments: Vec<String> = new_module.split("::").map(|s| s.to_string()).collect();
        new_segments.push(name.to_string());
        let new_path_str = new_segments.join("::");
        if let Ok(new_path) = syn::parse_str::<syn::Path>(&new_path_str) {
            *path = new_path;
            self.changed = true;
        }
    }

    fn rewrite_use_tree(&mut self, tree: &mut syn::UseTree, prefix: &[String]) {
        match tree {
            syn::UseTree::Name(name) => {
                let mut full = prefix.to_vec();
                full.push(name.ident.to_string());
                if let Some(resolved) = self.resolver.resolve_path_segments(&full) {
                    if let Some((_, new_module)) = self.moveset.entries.get(&resolved) {
                        let name = resolved.rsplit("::").next().unwrap_or(&resolved);
                        let mut new_segments: Vec<String> = new_module.split("::").map(|s| s.to_string()).collect();
                        new_segments.push(name.to_string());
                        if let Some((first, rest)) = new_segments.split_first() {
                            *tree = build_use_tree(first, rest);
                            self.changed = true;
                        }
                    }
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut full = prefix.to_vec();
                full.push(rename.ident.to_string());
                if let Some(resolved) = self.resolver.resolve_path_segments(&full) {
                    if let Some((_, new_module)) = self.moveset.entries.get(&resolved) {
                        let name = resolved.rsplit("::").next().unwrap_or(&resolved);
                        let mut new_segments: Vec<String> = new_module.split("::").map(|s| s.to_string()).collect();
                        new_segments.push(name.to_string());
                        if let Some((first, rest)) = new_segments.split_first() {
                            let mut rebuilt = build_use_tree(first, rest);
                            if let syn::UseTree::Name(name_tree) = &mut rebuilt {
                                let rename_ident = rename.rename.clone();
                                rebuilt = syn::UseTree::Rename(syn::UseRename { ident: name_tree.ident.clone(), as_token: rename.as_token, rename: rename_ident });
                            }
                            *tree = rebuilt;
                            self.changed = true;
                        }
                    }
                }
            }
            syn::UseTree::Path(path) => {
                let mut next_prefix = prefix.to_vec();
                next_prefix.push(path.ident.to_string());
                self.rewrite_use_tree(&mut path.tree, &next_prefix);
            }
            syn::UseTree::Group(group) => {
                for item in &mut group.items {
                    self.rewrite_use_tree(item, prefix);
                }
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}

impl VisitMut for CanonicalRewriteVisitor<'_> {
    fn visit_path_mut(&mut self, node: &mut syn::Path) {
        self.rewrite_path(node);
        syn::visit_mut::visit_path_mut(self, node);
    }

    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        self.rewrite_use_tree(&mut node.tree, &Vec::new());
        syn::visit_mut::visit_item_use_mut(self, node);
    }

    fn visit_macro_mut(&mut self, node: &mut syn::Macro) {
        self.rewrite_path(&mut node.path);
        syn::visit_mut::visit_macro_mut(self, node);
    }
}

fn build_use_tree(head: &str, tail: &[String]) -> syn::UseTree {
    let ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    if tail.is_empty() {
        return syn::UseTree::Name(syn::UseName { ident });
    }
    let next = build_use_tree(&tail[0], &tail[1..]);
    syn::UseTree::Path(syn::UsePath { ident, colon2_token: Default::default(), tree: Box::new(next) })
}
