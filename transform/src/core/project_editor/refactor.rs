use crate::core::paths::module_path_for_file;
use crate::core::project_editor::propagate::build_symbol_index_and_occurrences;
use crate::core::symbol_id::normalize_symbol_id;
use crate::core::use_map::{normalize_use_prefix, path_to_string};
use crate::resolve::Resolver;
use crate::state::NodeRegistry;
use anyhow::Result;
use quote::ToTokens;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use syn::visit::{self, Visit};
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
    let project_root = registry.asts.keys().next().and_then(|f| f.parent()).map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));

    let mut touched = HashSet::new();
    for (file, ast) in registry.asts.iter_mut() {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        let rewrite_map = build_moveset_rewrite_map(moveset);
        let mut rewriter = CanonicalRewriteVisitor { module_path, rewrite_map, changed: false };
        rewriter.visit_file_mut(ast);
        if rewriter.changed {
            let rendered = crate::structured::render_file(ast);
            registry.sources.insert(file.clone(), std::sync::Arc::new(rendered));
            touched.insert(file.clone());
        }
    }
    Ok(touched)
}

struct CanonicalRewriteVisitor {
    module_path: String,
    rewrite_map: HashMap<String, (String, String)>,
    changed: bool,
}

impl CanonicalRewriteVisitor {
    fn rewrite_path(&mut self, path: &mut syn::Path) {
        let full = path_to_string(path, &self.module_path);
        let Some((new_module, name)) = self.rewrite_map.get(&full) else { return };
        let new_path_str = format!("{new_module}::{name}");
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
                let full = normalize_use_prefix(&full, &self.module_path).join("::");
                if let Some((new_module, name)) = self.rewrite_map.get(&full) {
                    let mut new_segments: Vec<String> = new_module.split("::").map(|s| s.to_string()).collect();
                    new_segments.push(name.to_string());
                    if let Some((first, rest)) = new_segments.split_first() {
                        *tree = build_use_tree(first, rest);
                        self.changed = true;
                    }
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut full = prefix.to_vec();
                full.push(rename.ident.to_string());
                let full = normalize_use_prefix(&full, &self.module_path).join("::");
                if let Some((new_module, name)) = self.rewrite_map.get(&full) {
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

impl VisitMut for CanonicalRewriteVisitor {
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

fn build_moveset_rewrite_map(moveset: &MoveSet) -> HashMap<String, (String, String)> {
    let mut map = HashMap::new();
    for (symbol_id, (old_module, new_module)) in &moveset.entries {
        let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
        let old_full = format!("{old_module}::{name}");
        map.insert(old_full, (new_module.clone(), name.to_string()));
    }
    map
}

pub(crate) fn run_pass2_scope_rehydration(registry: &mut NodeRegistry, moveset: &MoveSet) -> Result<HashSet<PathBuf>> {
    if moveset.entries.is_empty() {
        return Ok(HashSet::new());
    }
    let (symbol_table, _occurrences, alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let project_root = registry.asts.keys().next().and_then(|f| f.parent()).map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));

    let dest_modules: HashSet<String> = moveset.entries.values().map(|(_, new)| new.clone()).collect();
    let mut touched = HashSet::new();

    for (file, ast) in registry.asts.iter_mut() {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        if !dest_modules.contains(&module_path) {
            continue;
        }
        let resolver = Resolver::new(&module_path, &alias_graph, &symbol_table);
        let mut collector = ReferenceCollector::new(resolver);
        collector.visit_file(ast);
        let refs = collector.refs;

        if refs.is_empty() {
            continue;
        }

        let file_str = file.to_string_lossy().to_string();
        let mut existing_uses: HashSet<String> = ast
            .items
            .iter()
            .filter_map(|i| match i {
                syn::Item::Use(u) => Some(u.to_token_stream().to_string()),
                _ => None,
            })
            .collect();

        for symbol_id in refs {
            let (sym_mod, sym_name) = match split_module_and_name(&symbol_id) {
                Some(parts) => parts,
                None => continue,
            };
            if sym_mod == module_path {
                continue;
            }
            if is_in_scope(&alias_graph, &module_path, &file_str, sym_name, &symbol_id) {
                continue;
            }
            let use_str = format!("use {symbol_id};");
            if let Ok(parsed) = syn::parse_str::<syn::ItemUse>(&use_str) {
                let rendered = parsed.to_token_stream().to_string();
                if existing_uses.insert(rendered) {
                    ast.items.insert(0, syn::Item::Use(parsed));
                    touched.insert(file.clone());
                }
            }
        }
        if touched.contains(file) {
            let rendered = crate::structured::render_file(ast);
            registry.sources.insert(file.clone(), std::sync::Arc::new(rendered));
        }
    }

    Ok(touched)
}

pub(crate) fn run_pass3_orphan_cleanup(registry: &mut NodeRegistry) -> Result<HashSet<PathBuf>> {
    let (symbol_table, occurrences, alias_graph) = build_symbol_index_and_occurrences(registry)?;
    let mut occ_by_file: HashMap<String, Vec<crate::model::types::SymbolOccurrence>> = HashMap::new();
    for occ in occurrences {
        occ_by_file.entry(occ.file.clone()).or_default().push(occ);
    }
    let project_root = registry.asts.keys().next().and_then(|f| f.parent()).map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));

    let mut touched = HashSet::new();
    for (file, ast) in registry.asts.iter_mut() {
        let file_str = file.to_string_lossy().to_string();
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        let occs = occ_by_file.get(&file_str);
        let mut keep_items = Vec::with_capacity(ast.items.len());
        for item in ast.items.drain(..) {
            let syn::Item::Use(use_item) = &item else {
                keep_items.push(item);
                continue;
            };
            if !is_private_use(use_item) {
                keep_items.push(item);
                continue;
            }
            let uses = collect_use_imports(use_item, &module_path);
            let mut used = false;
            for u in &uses {
                // If we can't resolve this import to a known in-crate symbol,
                // keep it. This avoids dropping std/external imports and any
                // paths outside the symbol table.
                let is_in_crate = u.source_path.starts_with("crate::");
                if !is_in_crate || !symbol_table.symbols.contains_key(&u.source_path) {
                    used = true;
                    break;
                }
                if u.is_glob {
                    let prefix = format!("{}::", u.source_path);
                    if occs.map(|o| o.iter().any(|o| o.kind != "use" && o.kind != "use_path" && o.id.starts_with(&prefix))).unwrap_or(false) {
                        used = true;
                        break;
                    }
                } else {
                    if occs.map(|o| o.iter().any(|o| o.kind != "use" && o.kind != "use_path" && o.id == u.source_path)).unwrap_or(false) {
                        used = true;
                        break;
                    }
                }
            }
            if used {
                keep_items.push(item);
            } else {
                touched.insert(file.clone());
            }
        }
        ast.items = keep_items;
        if touched.contains(file) {
            let rendered = crate::structured::render_file(ast);
            registry.sources.insert(file.clone(), std::sync::Arc::new(rendered));
        }
    }
    let _ = alias_graph; // keep for future expansion
    Ok(touched)
}

struct ReferenceCollector<'a> {
    resolver: Resolver<'a>,
    refs: HashSet<String>,
}

impl<'a> ReferenceCollector<'a> {
    fn new(resolver: Resolver<'a>) -> Self {
        Self { resolver, refs: HashSet::new() }
    }

    fn consider_path(&mut self, path: &syn::Path) {
        if path.leading_colon.is_some() {
            return;
        }
        let first = match path.segments.first() {
            Some(s) => s,
            None => return,
        };
        let head = first.ident.to_string();
        if matches!(head.as_str(), "crate" | "self" | "super") {
            return;
        }
        if path.segments.len() != 1 {
            return;
        }
        if let Some(resolved) = self.resolver.resolve_path_segments(&[head]) {
            self.refs.insert(resolved);
        }
    }
}

impl<'a> Visit<'a> for ReferenceCollector<'a> {
    fn visit_path(&mut self, node: &'a syn::Path) {
        self.consider_path(node);
        visit::visit_path(self, node);
    }

    fn visit_macro(&mut self, node: &'a syn::Macro) {
        self.consider_path(&node.path);
        visit::visit_macro(self, node);
    }
}

struct UseImport {
    source_path: String,
    is_glob: bool,
}

fn collect_use_imports(item: &syn::ItemUse, module_path: &str) -> Vec<UseImport> {
    let mut out = Vec::new();
    let mut prefix: Vec<String> = Vec::new();
    if item.leading_colon.is_some() {
        prefix.push("crate".to_string());
    }
    collect_use_tree_imports(&item.tree, &mut prefix, module_path, &mut out);
    out
}

fn collect_use_tree_imports(tree: &syn::UseTree, prefix: &mut Vec<String>, module_path: &str, out: &mut Vec<UseImport>) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            collect_use_tree_imports(&p.tree, prefix, module_path, out);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(name.ident.to_string());
            out.push(UseImport { source_path: full.join("::"), is_glob: false });
        }
        syn::UseTree::Rename(rename) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(rename.ident.to_string());
            out.push(UseImport { source_path: full.join("::"), is_glob: false });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree_imports(item, prefix, module_path, out);
            }
        }
        syn::UseTree::Glob(_) => {
            let full = normalize_use_prefix(prefix, module_path).join("::");
            out.push(UseImport { source_path: full, is_glob: true });
        }
    }
}

fn is_private_use(item: &syn::ItemUse) -> bool {
    match &item.vis {
        syn::Visibility::Public(_) => false,
        syn::Visibility::Restricted(restricted) => restricted.path.is_ident("crate"),
        _ => true,
    }
}

fn is_in_scope(alias_graph: &crate::alias::AliasGraph, module_path: &str, file: &str, local_name: &str, symbol_id: &str) -> bool {
    if let Some(resolved) = alias_graph.resolve_local(module_path, local_name) {
        if resolved == symbol_id {
            return true;
        }
        return true; // local name already taken
    }
    if alias_graph.get_importers(symbol_id).iter().any(|n| n.file == file) {
        return true;
    }
    let prefix = format!("{}::", extract_module(symbol_id));
    for glob in alias_graph.get_glob_imports(module_path) {
        if prefix.starts_with(&format!("{}::", glob.source_path)) {
            return true;
        }
    }
    false
}

fn extract_module(path: &str) -> String {
    path.rsplit_once("::").map(|(m, _)| m.to_string()).unwrap_or_else(|| "crate".to_string())
}

fn split_module_and_name(path: &str) -> Option<(&str, &str)> {
    path.rsplit_once("::")
}
