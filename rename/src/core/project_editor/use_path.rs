use super::utils::find_project_root;
use super::QueuedOp;
use crate::alias::AliasGraph;
use crate::core::collect::{add_file_module_symbol, collect_symbols};
use crate::core::paths::module_path_for_file;
use crate::core::symbol_id::normalize_symbol_id;
use crate::model::types::SymbolIndex;
use crate::state::NodeRegistry;
use crate::structured::use_tree::UsePathRewritePass;
use crate::structured::{StructuredEditOptions, StructuredPass};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

pub(super) fn run_use_path_rewrite(registry: &mut NodeRegistry, changesets: &HashMap<PathBuf, Vec<QueuedOp>>) -> Result<HashSet<PathBuf>> {
    let updates = collect_use_path_updates(changesets);
    if updates.is_empty() {
        return Ok(HashSet::new());
    }
    let project_root = find_project_root(registry)?.unwrap_or_else(|| PathBuf::from("."));
    let mut symbol_table = SymbolIndex::default();
    let mut symbols = Vec::new();
    let mut symbol_set: HashSet<String> = HashSet::new();
    let mut alias_graph = AliasGraph::new();
    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(&project_root, file));
        add_file_module_symbol(&module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);
        let file_alias_graph = collect_symbols(ast, &module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);
        for node in file_alias_graph.all_nodes() {
            alias_graph.add_use_node(node.clone());
        }
    }
    alias_graph.build_edges();
    let mut touched = HashSet::new();
    let config = StructuredEditOptions::new(false, false, true);
    let alias_graph = std::sync::Arc::new(alias_graph);
    let symbol_table = std::sync::Arc::new(symbol_table);
    for (file, ast) in registry.asts.iter_mut() {
        let alias_nodes = alias_graph.nodes_in_file(&file.to_string_lossy()).into_iter().cloned().collect::<Vec<_>>();
        let resolver = crate::resolve::ResolverContext { module_path: module_path_for_file(&project_root, file), alias_graph: alias_graph.clone(), symbol_table: symbol_table.clone() };
        let mut pass = UsePathRewritePass::new(updates.clone(), alias_nodes, config.clone(), resolver);
        if pass.execute(file, "", ast)? {
            touched.insert(file.clone());
        }
    }
    Ok(touched)
}

fn collect_use_path_updates(changesets: &HashMap<PathBuf, Vec<QueuedOp>>) -> HashMap<String, String> {
    let mut updates = HashMap::new();
    for ops in changesets.values() {
        for queued in ops {
            match &queued.op {
                crate::structured::NodeOp::MutateField { mutation, .. } => {
                    if let crate::structured::FieldMutation::RenameIdent(new_name) = mutation {
                        let old_id = normalize_symbol_id(&queued.symbol_id);
                        if let Some(new_id) = replace_last_segment(&old_id, new_name) {
                            updates.insert(old_id, new_id);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    updates
}

fn replace_last_segment(path: &str, new_name: &str) -> Option<String> {
    let mut parts: Vec<&str> = path.split("::").collect();
    if parts.is_empty() {
        return None;
    }
    *parts.last_mut().unwrap() = new_name;
    Some(parts.join("::"))
}
