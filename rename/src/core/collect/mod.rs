use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use crate::alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};
use crate::fs;
use crate::occurrence::EnhancedOccurrenceVisitor;

use super::paths::module_path_for_file;
use super::symbol_id::normalize_symbol_id;
use crate::model::types::{AliasGraphReport, LineColumn, SpanRange, SymbolIndex, SymbolIndexReport, SymbolRecord};
use super::use_map::build_use_map;

mod collector;

use collector::SymbolCollector;

pub fn collect_names(project: &Path) -> Result<SymbolIndexReport> {
    // Run on a thread with 64 MB stack — syn::visit expression traversal is deeply recursive
    // and overflows the default 8 MB OS stack on large codebases.
    let project = project.to_path_buf();
    let result = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || collect_names_inner(&project))
        .expect("failed to spawn collector thread")
        .join()
        .expect("collector thread panicked");
    result
}

fn collect_names_inner(project: &Path) -> Result<SymbolIndexReport> {
    let files = fs::collect_rs_files(project)?;
    let mut symbols = Vec::new();
    let mut occurrences = Vec::new();
    let mut symbol_set: HashSet<String> = HashSet::new();

    // Global alias graph aggregation
    let mut global_alias_graph = AliasGraph::new();

    let mut symbol_table = SymbolIndex::default();
    for file in &files {
        let module_path = normalize_symbol_id(&module_path_for_file(project, file));
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content).with_context(|| format!("Failed to parse {}", file.display()))?;
        add_file_module_symbol(&module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);
        let file_alias_graph = collect_symbols(&ast, &module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);

        // Merge file alias graph into global graph
        for node in file_alias_graph.all_nodes() {
            global_alias_graph.add_use_node(node.clone());
        }
    }

    // Build edges after all nodes are collected
    global_alias_graph.build_edges();

    for file in &files {
        let module_path = normalize_symbol_id(&module_path_for_file(project, file));
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content).with_context(|| format!("Failed to parse {}", file.display()))?;
        let use_map = build_use_map(&ast, &module_path);
        let mut visitor = EnhancedOccurrenceVisitor::new(&module_path, file, &symbol_table, &use_map, &global_alias_graph, &mut occurrences);
        visitor.visit_file_items(&ast);
    }

    // Perform visibility leak analysis
    let symbol_visibility: HashMap<String, VisibilityScope> = symbol_table
        .symbols
        .iter()
        .map(|(id, entry): (&String, &_)| {
            // Extract visibility from attributes or default to public
            // This is simplified - full implementation would parse visibility from AST
            let vis = if entry.kind == "pub use" { VisibilityScope::Public } else { VisibilityScope::Private };
            (id.clone(), vis)
        })
        .collect();

    let visibility_analysis = global_alias_graph.analyze_visibility_leaks(&symbol_visibility);

    // Create alias graph report
    let use_nodes: Vec<ImportNode> = global_alias_graph.all_nodes().into_iter().cloned().collect();
    let edges = global_alias_graph.all_edges();
    let total_imports = use_nodes.iter().filter(|n| matches!(n.kind, UseKind::Simple | UseKind::Aliased)).count();
    let total_reexports = use_nodes.iter().filter(|n| matches!(n.kind, UseKind::ReExport | UseKind::ReExportAliased)).count();
    let glob_imports = use_nodes.iter().filter(|n| n.kind == UseKind::Glob).count();

    let alias_graph_report = AliasGraphReport { use_nodes, edge_count: edges.len(), total_imports, total_reexports, glob_imports };

    Ok(SymbolIndexReport { version: 1, symbols, occurrences, alias_graph: alias_graph_report, visibility_analysis: Some(visibility_analysis) })
}

pub fn emit_names(project: &Path, out: &Path) -> Result<()> {
    let report = collect_names(project)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

fn update_symbol_snapshot<F>(out: &mut Vec<SymbolRecord>, id: &str, mut update: F)
where F: FnMut(&mut SymbolRecord) {
    if let Some(entry) = out.iter_mut().find(|entry| entry.id == id) {
        update(entry);
    }
}

fn merge_symbol_metadata(target: &mut SymbolRecord, source: &SymbolRecord) {
    if target.kind != source.kind {
        return;
    }
    if target.kind == "module" {
        if source.declaration_file.is_some() {
            target.declaration_file = source.declaration_file.clone();
            target.file = source.file.clone();
            target.span = source.span.clone();
            target.doc_comments = source.doc_comments.clone();
            target.attributes = source.attributes.clone();
        }
        if source.definition_file.is_some() {
            target.definition_file = source.definition_file.clone();
        }
        if source.alias.is_some() {
            target.alias = source.alias.clone();
        }
    }
}

pub(crate) fn add_file_module_symbol(module_path: &str, file: &Path, symbol_table: &mut SymbolIndex, out: &mut Vec<SymbolRecord>, symbol_set: &mut HashSet<String>) {
    let module_id = normalize_symbol_id(module_path);
    let module_path = module_id.as_str();
    if module_path == "crate" {
        return;
    }

    let file_string = file.to_string_lossy().to_string();

    if let Some(existing) = symbol_table.symbols.get_mut(module_path) {
        existing.definition_file = Some(file_string.clone());
        update_symbol_snapshot(out, module_path, |entry| {
            entry.definition_file = Some(file_string.clone());
        });
        return;
    }
    if symbol_set.contains(module_path) {
        return;
    }
    let name = module_path.split("::").last().unwrap_or(module_path).to_string();
    let entry = SymbolRecord {
        id: module_path.to_string(),
        kind: "module".to_string(),
        name,
        module: module_path.to_string(),
        file: file_string.clone(),
        declaration_file: None,
        definition_file: Some(file_string),
        span: stub_range(),
        alias: None,
        doc_comments: Vec::new(),
        attributes: Vec::new(),
    };
    symbol_set.insert(entry.id.clone());
    symbol_table.symbols.insert(entry.id.clone(), entry.clone());
    out.push(entry);
}

fn stub_range() -> SpanRange {
    SpanRange { start: LineColumn { line: 1, column: 1 }, end: LineColumn { line: 1, column: 1 } }
}

pub(crate) fn collect_symbols(ast: &syn::File, module_path: &str, file: &Path, symbol_table: &mut SymbolIndex, out: &mut Vec<SymbolRecord>, symbol_set: &mut HashSet<String>) -> AliasGraph {
    let module_id = normalize_symbol_id(module_path);
    let module_path = module_id.as_str();
    let mut alias_graph = AliasGraph::default();
    let mut collector = SymbolCollector::new(file, &mut alias_graph);
    collector.walk(ast, module_path);
    for sym in collector.into_symbols() {
        if let Some(existing) = symbol_table.symbols.get_mut(&sym.id) {
            merge_symbol_metadata(existing, &sym);
            let symbol_id = sym.id.clone();
            update_symbol_snapshot(out, &symbol_id, |entry| {
                merge_symbol_metadata(entry, &sym);
            });
        } else {
            symbol_set.insert(sym.id.clone());
            symbol_table.symbols.insert(sym.id.clone(), sym.clone());
            out.push(sym);
        }
    }
    alias_graph
}
