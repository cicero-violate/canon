//! semantic: domain=tooling

use anyhow::{Context, Result, bail};
use proc_macro2::Span;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::visit::{self, Visit};

use crate::fs;

use super::alias::VisibilityLeakAnalysis;
use super::alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};
use super::occurrence::EnhancedOccurrenceVisitor;
use super::rewrite::{RewriteBufferSet, SourceTextEdit};
use super::structured::{
    StructuredAttributeResult, StructuredEditConfig, rewrite_doc_and_attr_literals,
    structured_edit_config,
};

/// D1: Result of flushing buffers with detailed file information
///
/// Provides diagnostics about what was actually written to disk:
/// - touched_files: List of files that were modified
/// - total_edits: Total number of text edits applied
///
/// Use this to:
/// - Report what changed in structured editing pipelines
/// - Verify expected number of modifications
/// - Debug buffer flush operations
#[derive(Debug)]
pub struct RewriteSummary {
    pub touched_files: Vec<PathBuf>,
    pub total_edits: usize,
}

impl RewriteSummary {
    pub fn is_empty(&self) -> bool {
        self.touched_files.is_empty()
    }

    pub fn file_count(&self) -> usize {
        self.touched_files.len()
    }
}

/// B2: Structured files tracking for pipeline coordination
///
/// # D2: Enhanced with per-pass file tracking
///
/// Tracks which files were touched by each structured editing pass:
/// - doc_files: Files modified by doc comment literal rewrites
/// - attr_files: Files modified by attribute literal rewrites  
/// - use_files: Files modified by use statement synthesis
///
/// This enables:
/// - Detailed diagnostics showing what changed where
/// - Dry-run previews with pass-specific breakdowns
/// - Debugging structured editing pipelines
#[derive(Default)]
pub struct StructuredEditTracker {
    files: HashSet<String>,
    pub(crate) doc_files: HashSet<String>,
    pub(crate) attr_files: HashSet<String>,
    pub(crate) use_files: HashSet<String>,
}

impl StructuredEditTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_doc_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.doc_files.insert(file);
    }

    pub fn mark_attr_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.attr_files.insert(file);
    }

    pub fn mark_use_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.use_files.insert(file);
    }

    pub fn mark_generic(&mut self, file: String) {
        self.files.insert(file);
    }

    pub fn all_files(&self) -> &HashSet<String> {
        &self.files
    }

    pub fn into_set(self) -> HashSet<String> {
        self.files
    }

    /// D2: Get files touched by doc literal rewrites
    pub fn doc_files(&self) -> &HashSet<String> {
        &self.doc_files
    }

    /// D2: Get files touched by attr literal rewrites
    pub fn attr_files(&self) -> &HashSet<String> {
        &self.attr_files
    }

    /// D2: Get files touched by use statement rewrites
    pub fn use_files(&self) -> &HashSet<String> {
        &self.use_files
    }

    pub fn summary(&self, config: &StructuredEditConfig) -> String {
        let mut parts = Vec::new();
        if config.doc_literals_enabled() && !self.doc_files.is_empty() {
            parts.push(format!("docs:{}", self.doc_files.len()));
        }
        if config.attr_literals_enabled() && !self.attr_files.is_empty() {
            parts.push(format!("attrs:{}", self.attr_files.len()));
        }
        if config.use_statements_enabled() && !self.use_files.is_empty() {
            parts.push(format!("uses:{}", self.use_files.len()));
        }
        if parts.is_empty() {
            format!("{} files via structured edits", self.files.len())
        } else {
            format!(
                "{} files via structured edits ({})",
                self.files.len(),
                parts.join(", ")
            )
        }
    }
}

#[derive(Serialize)]
pub struct SymbolIndexReport {
    pub version: i64,
    pub symbols: Vec<SymbolRecord>,
    pub occurrences: Vec<SymbolOccurrence>,
    pub alias_graph: AliasGraphReport,
    pub visibility_analysis: Option<VisibilityLeakAnalysis>,
}

#[derive(Serialize)]
pub struct AliasGraphReport {
    pub use_nodes: Vec<ImportNode>,
    pub edge_count: usize,
    pub total_imports: usize,
    pub total_reexports: usize,
    pub glob_imports: usize,
}

#[derive(Serialize, Clone)]
pub struct SymbolRecord {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub module: String,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declaration_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition_file: Option<String>,
    pub span: SpanRange,
    pub alias: Option<String>,
    pub doc_comments: Vec<String>,
    pub attributes: Vec<String>,
}

#[derive(Serialize)]
pub struct SymbolOccurrence {
    pub id: String,
    pub file: String,
    pub kind: String,
    pub span: SpanRange,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}

pub fn run_names(args: &[String]) -> Result<()> {
    let mut project = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => {
                project = Some(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out = Some(&args[i + 1]);
                i += 2;
            }
            _ => i += 1,
        }
    }
    let project = project.context("--project required")?;
    let out = out.map_or(".semantic-lint/names.json", |v| v);
    emit_names(Path::new(project), Path::new(out))
}

pub fn run_rename(args: &[String]) -> Result<()> {
    let mut project = None;
    let mut map_path = None;
    let mut out_path = None;
    let mut dry_run = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" => {
                project = Some(&args[i + 1]);
                i += 2;
            }
            "--map" => {
                map_path = Some(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out_path = Some(&args[i + 1]);
                i += 2;
            }
            "--dry-run" => {
                dry_run = true;
                i += 1;
            }
            _ => i += 1,
        }
    }
    let project = project.context("--project required")?;
    let map_path = map_path.context("--map required")?;
    apply_rename(
        Path::new(project),
        Path::new(map_path),
        dry_run,
        out_path.map(Path::new),
    )
}

pub fn collect_names(project: &Path) -> Result<SymbolIndexReport> {
    let files = fs::collect_rs_files(project)?;
    let mut symbols = Vec::new();
    let mut occurrences = Vec::new();
    let mut symbol_set: HashSet<String> = HashSet::new();

    // Global alias graph aggregation
    let mut global_alias_graph = AliasGraph::new();

    let mut symbol_table = SymbolIndex::default();
    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let file_alias_graph = collect_symbols(
            &ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );

        // Merge file alias graph into global graph
        for node in file_alias_graph.all_nodes() {
            global_alias_graph.add_use_node(node.clone());
        }
    }

    // Build edges after all nodes are collected
    global_alias_graph.build_edges();

    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        let use_map = build_use_map(&ast, &module_path);
        let mut visitor = EnhancedOccurrenceVisitor::new(
            &module_path,
            file,
            &symbol_table,
            &use_map,
            &mut occurrences,
        );
        visitor.visit_file(&ast);
    }

    // Perform visibility leak analysis
    let symbol_visibility: HashMap<String, VisibilityScope> = symbol_table
        .symbols
        .iter()
        .map(|(id, entry): (&String, &_)| {
            // Extract visibility from attributes or default to public
            // This is simplified - full implementation would parse visibility from AST
            let vis = if entry.kind == "pub use" {
                VisibilityScope::Public
            } else {
                VisibilityScope::Private
            };
            (id.clone(), vis)
        })
        .collect();

    let visibility_analysis = global_alias_graph.analyze_visibility_leaks(&symbol_visibility);

    // Create alias graph report
    let use_nodes: Vec<ImportNode> = global_alias_graph
        .all_nodes()
        .into_iter()
        .cloned()
        .collect();
    let edges = global_alias_graph.all_edges();
    let total_imports = use_nodes
        .iter()
        .filter(|n| matches!(n.kind, UseKind::Simple | UseKind::Aliased))
        .count();
    let total_reexports = use_nodes
        .iter()
        .filter(|n| matches!(n.kind, UseKind::ReExport | UseKind::ReExportAliased))
        .count();
    let glob_imports = use_nodes.iter().filter(|n| n.kind == UseKind::Glob).count();

    let alias_graph_report = AliasGraphReport {
        use_nodes,
        edge_count: edges.len(),
        total_imports,
        total_reexports,
        glob_imports,
    };

    Ok(SymbolIndexReport {
        version: 1,
        symbols,
        occurrences,
        alias_graph: alias_graph_report,
        visibility_analysis: Some(visibility_analysis),
    })
}

pub fn emit_names(project: &Path, out: &Path) -> Result<()> {
    let report = collect_names(project)?;
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

pub fn apply_rename_with_map(
    project: &Path,
    mapping: &HashMap<String, String>,
    dry_run: bool,
    out_path: Option<&Path>,
) -> Result<()> {
    if mapping.is_empty() {
        println!("No renames applied (empty map).");
        return Ok(());
    }
    for (id, name) in mapping {
        if !is_valid_ident(name) {
            bail!("Invalid identifier for {}: {}", id, name);
        }
    }

    let files = fs::collect_rs_files(project)?;
    let mut symbol_table = SymbolIndex::default();
    let mut symbol_set: HashSet<String> = HashSet::new();
    let mut symbols = Vec::new();

    // Build global alias graph for rename operations
    let mut global_alias_graph = AliasGraph::new();

    // B2: Initialize structured edit configuration and tracker
    let structured_config = structured_edit_config();
    let structured_mode = structured_config.is_enabled();
    let mut structured_tracker = StructuredEditTracker::new();
    let mut rewrite_buffers = RewriteBufferSet::new();

    // B2: Report active configuration
    if structured_mode {
        eprintln!(
            "Structured editing enabled: {}",
            structured_config.summary()
        );
    }

    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let file_alias_graph = collect_symbols(
            &ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );

        // Merge into global alias graph
        for node in file_alias_graph.all_nodes() {
            global_alias_graph.add_use_node(node.clone());
        }
    }

    // Build edges for rename resolution
    global_alias_graph.build_edges();

    let file_renames = plan_file_renames(&symbol_table, &mapping)?;

    let mut all_edits: Vec<SymbolEdit> = Vec::new();
    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        let use_map = build_use_map(&ast, &module_path);

        // B2: Execute doc/attr pass if enabled
        let mut structured_attr = StructuredAttributeResult::new();
        if structured_config.doc_or_attr_enabled() {
            structured_attr = rewrite_doc_and_attr_literals(
                file,
                &content,
                &ast,
                &mapping,
                &structured_config,
                &mut rewrite_buffers,
            )?;
            if structured_attr.changed {
                let file_str = file.to_string_lossy().to_string();
                // Track which specific pass made changes
                if structured_config.doc_literals_enabled() {
                    structured_tracker.mark_doc_edit(file_str.clone());
                }
                if structured_config.attr_literals_enabled() {
                    structured_tracker.mark_attr_edit(file_str);
                }
            }
        }

        let mut occurrences = Vec::new();
        let mut visitor = EnhancedOccurrenceVisitor::new(
            &module_path,
            file,
            &symbol_table,
            &use_map,
            &mut occurrences,
        );
        visitor.visit_file(&ast);

        // Add symbol definitions as occurrences so they get renamed
        for (symbol_id, symbol_entry) in &symbol_table.symbols {
            if symbol_entry.file == file.to_string_lossy().to_string() {
                // Only add symbols defined in this file
                if mapping.contains_key(symbol_id.as_str()) {
                    occurrences.push(SymbolOccurrence {
                        id: symbol_id.clone(),
                        file: symbol_entry.file.clone(),
                        kind: format!("{}_definition", symbol_entry.kind),
                        span: symbol_entry.span.clone(),
                    });
                }
            }
        }

        let mut edits = Vec::new();
        for occ in occurrences {
            if structured_config.doc_or_attr_enabled()
                && occ.kind == "attribute"
                && structured_attr.should_skip(&occ.span)
            {
                continue;
            }
            if let Some(new_name) = mapping.get(&occ.id) {
                edits.push(SymbolEdit {
                    id: occ.id.clone(),
                    file: occ.file.clone(),
                    kind: occ.kind.clone(),
                    start: occ.span.start.clone(),
                    end: occ.span.end.clone(),
                    new_name: new_name.clone(),
                });
            }
        }

        if !dry_run && !edits.is_empty() {
            queue_file_edits(&mut rewrite_buffers, file, &content, &edits)?;
        }
        all_edits.extend(edits);
    }

    // Collect alias renames
    let alias_edits = collect_and_rename_aliases(project, &symbol_table, &mapping)?;
    if !dry_run && !alias_edits.is_empty() {
        queue_alias_edits(&mut rewrite_buffers, &alias_edits)?;
    }
    all_edits.extend(alias_edits);

    if dry_run {
        let out = out_path.unwrap_or_else(|| Path::new(".semantic-lint/rename_preview.json"));
        write_preview(
            out,
            &all_edits,
            &file_renames,
            &structured_tracker,
            &structured_config,
        )?;
        return Ok(());
    }

    flush_and_format(&mut rewrite_buffers)?;

    for rename in &file_renames {
        if rename.from == rename.to {
            continue;
        }
        if Path::new(&rename.to).exists() {
            bail!("File already exists: {}", rename.to);
        }
        // Create parent directories if doing a directory move
        if rename.is_directory_move {
            if let Some(parent) = Path::new(&rename.to).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::rename(&rename.from, &rename.to)?;
    }

    // Update mod declarations after moving files
    update_mod_declarations(project, &symbol_table, &file_renames)?;

    // B2: Update use statement paths after moving files (track structured use edits)
    update_use_paths(
        project,
        &file_renames,
        &structured_config,
        &mut rewrite_buffers,
        &global_alias_graph,
        &mut structured_tracker,
    )?;

    // B2: Report structured edit summary
    if structured_mode && !structured_tracker.all_files().is_empty() {
        eprintln!("{}", structured_tracker.summary(&structured_config));
    }

    if structured_mode {
        flush_and_format(&mut rewrite_buffers)?;
    }

    let mut edited_files = HashSet::new();
    for edit in &all_edits {
        edited_files.insert(edit.file.clone());
    }
    if file_renames.is_empty() {
        println!(
            "Renamed {} occurrences across {} files.",
            all_edits.len(),
            edited_files.len()
        );
    } else {
        println!(
            "Renamed {} occurrences across {} files ({} file renames).",
            all_edits.len(),
            edited_files.len(),
            file_renames.len()
        );
    }

    Ok(())
}

#[derive(Default)]
pub struct SymbolIndex {
    pub symbols: HashMap<String, SymbolRecord>,
}

#[derive(Clone, Serialize)]
struct SymbolEdit {
    id: String,
    file: String,
    kind: String,
    start: LineColumn,
    end: LineColumn,
    new_name: String,
}

#[derive(Clone, Serialize)]
struct FileRename {
    from: String,
    to: String,
    is_directory_move: bool,
    old_module_id: String,
    new_module_id: String,
}

fn update_symbol_snapshot<F>(out: &mut Vec<SymbolRecord>, id: &str, mut update: F)
where
    F: FnMut(&mut SymbolRecord),
{
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

fn add_file_module_symbol(
    module_path: &str,
    file: &Path,
    symbol_table: &mut SymbolIndex,
    out: &mut Vec<SymbolRecord>,
    symbol_set: &mut HashSet<String>,
) {
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
    let name = module_path
        .split("::")
        .last()
        .unwrap_or(module_path)
        .to_string();
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
    SpanRange {
        start: LineColumn { line: 1, column: 1 },
        end: LineColumn { line: 1, column: 1 },
    }
}

fn collect_symbols(
    ast: &syn::File,
    module_path: &str,
    file: &Path,
    symbol_table: &mut SymbolIndex,
    out: &mut Vec<SymbolRecord>,
    symbol_set: &mut HashSet<String>,
) -> AliasGraph {
    let mut alias_graph = AliasGraph::default();
    let mut collector = SymbolCollector::new(module_path, file, &mut alias_graph);
    collector.visit_file(ast);
    for sym in collector.symbols {
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

struct SymbolCollector<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbols: Vec<SymbolRecord>,
    current_impl: Option<ImplContext>,
    alias_graph: &'a mut AliasGraph,
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}

impl<'a> SymbolCollector<'a> {
    fn new(module_path: &'a str, file: &'a Path, alias_graph: &'a mut AliasGraph) -> Self {
        Self {
            module_path,
            file,
            symbols: Vec::new(),
            current_impl: None,
            alias_graph,
        }
    }

    fn add_symbol(
        &mut self,
        id: String,
        kind: &str,
        name: &str,
        span: Span,
        docs: Vec<String>,
        attrs: Vec<String>,
    ) {
        let file_path = self.file.to_string_lossy().to_string();
        self.symbols.push(SymbolRecord {
            id,
            kind: kind.to_string(),
            name: name.to_string(),
            module: self.module_path.to_string(),
            file: file_path.clone(),
            declaration_file: None,
            definition_file: Some(file_path),
            span: span_to_range(span),
            alias: None,
            doc_comments: docs,
            attributes: attrs,
        });
    }

    fn extract_docs_and_attrs(attrs: &[syn::Attribute]) -> (Vec<String>, Vec<String>) {
        let mut docs = Vec::new();
        let mut attributes = Vec::new();

        for attr in attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            docs.push(lit_str.value());
                        }
                    }
                }
            } else {
                attributes.push(quote::quote!(#attr).to_string());
            }
        }

        (docs, attributes)
    }

    fn collect_use_tree_root(
        &mut self,
        tree: &syn::UseTree,
        docs: &[String],
        attrs: &[String],
        visibility: VisibilityScope,
    ) {
        self.collect_use_tree_with_prefix(tree, String::new(), docs, attrs, visibility);
    }

    fn collect_use_tree_with_prefix(
        &mut self,
        tree: &syn::UseTree,
        prefix: String,
        docs: &[String],
        attrs: &[String],
        visibility: VisibilityScope,
    ) {
        match tree {
            syn::UseTree::Path(path) => {
                let new_prefix = if prefix.is_empty() {
                    path.ident.to_string()
                } else {
                    format!("{}::{}", prefix, path.ident)
                };
                self.collect_use_tree_with_prefix(&path.tree, new_prefix, docs, attrs, visibility);
            }
            syn::UseTree::Name(name) => {
                // Simple use: use foo::Bar;
                let source_path = if prefix.is_empty() {
                    name.ident.to_string()
                } else {
                    format!("{}::{}", prefix, name.ident)
                };
                let local_name = name.ident.to_string();
                let id = format!("{}::use::{}", self.module_path, local_name);

                self.add_symbol(
                    id.clone(),
                    "use",
                    &local_name,
                    name.ident.span(),
                    docs.to_vec(),
                    attrs.to_vec(),
                );

                // Determine use kind based on visibility
                let kind = if matches!(visibility, VisibilityScope::Public) {
                    UseKind::ReExport
                } else {
                    UseKind::Simple
                };

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name,
                    original_name: None,
                    kind,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Rename(rename) => {
                // Renamed use: use foo::Bar as Baz;
                let source_path = if prefix.is_empty() {
                    rename.ident.to_string()
                } else {
                    format!("{}::{}", prefix, rename.ident)
                };
                let original_name = rename.ident.to_string();
                let local_name = rename.rename.to_string();
                let id = format!("{}::use::{}", self.module_path, local_name);

                self.add_symbol(
                    id.clone(),
                    "use_as",
                    &local_name,
                    rename.rename.span(),
                    docs.to_vec(),
                    attrs.to_vec(),
                );

                // Determine use kind based on visibility
                let kind = if matches!(visibility, VisibilityScope::Public) {
                    UseKind::ReExportAliased
                } else {
                    UseKind::Aliased
                };

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name,
                    original_name: Some(original_name),
                    kind,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Glob(glob) => {
                // Glob import: use foo::*;
                let source_path = prefix;
                let id = format!(
                    "{}::use::*::{}",
                    self.module_path,
                    source_path.replace("::", "_")
                );

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name: "*".to_string(),
                    original_name: None,
                    kind: UseKind::Glob,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Group(group) => {
                // Group: use foo::{Bar, Baz};
                for item in &group.items {
                    self.collect_use_tree_with_prefix(
                        item,
                        prefix.clone(),
                        docs,
                        attrs,
                        visibility.clone(),
                    );
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for SymbolCollector<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let mod_id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        let file_path = self.file.to_string_lossy().to_string();
        let is_inline = i.content.is_some();
        self.symbols.push(SymbolRecord {
            id: mod_id,
            kind: "module".to_string(),
            name: i.ident.to_string(),
            module: self.module_path.to_string(),
            file: file_path.clone(),
            declaration_file: Some(file_path.clone()),
            definition_file: if is_inline {
                Some(file_path.clone())
            } else {
                None
            },
            span: span_to_range(i.ident.span()),
            alias: None,
            doc_comments: docs,
            attributes: attrs,
        });
        visit::visit_item_mod(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "struct",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for field in &i.fields {
            if let Some(ident) = &field.ident {
                let (field_docs, field_attrs) = Self::extract_docs_and_attrs(&field.attrs);
                let fid = format!("{}::{}", id, ident);
                self.add_symbol(
                    fid,
                    "field",
                    &ident.to_string(),
                    ident.span(),
                    field_docs,
                    field_attrs,
                );
            }
        }
        visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "enum",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for variant in &i.variants {
            let (var_docs, var_attrs) = Self::extract_docs_and_attrs(&variant.attrs);
            let vid = format!("{}::{}", id, variant.ident);
            self.add_symbol(
                vid.clone(),
                "variant",
                &variant.ident.to_string(),
                variant.ident.span(),
                var_docs,
                var_attrs,
            );
            for field in &variant.fields {
                if let Some(ident) = &field.ident {
                    let (field_docs, field_attrs) = Self::extract_docs_and_attrs(&field.attrs);
                    let fid = format!("{}::{}", vid, ident);
                    self.add_symbol(
                        fid,
                        "field",
                        &ident.to_string(),
                        ident.span(),
                        field_docs,
                        field_attrs,
                    );
                }
            }
        }
        visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "trait",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for item in &i.items {
            match item {
                syn::TraitItem::Fn(method) => {
                    let (m_docs, m_attrs) = Self::extract_docs_and_attrs(&method.attrs);
                    let mid = format!("{}::{}", id, method.sig.ident);
                    self.add_symbol(
                        mid,
                        "trait_method",
                        &method.sig.ident.to_string(),
                        method.sig.ident.span(),
                        m_docs,
                        m_attrs,
                    );
                }
                syn::TraitItem::Const(const_item) => {
                    let (c_docs, c_attrs) = Self::extract_docs_and_attrs(&const_item.attrs);
                    let cid = format!("{}::{}", id, const_item.ident);
                    self.add_symbol(
                        cid,
                        "trait_const",
                        &const_item.ident.to_string(),
                        const_item.ident.span(),
                        c_docs,
                        c_attrs,
                    );
                }
                syn::TraitItem::Type(type_item) => {
                    let (t_docs, t_attrs) = Self::extract_docs_and_attrs(&type_item.attrs);
                    let tid = format!("{}::{}", id, type_item.ident);
                    self.add_symbol(
                        tid,
                        "trait_type",
                        &type_item.ident.to_string(),
                        type_item.ident.span(),
                        t_docs,
                        t_attrs,
                    );
                }
                _ => {}
            }
        }
        visit::visit_item_trait(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let id = module_child_path(self.module_path, i.sig.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "function",
            &i.sig.ident.to_string(),
            i.sig.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_fn(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "type_alias",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "const",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_const(self, i);
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "static",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_static(self, i);
    }

    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "union",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for field in &i.fields.named {
            if let Some(ident) = &field.ident {
                let (field_docs, field_attrs) = Self::extract_docs_and_attrs(&field.attrs);
                let fid = format!("{}::{}", id, ident);
                self.add_symbol(
                    fid,
                    "field",
                    &ident.to_string(),
                    ident.span(),
                    field_docs,
                    field_attrs,
                );
            }
        }
        visit::visit_item_union(self, i);
    }

    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        let name = i
            .rename
            .as_ref()
            .map(|(_, ident)| ident.to_string())
            .unwrap_or_else(|| i.ident.to_string());
        let id = module_child_path(self.module_path, name.clone());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(id, "extern_crate", &name, i.ident.span(), docs, attrs);
        visit::visit_item_extern_crate(self, i);
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        let visibility = VisibilityScope::from(&i.vis);
        self.collect_use_tree_root(&i.tree, &docs, &attrs, visibility);
        visit::visit_item_use(self, i);
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        if let Some(ident) = &i.ident {
            let id = module_child_path(self.module_path, ident.to_string());
            let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
            // Distinguish between macro_rules! and macro invocations
            let kind = if i.mac.path.is_ident("macro_rules") {
                "macro_rules"
            } else {
                "macro"
            };
            self.add_symbol(id, kind, &ident.to_string(), ident.span(), docs, attrs);
        }
        visit::visit_item_macro(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let struct_path = type_path_string(&i.self_ty, self.module_path);
        let trait_path = i
            .trait_
            .as_ref()
            .map(|(_, path, _)| path_to_string(path, self.module_path));
        self.current_impl = Some(ImplContext {
            struct_path,
            trait_path,
        });
        visit::visit_item_impl(self, i);
        self.current_impl = None;
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if let Some(ctx) = &self.current_impl {
            let name = i.sig.ident.to_string();
            let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
            let id = if let Some(trait_path) = &ctx.trait_path {
                format!("{} as {}::{}", ctx.struct_path, trait_path, name)
            } else {
                format!("{}::{}", ctx.struct_path, name)
            };
            self.add_symbol(id, "method", &name, i.sig.ident.span(), docs, attrs);
        }
        visit::visit_impl_item_fn(self, i);
    }
}

struct SymbolOccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    current_impl: Option<ImplContext>,
    current_struct: Option<String>,
    type_context: LocalTypeContext,
}

impl<'a> SymbolOccurrenceVisitor<'a> {
    fn new(
        module_path: &'a str,
        file: &'a Path,
        symbol_table: &'a SymbolIndex,
        use_map: &'a HashMap<String, String>,
        occurrences: &'a mut Vec<SymbolOccurrence>,
    ) -> Self {
        Self {
            module_path,
            file,
            symbol_table,
            use_map,
            occurrences,
            current_impl: None,
            current_struct: None,
            type_context: LocalTypeContext::new(symbol_table),
        }
    }

    fn add_occurrence(&mut self, id: String, kind: &str, span: Span) {
        self.occurrences.push(SymbolOccurrence {
            id,
            file: self.file.to_string_lossy().to_string(),
            kind: kind.to_string(),
            span: span_to_range(span),
        });
    }

    fn infer_receiver_type(&self, receiver: &syn::Expr) -> Option<String> {
        match receiver {
            // Variable reference: x.method()
            syn::Expr::Path(expr_path) => {
                if let Some(ident) = expr_path.path.get_ident() {
                    let var_name = ident.to_string();
                    if var_name == "self" {
                        return self.current_struct.clone();
                    }
                    return self.type_context.get_variable_type(&var_name).cloned();
                } else {
                    // Qualified path like SomeType::value()
                    resolve_path(
                        &expr_path.path,
                        self.module_path,
                        self.use_map,
                        self.symbol_table,
                    )
                }
            }
            // Constructor: SomeType { ... }.method()
            syn::Expr::Struct(expr_struct) => resolve_path(
                &expr_struct.path,
                self.module_path,
                self.use_map,
                self.symbol_table,
            ),
            // Method chain: a.b().c()
            syn::Expr::MethodCall(method_call) => {
                let method_name = method_call.method.to_string();
                if let Some(receiver_type) = self.infer_receiver_type(&method_call.receiver) {
                    // Try to get return type of the method (limited support)
                    self.infer_method_return_type(&receiver_type, &method_name)
                } else {
                    None
                }
            }
            // Reference: &x.method()
            syn::Expr::Reference(expr_ref) => self.infer_receiver_type(&expr_ref.expr),
            // Field access: x.field.method()
            syn::Expr::Field(expr_field) => {
                if let Some(base_type) = self.infer_receiver_type(&expr_field.base) {
                    if let syn::Member::Named(field_name) = &expr_field.member {
                        self.infer_field_type(&base_type, &field_name.to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn infer_method_return_type(&self, receiver_type: &str, method_name: &str) -> Option<String> {
        // Limited support: would need full signature parsing
        // For now, just return None - could be extended with signature analysis
        None
    }

    fn infer_field_type(&self, struct_type: &str, field_name: &str) -> Option<String> {
        let field_id = format!("{}::{}", struct_type, field_name);
        if let Some(sym) = self.symbol_table.symbols.get(&field_id) {
            if sym.kind == "field" {
                // Would need type annotation parsing to get the actual type
                // For now, return None - this is a limitation
                return None;
            }
        }
        None
    }
}

// Helper methods for OccurrenceVisitor (not part of Visit trait)
impl SymbolOccurrenceVisitor<'_> {
    fn infer_expr_type_helper(&self, expr: &syn::Expr) -> Option<String> {
        match expr {
            syn::Expr::Path(expr_path) => resolve_path(
                &expr_path.path,
                self.module_path,
                self.use_map,
                self.symbol_table,
            ),
            syn::Expr::Struct(expr_struct) => resolve_path(
                &expr_struct.path,
                self.module_path,
                self.use_map,
                self.symbol_table,
            ),
            syn::Expr::Call(expr_call) => {
                if let syn::Expr::Path(path_expr) = &*expr_call.func {
                    resolve_path(
                        &path_expr.path,
                        self.module_path,
                        self.use_map,
                        self.symbol_table,
                    )
                } else {
                    None
                }
            }
            syn::Expr::MethodCall(_) => None,
            _ => None,
        }
    }
}

impl<'ast> Visit<'ast> for SymbolOccurrenceVisitor<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        if self.symbol_table.symbols.contains_key(&id) {
            self.add_occurrence(id, "def", i.ident.span());
        }
        visit::visit_item_mod(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let struct_path = type_path_string(&i.self_ty, self.module_path);
        let trait_path = i
            .trait_
            .as_ref()
            .map(|(_, path, _)| path_to_string(path, self.module_path));
        self.current_impl = Some(ImplContext {
            struct_path: struct_path.clone(),
            trait_path: trait_path.clone(),
        });
        self.current_struct = Some(struct_path);
        visit::visit_item_impl(self, i);
        self.current_impl = None;
        self.current_struct = None;
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id, "def", i.ident.span());
        for field in &i.fields {
            if let Some(ident) = &field.ident {
                let fid = format!(
                    "{}::{}",
                    module_child_path(self.module_path, i.ident.to_string()),
                    ident
                );
                self.add_occurrence(fid, "def", ident.span());
            }
        }
        visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id.clone(), "def", i.ident.span());
        for variant in &i.variants {
            let vid = format!("{}::{}", id, variant.ident);
            self.add_occurrence(vid.clone(), "def", variant.ident.span());
            for field in &variant.fields {
                if let Some(ident) = &field.ident {
                    let fid = format!("{}::{}", vid, ident);
                    self.add_occurrence(fid, "def", ident.span());
                }
            }
        }
        visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id.clone(), "def", i.ident.span());
        for item in &i.items {
            if let syn::TraitItem::Fn(method) = item {
                let mid = format!("{}::{}", id, method.sig.ident);
                self.add_occurrence(mid, "def", method.sig.ident.span());
            }
        }
        visit::visit_item_trait(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let id = module_child_path(self.module_path, i.sig.ident.to_string());
        self.add_occurrence(id, "def", i.sig.ident.span());
        visit::visit_item_fn(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id, "def", i.ident.span());
        visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id, "def", i.ident.span());
        visit::visit_item_const(self, i);
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        self.add_occurrence(id, "def", i.ident.span());
        visit::visit_item_static(self, i);
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        if let Some(ident) = &i.ident {
            let id = module_child_path(self.module_path, ident.to_string());
            self.add_occurrence(id, "def", ident.span());
        }
        visit::visit_item_macro(self, i);
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let mut prefix = Vec::new();
        if i.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        record_use_tree(
            &i.tree,
            &mut prefix,
            self.module_path,
            self.symbol_table,
            self.occurrences,
            self.file,
        );
        visit::visit_item_use(self, i);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if let Some(ctx) = &self.current_impl {
            let name = i.sig.ident.to_string();
            let id = if let Some(trait_path) = &ctx.trait_path {
                format!("{} as {}::{}", ctx.struct_path, trait_path, name)
            } else {
                format!("{}::{}", ctx.struct_path, name)
            };
            self.add_occurrence(id, "def", i.sig.ident.span());
        }
        visit::visit_impl_item_fn(self, i);
    }

    fn visit_path(&mut self, path: &'ast syn::Path) {
        if let Some(id) = resolve_path(path, self.module_path, self.use_map, &self.symbol_table) {
            if let Some(seg) = path.segments.last() {
                self.add_occurrence(id.clone(), "use", seg.ident.span());
            }
            let full_parts: Vec<&str> = id.split("::").collect();
            if full_parts.len() == path.segments.len() {
                for (idx, seg) in path.segments.iter().enumerate() {
                    let prefix = full_parts[..=idx].join("::");
                    if let Some(sym) = self.symbol_table.symbols.get(&prefix) {
                        if sym.kind == "module" {
                            self.add_occurrence(prefix, "use", seg.ident.span());
                        }
                    }
                }
            }
        }
        visit::visit_path(self, path);
    }

    fn visit_expr_method_call(&mut self, i: &'ast syn::ExprMethodCall) {
        let method_name = i.method.to_string();
        let mut resolved = false;

        // Try to infer receiver type from the expression
        if let Some(receiver_type) = self.infer_receiver_type(&i.receiver) {
            if let Some(method_id) = self
                .type_context
                .resolve_method(&receiver_type, &method_name)
            {
                self.add_occurrence(method_id, "use", i.method.span());
                resolved = true;
            }
        }

        // Fallback to impl context if we're inside an impl block
        if !resolved {
            if let Some(ctx) = &self.current_impl {
                let id = if let Some(trait_path) = &ctx.trait_path {
                    format!("{} as {}::{}", ctx.struct_path, trait_path, method_name)
                } else {
                    format!("{}::{}", ctx.struct_path, method_name)
                };
                if self.symbol_table.symbols.contains_key(&id) {
                    self.add_occurrence(id, "use", i.method.span());
                }
            }
        }

        visit::visit_expr_method_call(self, i);
    }

    fn visit_expr_field(&mut self, i: &'ast syn::ExprField) {
        if let (Some(struct_path), syn::Member::Named(ident)) = (&self.current_struct, &i.member) {
            let id = format!("{}::{}", struct_path, ident);
            if self.symbol_table.symbols.contains_key(&id) {
                self.add_occurrence(id, "use", ident.span());
            }
        }
        visit::visit_expr_field(self, i);
    }

    fn visit_expr_struct(&mut self, i: &'ast syn::ExprStruct) {
        if let Some(struct_id) =
            resolve_path(&i.path, self.module_path, self.use_map, &self.symbol_table)
        {
            for field in &i.fields {
                if let syn::Member::Named(ident) = &field.member {
                    let id = format!("{}::{}", struct_id, ident);
                    if self.symbol_table.symbols.contains_key(&id) {
                        self.add_occurrence(id, "use", ident.span());
                    }
                }
            }
        }
        visit::visit_expr_struct(self, i);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        // Track let bindings for type inference
        if let syn::Pat::Ident(pat_ident) = &local.pat {
            let var_name = pat_ident.ident.to_string();

            // Try to infer type from initializer
            if let Some(init) = &local.init {
                if let Some(inferred_type) = self.infer_expr_type_helper(&init.expr) {
                    self.type_context.bind_variable(var_name, inferred_type);
                }
            }
        }

        visit::visit_local(self, local);
    }
}

fn resolve_path(
    path: &syn::Path,
    module_path: &str,
    use_map: &HashMap<String, String>,
    table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    if segments.is_empty() {
        return None;
    }
    let mut base: Vec<String> = Vec::new();
    let mut rest = segments.as_slice();

    if segments[0] == "crate" {
        base.push("crate".to_string());
        rest = &segments[1..];
    } else if segments[0] == "self" || segments[0] == "super" {
        let mut module_parts: Vec<String> =
            module_path.split("::").map(|s| s.to_string()).collect();
        let mut idx = 0;
        while idx < segments.len() && segments[idx] == "super" {
            if module_parts.len() > 1 {
                module_parts.pop();
            }
            idx += 1;
        }
        if idx < segments.len() && segments[idx] == "self" {
            idx += 1;
        }
        base = module_parts;
        rest = &segments[idx..];
    } else if let Some(target) = use_map.get(&segments[0]) {
        base = target.split("::").map(|s| s.to_string()).collect();
        rest = &segments[1..];
    } else {
        base = module_path.split("::").map(|s| s.to_string()).collect();
    }

    let mut full = base;
    full.extend(rest.iter().cloned());
    let full_path = full.join("::");
    if table.symbols.contains_key(&full_path) {
        Some(full_path)
    } else {
        None
    }
}

fn build_use_map(ast: &syn::File, module_path: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for item in &ast.items {
        let syn::Item::Use(u) = item else { continue };
        let mut prefix = Vec::new();
        if u.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        use_tree_to_map(&u.tree, &mut prefix, module_path, &mut map);
    }
    map
}

fn use_tree_to_map(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    module_path: &str,
    map: &mut HashMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            use_tree_to_map(&p.tree, prefix, module_path, map);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(name.ident.to_string());
            map.insert(name.ident.to_string(), full.join("::"));
        }
        syn::UseTree::Rename(rename) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(rename.ident.to_string());
            map.insert(rename.rename.to_string(), full.join("::"));
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                use_tree_to_map(item, prefix, module_path, map);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

fn record_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    module_path: &str,
    table: &SymbolIndex,
    occurrences: &mut Vec<SymbolOccurrence>,
    file: &Path,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            record_use_tree(&p.tree, prefix, module_path, table, occurrences, file);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(name.ident.to_string());
            let id = full.join("::");
            if table.symbols.contains_key(&id) {
                occurrences.push(SymbolOccurrence {
                    id,
                    file: file.to_string_lossy().to_string(),
                    kind: "use".to_string(),
                    span: span_to_range(name.ident.span()),
                });
            }
        }
        syn::UseTree::Rename(rename) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(rename.ident.to_string());
            let id = full.join("::");
            if table.symbols.contains_key(&id) {
                occurrences.push(SymbolOccurrence {
                    id: id.clone(),
                    file: file.to_string_lossy().to_string(),
                    kind: "use_alias".to_string(),
                    span: span_to_range(rename.ident.span()),
                });
                // Also track the alias name as a separate occurrence
                let alias_id = format!("{}@alias:{}", id, rename.rename);
                occurrences.push(SymbolOccurrence {
                    id: alias_id,
                    file: file.to_string_lossy().to_string(),
                    kind: "alias_def".to_string(),
                    span: span_to_range(rename.rename.span()),
                });
            }
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                record_use_tree(item, prefix, module_path, table, occurrences, file);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

fn normalize_use_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    if prefix.first().map(|s| s.as_str()) == Some("crate") {
        return prefix.to_vec();
    }
    if prefix.first().map(|s| s.as_str()) == Some("self")
        || prefix.first().map(|s| s.as_str()) == Some("super")
    {
        return resolve_relative_prefix(prefix, module_path);
    }
    let mut out: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    out.extend(prefix.iter().cloned());
    out
}

fn resolve_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    let mut module_parts: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    let mut idx = 0usize;
    while idx < prefix.len() && prefix[idx] == "super" {
        if module_parts.len() > 1 {
            module_parts.pop();
        }
        idx += 1;
    }
    if idx < prefix.len() && prefix[idx] == "self" {
        idx += 1;
    }
    let mut out = module_parts;
    out.extend(prefix[idx..].iter().cloned());
    out
}

fn plan_file_renames(
    table: &SymbolIndex,
    mapping: &HashMap<String, String>,
) -> Result<Vec<FileRename>> {
    let mut renames = Vec::new();
    for (id, new_name) in mapping {
        let Some(sym) = table.symbols.get(id) else {
            continue;
        };
        if sym.kind != "module" {
            continue;
        }

        let Some(def_path_string) = sym.definition_file.clone() else {
            continue;
        };
        let def_path = PathBuf::from(&def_path_string);

        // Check if new_name contains path separators (directory move)
        let is_directory_move = new_name.contains('/') || new_name.contains("::");

        if is_directory_move {
            // Handle directory moves: new_name is a new module path
            if let Some(new_path) = compute_new_file_path(&def_path_string, &sym.id, new_name)? {
                let new_module_id = if new_name.starts_with("crate::") {
                    new_name.to_string()
                } else if new_name.contains("::") {
                    new_name.to_string()
                } else {
                    // Simple name - replace last component
                    let parts: Vec<&str> = sym.id.split("::").collect();
                    if parts.len() > 1 {
                        format!("{}::{}", parts[..parts.len() - 1].join("::"), new_name)
                    } else {
                        format!("crate::{}", new_name)
                    }
                };
                renames.push(FileRename {
                    from: def_path_string.clone(),
                    to: new_path,
                    is_directory_move: true,
                    old_module_id: sym.id.clone(),
                    new_module_id,
                });
            }
        } else {
            // Handle simple renames within same directory
            let path = def_path.as_path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };

            // For mod.rs files, the module name is the parent directory name
            let matches_module = if stem == "mod" {
                // For src/refactor/mod.rs, check if parent dir name matches
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .map(|dir_name| dir_name == sym.name)
                    .unwrap_or(false)
            } else {
                stem == sym.name
            };

            if !matches_module {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }

            // Determine the new path
            let new_path = if stem == "mod" {
                // For mod.rs files, we rename the parent directory
                // src/refactor/mod.rs -> the directory src/refactor/ should become src/refactoring/
                // Store the directory path, not the mod.rs file path
                path.parent().unwrap().to_path_buf()
            } else {
                // Rename file: src/refactor.rs -> src/refactoring.rs
                let mut new_path = path.to_path_buf();
                new_path.set_file_name(format!("{}.rs", new_name));
                new_path
            };

            let new_module_id = {
                let parts: Vec<&str> = sym.id.split("::").collect();
                if parts.len() > 1 {
                    format!("{}::{}", parts[..parts.len() - 1].join("::"), new_name)
                } else {
                    format!("crate::{}", new_name)
                }
            };

            let (from, to) = if stem == "mod" {
                // Rename the directory
                let old_dir = path.parent().unwrap();
                let new_dir = old_dir.parent().unwrap().join(new_name);
                (
                    old_dir.to_string_lossy().to_string(),
                    new_dir.to_string_lossy().to_string(),
                )
            } else {
                // Rename the file
                (
                    def_path_string.clone(),
                    new_path.to_string_lossy().to_string(),
                )
            };

            renames.push(FileRename {
                from,
                to,
                is_directory_move: false,
                old_module_id: sym.id.clone(),
                new_module_id,
            });
        }
    }
    Ok(renames)
}

/// Compute new file path when moving a module to a new location
/// new_module_path can be either:
///   - A module path like "crate::new::location::module"
///   - A file path like "src/new/location/module.rs"
fn compute_new_file_path(
    old_file: &str,
    old_module_id: &str,
    new_module_path: &str,
) -> Result<Option<String>> {
    let old_path = Path::new(old_file);

    // Determine project root by finding 'src' directory
    let mut project_root = old_path.to_path_buf();
    let mut found_src = false;
    while let Some(parent) = project_root.parent() {
        if project_root.file_name().and_then(|s| s.to_str()) == Some("src") {
            found_src = true;
            project_root = parent.to_path_buf();
            break;
        }
        project_root = parent.to_path_buf();
    }

    if !found_src {
        // Can't determine project structure
        return Ok(None);
    }

    // Parse new module path
    let new_path_str = if new_module_path.starts_with("crate::") {
        // Module path format: "crate::foo::bar"
        new_module_path.trim_start_matches("crate::")
    } else if new_module_path.contains("::") {
        // Relative module path
        new_module_path
    } else if new_module_path.contains('/') {
        // File path format: "src/foo/bar.rs" or "foo/bar"
        new_module_path
            .trim_start_matches("src/")
            .trim_end_matches(".rs")
    } else {
        // Simple name
        new_module_path
    };

    // Convert module path to file path
    let parts: Vec<&str> = new_path_str.split("::").collect();
    let mut new_file_path = project_root.join("src");

    // Check if old file was mod.rs
    let is_mod_rs = old_path.file_name().and_then(|s| s.to_str()) == Some("mod.rs");

    if parts.is_empty() {
        return Ok(None);
    }

    // Build directory structure
    for part in &parts[..parts.len() - 1] {
        new_file_path.push(part);
    }

    // Add final component
    let last_part = parts[parts.len() - 1];
    if is_mod_rs {
        // mod.rs stays as mod.rs in new location
        new_file_path.push(last_part);
        new_file_path.push("mod.rs");
    } else {
        // Regular file: foo.rs
        new_file_path.push(format!("{}.rs", last_part));
    }

    Ok(Some(new_file_path.to_string_lossy().to_string()))
}

fn module_path_for_file(project: &Path, file: &Path) -> String {
    let mut rel = file.strip_prefix(project).unwrap_or(file).to_path_buf();
    if rel.components().next().map(|c| c.as_os_str()) == Some("src".as_ref()) {
        rel = rel.strip_prefix("src").unwrap_or(&rel).to_path_buf();
    }
    let mut parts: Vec<String> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
        .collect();
    if parts.is_empty() {
        return "crate".to_string();
    }
    if let Some(last) = parts.last_mut() {
        if last == "lib.rs" || last == "main.rs" {
            parts.pop();
        } else if last == "mod.rs" {
            parts.pop();
        } else if last.ends_with(".rs") {
            *last = last.trim_end_matches(".rs").to_string();
        }
    }
    if parts.is_empty() {
        "crate".to_string()
    } else {
        format!("crate::{}", parts.join("::"))
    }
}

fn module_child_path(module_path: &str, child: String) -> String {
    if module_path == "crate" {
        format!("crate::{}", child)
    } else {
        format!("{}::{}", module_path, child)
    }
}

fn path_to_string(path: &syn::Path, module_path: &str) -> String {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    if segments.first().map(|s| s.as_str()) == Some("crate") {
        segments.join("::")
    } else if segments.first().map(|s| s.as_str()) == Some("self")
        || segments.first().map(|s| s.as_str()) == Some("super")
    {
        let rel = resolve_relative_prefix(&segments, module_path);
        rel.join("::")
    } else {
        format!("{}::{}", module_path, segments.join("::"))
    }
}

fn type_path_string(ty: &syn::Type, module_path: &str) -> String {
    if let syn::Type::Path(tp) = ty {
        path_to_string(&tp.path, module_path)
    } else {
        format!("{}::{}", module_path, "Self")
    }
}

pub fn span_to_range(span: Span) -> SpanRange {
    let start = span.start();
    let end = span.end();
    SpanRange {
        start: LineColumn {
            line: start.line as i64,
            column: start.column as i64 + 1,
        },
        end: LineColumn {
            line: end.line as i64,
            column: end.column as i64 + 1,
        },
    }
}

/// B2: Write preview with structured edit tracking
fn write_preview(
    out: &Path,
    edits: &[SymbolEdit],
    renames: &[FileRename],
    structured_tracker: &StructuredEditTracker,
    config: &StructuredEditConfig,
) -> Result<()> {
    let mut by_file: BTreeMap<String, Vec<&SymbolEdit>> = BTreeMap::new();
    for edit in edits {
        by_file.entry(edit.file.clone()).or_default().push(edit);
    }
    let mut files = BTreeMap::new();
    for (file, edits) in by_file {
        let list: Vec<_> = edits
            .into_iter()
            .map(|e| {
                serde_json::json!({
                    "id": e.id,
                    "kind": e.kind,
                    "start": e.start,
                    "end": e.end,
                    "new_name": e.new_name,
                })
            })
            .collect();
        files.insert(file, list);
    }
    let rename_list: Vec<_> = renames
        .iter()
        .map(|r| {
            serde_json::json!({
                "from": r.from,
                "to": r.to,
                "is_directory_move": r.is_directory_move,
                "old_module_id": r.old_module_id,
                "new_module_id": r.new_module_id
            })
        })
        .collect();
    // B2: Include detailed structured edit breakdown in preview
    let mut structured: Vec<_> = structured_tracker.all_files().iter().cloned().collect();
    structured.sort();

    // D2: Enhanced preview metadata with per-pass breakdowns
    let mut doc_files: Vec<_> = structured_tracker.doc_files().iter().cloned().collect();
    doc_files.sort();
    let mut attr_files: Vec<_> = structured_tracker.attr_files().iter().cloned().collect();
    attr_files.sort();
    let mut use_files: Vec<_> = structured_tracker.use_files().iter().cloned().collect();
    use_files.sort();

    let preview = serde_json::json!({
        "files": files,
        "file_renames": rename_list,
        "structured_files": structured,
        "structured_edits": {
            "enabled": config.is_enabled(),
            "config": config.summary(),
            "summary": structured_tracker.summary(config),
            "total_files": structured_tracker.all_files().len(),
            "by_pass": {
                "doc_literals": {
                    "enabled": config.doc_literals_enabled(),
                    "files": doc_files,
                    "count": structured_tracker.doc_files().len(),
                },
                "attr_literals": {
                    "enabled": config.attr_literals_enabled(),
                    "files": attr_files,
                    "count": structured_tracker.attr_files().len(),
                },
                "use_statements": {
                    "enabled": config.use_statements_enabled(),
                    "files": use_files,
                    "count": structured_tracker.use_files().len(),
                },
            },
        },
    });
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&preview)?)?;
    Ok(())
}

pub(crate) fn span_to_offsets(
    content: &str,
    start: &LineColumn,
    end: &LineColumn,
) -> (usize, usize) {
    let mut line_starts = Vec::new();
    let mut offset = 0usize;
    for line in content.split_inclusive('\n') {
        line_starts.push(offset);
        offset += line.len();
    }
    let start_line = (start.line.max(1) - 1) as usize;
    let end_line = (end.line.max(1) - 1) as usize;
    let start_offset =
        line_starts.get(start_line).cloned().unwrap_or(0) + start.column.saturating_sub(1) as usize;
    let end_offset = line_starts.get(end_line).cloned().unwrap_or(start_offset)
        + end.column.saturating_sub(1) as usize;
    (start_offset, end_offset)
}

/// D1: Format files with per-file error context
///
/// Only formats files that exist and were actually modified.
/// Surfaces errors with file context for better diagnostics.
fn format_files(paths: &[PathBuf]) -> Result<Vec<String>> {
    let existing: Vec<_> = paths.iter().filter(|p| p.exists()).collect();
    if existing.is_empty() {
        return Ok(Vec::new());
    }

    let mut errors = Vec::new();

    // D1: Format files individually to isolate errors
    for path in &existing {
        let mut cmd = Command::new("rustfmt");
        cmd.arg("--edition").arg("2021");
        cmd.arg(path);

        match cmd.status() {
            Ok(status) if status.success() => {
                // Success - continue
            }
            Ok(status) => {
                errors.push(format!(
                    "rustfmt failed for {}: exit code {}",
                    path.display(),
                    status.code().unwrap_or(-1)
                ));
            }
            Err(e) => {
                errors.push(format!(
                    "failed to run rustfmt on {}: {}",
                    path.display(),
                    e
                ));
            }
        }
    }

    if !errors.is_empty() {
        eprintln!("Warning: {} rustfmt errors occurred:", errors.len());
        for error in &errors {
            eprintln!("  - {}", error);
        }
    }

    Ok(errors)
}

/// D1: Flush buffers and format only mutated files
///
/// Contract:
/// - Only flushes dirty buffers (files with pending edits)
/// - Only runs rustfmt on files that were actually written
/// - Surfaces rustfmt errors with file context
/// - Reports formatted files and edit counts
///
/// # D1: Example
///
/// ```no_run
/// use semantic_lint::rename::rewrite::RewriteBufferSet;
/// use std::path::Path;
///
/// let mut buffers = RewriteBufferSet::new();
/// let file = Path::new("src/main.rs");
/// let content = std::fs::read_to_string(file).unwrap();
///
/// // Queue some edits
/// let buffer = buffers.ensure_buffer(file, &content);
/// buffer.replace(0, 3, "pub").unwrap();
///
/// // Flush will only format files that were actually modified
/// // and report any rustfmt errors with file context
/// let touched = buffers.flush().unwrap();
/// println!("Modified {} files", touched.len());
/// ```
fn flush_and_format(buffers: &mut RewriteBufferSet) -> Result<()> {
    let total_edits = buffers.total_edit_count();
    let touched = buffers.flush()?;

    if !touched.is_empty() {
        let format_errors = format_files(&touched)?;
        if !format_errors.is_empty() {
            eprintln!(
                "Note: {} file(s) were modified but rustfmt encountered errors",
                touched.len()
            );
        } else {
            eprintln!(
                "Formatted {} file(s) ({} edits applied)",
                touched.len(),
                total_edits
            );
        }
    }

    Ok(())
}

fn queue_file_edits(
    buffers: &mut RewriteBufferSet,
    file: &Path,
    content: &str,
    edits: &[SymbolEdit],
) -> Result<()> {
    if edits.is_empty() {
        return Ok(());
    }
    let mut text_edits = Vec::new();
    for edit in edits {
        let (start, end) = span_to_offsets(content, &edit.start, &edit.end);
        if start >= end || end > content.len() {
            continue;
        }
        text_edits.push(SourceTextEdit {
            start,
            end,
            text: edit.new_name.clone(),
        });
    }
    buffers.queue_edits(file, content, text_edits)?;
    Ok(())
}

fn queue_alias_edits(buffers: &mut RewriteBufferSet, edits: &[SymbolEdit]) -> Result<()> {
    let mut by_file: HashMap<String, Vec<SymbolEdit>> = HashMap::new();
    for edit in edits {
        by_file
            .entry(edit.file.clone())
            .or_default()
            .push(edit.clone());
    }
    for (path_str, file_edits) in by_file {
        let path = Path::new(&path_str);
        let content = std::fs::read_to_string(path)?;
        queue_file_edits(buffers, path, &content, &file_edits)?;
    }
    Ok(())
}

fn is_valid_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// Update mod declarations after file renames/moves
fn update_mod_declarations(
    project: &Path,
    table: &SymbolIndex,
    file_renames: &[FileRename],
) -> Result<()> {
    if file_renames.is_empty() {
        return Ok(());
    }

    let mut rename_lookup: HashMap<String, PathBuf> = HashMap::new();
    for rename in file_renames {
        rename_lookup.insert(rename.from.clone(), PathBuf::from(&rename.to));
    }

    // Build mapping of affected modules
    let mut module_changes: HashMap<String, ModuleRenamePlan> = HashMap::new();
    let mut files_to_process: HashSet<PathBuf> = HashSet::new();
    let mut fallback_required = false;

    for rename in file_renames {
        let old_parts: Vec<&str> = rename.old_module_id.split("::").collect();
        let new_parts: Vec<&str> = rename.new_module_id.split("::").collect();

        if old_parts.len() < 2 || new_parts.len() < 2 {
            continue; // crate-level modules don't need mod declarations updated
        }

        let old_module_name = old_parts.last().unwrap();
        let new_module_name = new_parts.last().unwrap();
        let old_parent = old_parts[..old_parts.len() - 1].join("::");
        let new_parent = new_parts[..new_parts.len() - 1].join("::");

        let change = ModuleRenamePlan {
            old_name: old_module_name.to_string(),
            new_name: new_module_name.to_string(),
            old_parent: old_parent.clone(),
            new_parent: new_parent.clone(),
        };
        module_changes.insert(rename.old_module_id.clone(), change);

        if let Some(entry) = table.symbols.get(&rename.old_module_id) {
            if let Some(decl_file) = entry.declaration_file.as_ref() {
                let candidate = resolve_renamed_path(PathBuf::from(decl_file), &rename_lookup);
                if candidate.exists() {
                    files_to_process.insert(candidate);
                } else {
                    fallback_required = true;
                }
            } else {
                fallback_required = true;
            }
        } else {
            fallback_required = true;
        }

        if old_parent != new_parent {
            match resolve_parent_definition_file(project, table, &new_parent, &rename_lookup) {
                Some(path) if path.exists() => {
                    files_to_process.insert(path);
                }
                Some(_) | None => fallback_required = true,
            }
        }
    }

    let target_files: Vec<PathBuf> = if fallback_required || files_to_process.is_empty() {
        fs::collect_rs_files(project)?
    } else {
        files_to_process.into_iter().collect()
    };

    for file in &target_files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;

        let mut mod_edits = Vec::new();

        // Find mod declarations that need updating
        for item in &ast.items {
            if let syn::Item::Mod(item_mod) = item {
                let mod_name = item_mod.ident.to_string();
                let child_module_id = module_child_path(&module_path, mod_name.clone());

                if let Some(change) = module_changes.get(&child_module_id) {
                    if change.old_parent == change.new_parent && change.old_parent == module_path {
                        if change.old_name != change.new_name {
                            mod_edits.push(ModuleDeclarationEdit {
                            kind: ModuleDeclarationEditKind::Rename,
                                span: span_to_range(item_mod.ident.span()),
                                old_name: change.old_name.clone(),
                                new_name: Some(change.new_name.clone()),
                            });
                        }
                    } else if change.old_parent == module_path {
                        mod_edits.push(ModuleDeclarationEdit {
                            kind: ModuleDeclarationEditKind::Remove,
                            span: span_to_range(item_mod.ident.span()),
                            old_name: change.old_name.clone(),
                            new_name: None,
                        });
                    } else if change.new_parent == module_path {
                        if change.old_name != change.new_name {
                            mod_edits.push(ModuleDeclarationEdit {
                            kind: ModuleDeclarationEditKind::Rename,
                                span: span_to_range(item_mod.ident.span()),
                                old_name: change.old_name.clone(),
                                new_name: Some(change.new_name.clone()),
                            });
                        }
                    }
                }
            }
        }

        // Check if we need to add new mod declarations
        for change in module_changes.values() {
            if change.new_parent == module_path {
                let has_declaration = ast.items.iter().any(|item| {
                    if let syn::Item::Mod(item_mod) = item {
                        item_mod.ident.to_string() == change.new_name
                    } else {
                        false
                    }
                });

                if !has_declaration && change.old_parent != module_path {
                    mod_edits.push(ModuleDeclarationEdit {
                        kind: ModuleDeclarationEditKind::Add,
                        span: stub_range(),
                        old_name: String::new(),
                        new_name: Some(change.new_name.clone()),
                    });
                }
            }
        }

        if !mod_edits.is_empty() {
            apply_mod_edits(&content, file, &mod_edits)?;
        }
    }

    Ok(())
}

fn resolve_renamed_path(path: PathBuf, lookup: &HashMap<String, PathBuf>) -> PathBuf {
    let key = path.to_string_lossy().to_string();
    lookup.get(&key).cloned().unwrap_or(path)
}

fn resolve_parent_definition_file(
    project: &Path,
    table: &SymbolIndex,
    parent_id: &str,
    rename_lookup: &HashMap<String, PathBuf>,
) -> Option<PathBuf> {
    let raw_path = if parent_id == "crate" {
        find_crate_root_file(project)?
    } else {
        let entry = table.symbols.get(parent_id)?;
        let file = entry
            .definition_file
            .as_ref()
            .or(entry.declaration_file.as_ref())?
            .to_string();
        PathBuf::from(file)
    };
    Some(resolve_renamed_path(raw_path, rename_lookup))
}

fn find_crate_root_file(project: &Path) -> Option<PathBuf> {
    let lib = project.join("src/lib.rs");
    if lib.exists() {
        return Some(lib);
    }
    let main = project.join("src/main.rs");
    if main.exists() {
        return Some(main);
    }
    None
}

#[derive(Debug)]
struct ModuleRenamePlan {
    old_name: String,
    new_name: String,
    old_parent: String,
    new_parent: String,
}

#[derive(Debug)]
struct ModuleDeclarationEdit {
    kind: ModuleDeclarationEditKind,
    span: SpanRange,
    old_name: String,
    new_name: Option<String>,
}

#[derive(Debug)]
enum ModuleDeclarationEditKind {
    Add,
    Remove,
    Rename,
}

/// Basic type inference context for method resolution
struct LocalTypeContext {
    // Maps variable names to their types (limited scope)
    local_bindings: HashMap<String, String>,
    // Symbol table reference for looking up methods
    symbol_table_ref: *const SymbolIndex,
}

impl LocalTypeContext {
    fn new(symbol_table: &SymbolIndex) -> Self {
        Self {
            local_bindings: HashMap::new(),
            symbol_table_ref: symbol_table as *const SymbolIndex,
        }
    }

    fn bind_variable(&mut self, name: String, type_path: String) {
        self.local_bindings.insert(name, type_path);
    }

    fn get_variable_type(&self, name: &str) -> Option<&String> {
        self.local_bindings.get(name)
    }

    fn resolve_method(&self, receiver_type: &str, method_name: &str) -> Option<String> {
        // Safety: We maintain the lifetime properly in OccurrenceVisitor
        let symbol_table = unsafe { &*self.symbol_table_ref };

        // Try direct impl method
        let direct_id = format!("{}::{}", receiver_type, method_name);
        if symbol_table.symbols.contains_key(&direct_id) {
            return Some(direct_id);
        }

        // Try trait methods - look for "Type as Trait::method" patterns
        for (id, sym) in &symbol_table.symbols {
            if sym.kind == "method"
                && id.contains(" as ")
                && id.ends_with(&format!("::{}", method_name))
            {
                if id.starts_with(receiver_type) {
                    return Some(id.clone());
                }
            }
        }

        None
    }

    fn clear_scope(&mut self) {
        self.local_bindings.clear();
    }
}

fn apply_mod_edits(content: &str, file: &Path, edits: &[ModuleDeclarationEdit]) -> Result<()> {
    let mut new_content = content.to_string();
    let lines: Vec<&str> = content.lines().collect();

    for edit in edits {
        match edit.kind {
            ModuleDeclarationEditKind::Remove => {
                // Find and remove the mod declaration line
                let line_idx = (edit.span.start.line - 1) as usize;
                if line_idx < lines.len() {
                    let line = lines[line_idx];
                    if line.trim().starts_with("mod ") && line.contains(&edit.old_name) {
                        new_content = new_content.replace(line, "");
                    }
                }
            }
            ModuleDeclarationEditKind::Rename => {
                if let Some(new_name) = &edit.new_name {
                    let (start, end) = span_to_offsets(content, &edit.span.start, &edit.span.end);
                    if start < end && end <= new_content.len() {
                        let current = &new_content[start..end];
                        if current == edit.old_name {
                            new_content.replace_range(start..end, new_name);
                        }
                    }
                }
            }
            ModuleDeclarationEditKind::Add => {
                if let Some(new_name) = &edit.new_name {
                    // Add mod declaration at the beginning of the file (after any initial comments/attrs)
                    let insert_pos = find_mod_insert_position(content);
                    let declaration = format!("mod {};\n", new_name);
                    new_content.insert_str(insert_pos, &declaration);
                }
            }
        }
    }

    if new_content != content {
        std::fs::write(file, new_content)?;
    }

    Ok(())
}

fn find_mod_insert_position(content: &str) -> usize {
    // Find position after file-level attributes and doc comments
    let mut pos = 0;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("//!")
            || trimmed.starts_with("#![")
            || trimmed.starts_with("//")
        {
            pos += line.len() + 1; // +1 for newline
        } else {
            break;
        }
    }
    pos
}

/// B2: Update use statement paths after module moves with structured tracking
fn update_use_paths(
    project: &Path,
    file_renames: &[FileRename],
    structured_config: &StructuredEditConfig,
    buffers: &mut RewriteBufferSet,
    alias_graph: &AliasGraph,
    structured_tracker: &mut StructuredEditTracker,
) -> Result<()> {
    if file_renames.is_empty() {
        return Ok(());
    }

    // Build mapping of old module paths to new module paths
    let mut path_updates: HashMap<String, String> = HashMap::new();
    for rename in file_renames {
        path_updates.insert(rename.old_module_id.clone(), rename.new_module_id.clone());
    }

    let files = fs::collect_rs_files(project)?;
    let structured_uses = structured_config.use_statements_enabled();
    for file in &files {
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;

        if structured_uses {
            use crate::rename::structured::orchestrator::StructuredPass;
            use crate::rename::structured::structured_edit_config;
            use crate::rename::structured::use_tree::UseTreePass;

            let file_key = file.to_string_lossy().to_string();
            let alias_nodes = alias_graph
                .nodes_in_file(&file_key)
                .into_iter()
                .cloned()
                .collect::<Vec<_>>();

            let mut pass =
                UseTreePass::new(path_updates.clone(), alias_nodes, structured_edit_config());

            if pass.execute(file, &content, &ast, buffers)? {
                structured_tracker.mark_use_edit(file_key);
            }

            continue;
        }

        let mut use_edits = Vec::new();

        // Process all use statements
        for item in &ast.items {
            if let syn::Item::Use(use_item) = item {
                collect_use_path_edits(
                    &use_item.tree,
                    use_item.leading_colon.is_some(),
                    &path_updates,
                    &mut use_edits,
                );
            }
        }

        if !use_edits.is_empty() {
            apply_use_path_edits(&content, file, &use_edits)?;
        }
    }

    Ok(())
}

#[derive(Debug)]
struct UsePathEdit {
    span: SpanRange,
    old_path: String,
    new_path: String,
}

fn collect_use_path_edits(
    tree: &syn::UseTree,
    has_leading_colon: bool,
    path_updates: &HashMap<String, String>,
    edits: &mut Vec<UsePathEdit>,
) {
    // Build the full path from the use tree
    let mut current_path = Vec::new();
    if has_leading_colon {
        current_path.push("crate".to_string());
    }

    collect_use_tree_paths(tree, &mut current_path, path_updates, edits);
}

fn collect_use_tree_paths(
    tree: &syn::UseTree,
    current_path: &mut Vec<String>,
    path_updates: &HashMap<String, String>,
    edits: &mut Vec<UsePathEdit>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let segment = path.ident.to_string();
            current_path.push(segment.clone());

            // Check if this path prefix needs updating
            let path_str = current_path.join("::");
            if let Some(new_path) = find_replacement_path(&path_str, path_updates) {
                edits.push(UsePathEdit {
                    span: span_to_range(path.ident.span()),
                    old_path: segment,
                    new_path: extract_segment_replacement(
                        &path_str,
                        &new_path,
                        current_path.len() - 1,
                    ),
                });
            }

            collect_use_tree_paths(&path.tree, current_path, path_updates, edits);
            current_path.pop();
        }
        syn::UseTree::Name(name) => {
            let segment = name.ident.to_string();
            current_path.push(segment.clone());

            let path_str = current_path.join("::");
            if let Some(new_path) = find_replacement_path(&path_str, path_updates) {
                edits.push(UsePathEdit {
                    span: span_to_range(name.ident.span()),
                    old_path: segment,
                    new_path: extract_segment_replacement(
                        &path_str,
                        &new_path,
                        current_path.len() - 1,
                    ),
                });
            }

            current_path.pop();
        }
        syn::UseTree::Rename(rename) => {
            let segment = rename.ident.to_string();
            current_path.push(segment.clone());

            let path_str = current_path.join("::");
            if let Some(new_path) = find_replacement_path(&path_str, path_updates) {
                edits.push(UsePathEdit {
                    span: span_to_range(rename.ident.span()),
                    old_path: segment,
                    new_path: extract_segment_replacement(
                        &path_str,
                        &new_path,
                        current_path.len() - 1,
                    ),
                });
            }

            current_path.pop();
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                collect_use_tree_paths(item, current_path, path_updates, edits);
            }
        }
        syn::UseTree::Glob(_) => {
            // For glob imports, check if the parent path needs updating
            let path_str = current_path.join("::");
            if let Some(_new_path) = find_replacement_path(&path_str, path_updates) {
                // Need to update the entire glob import path
                // This is handled by parent path segments
            }
        }
    }
}

/// Find if any prefix of the path needs to be replaced
pub(crate) fn find_replacement_path(
    path: &str,
    updates: &HashMap<String, String>,
) -> Option<String> {
    // Check exact match first
    if let Some(new_path) = updates.get(path) {
        return Some(new_path.clone());
    }

    // Check if any prefix of this path was moved
    let parts: Vec<&str> = path.split("::").collect();
    for i in (1..=parts.len()).rev() {
        let prefix = parts[..i].join("::");
        if let Some(new_prefix) = updates.get(&prefix) {
            // Replace the prefix and keep the suffix
            if i < parts.len() {
                let suffix = parts[i..].join("::");
                return Some(format!("{}::{}", new_prefix, suffix));
            } else {
                return Some(new_prefix.clone());
            }
        }
    }

    None
}

/// Extract the segment that should replace the current segment
pub(crate) fn extract_segment_replacement(
    old_full_path: &str,
    new_full_path: &str,
    segment_index: usize,
) -> String {
    let new_parts: Vec<&str> = new_full_path.split("::").collect();
    if segment_index < new_parts.len() {
        new_parts[segment_index].to_string()
    } else {
        // Fallback: return the last part
        new_parts.last().unwrap_or(&"").to_string()
    }
}

fn apply_use_path_edits(content: &str, file: &Path, edits: &[UsePathEdit]) -> Result<()> {
    if edits.is_empty() {
        return Ok(());
    }

    // Sort edits by position (reverse order to apply from end to start)
    let mut sorted_edits: Vec<&UsePathEdit> = edits.iter().collect();
    sorted_edits.sort_by(|a, b| {
        let a_pos = (a.span.start.line, a.span.start.column);
        let b_pos = (b.span.start.line, b.span.start.column);
        b_pos.cmp(&a_pos)
    });

    let mut new_content = content.to_string();

    for edit in sorted_edits {
        let (start, end) = span_to_offsets(content, &edit.span.start, &edit.span.end);
        if start < end && end <= new_content.len() {
            let current = &new_content[start..end];
            if current == edit.old_path {
                new_content.replace_range(start..end, &edit.new_path);
            }
        }
    }

    if new_content != content {
        std::fs::write(file, new_content)?;
    }

    Ok(())
}

/// Track and rename use statement aliases
fn collect_and_rename_aliases(
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

        let use_map = build_use_map(&ast, &module_path);

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
                    &use_map,
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
    use_map: &HashMap<String, String>,
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
        use_map,
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
    use_map: &HashMap<String, String>,
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
                use_map,
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
                collect_alias_usage_edits(file, &alias_name, new_alias, &target_id, use_map, edits);
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
                    use_map,
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
    use_map: &HashMap<String, String>,
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
                use_map,
                edits,
            };
            use syn::visit::Visit;
            visitor.visit_file(&ast);
        }
    }
}

struct AliasUsageVisitor<'a> {
    alias_name: String,
    new_alias: String,
    target_id: String,
    file: &'a Path,
    use_map: &'a HashMap<String, String>,
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
pub fn apply_rename(
    project: &Path,
    map_path: &Path,
    dry_run: bool,
    out_path: Option<&Path>,
) -> Result<()> {
    let mapping: HashMap<String, String> =
        serde_json::from_str(&std::fs::read_to_string(map_path)?)?;
    apply_rename_with_map(project, &mapping, dry_run, out_path)
}
