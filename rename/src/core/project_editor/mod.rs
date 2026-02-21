use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use syn::visit::Visit;

use crate::alias::AliasGraph;
use crate::core::collect::{add_file_module_symbol, collect_symbols};
use crate::core::mod_decls::update_mod_declarations;
use crate::core::oracle::StructuralEditOracle;
use crate::core::paths::{module_child_path, module_path_for_file};
use crate::core::symbol_id::normalize_symbol_id;
use crate::model::types::FileRename;
use crate::model::types::SymbolIndex;
use crate::core::use_map::{path_to_string, type_path_string};
use crate::fs;
use crate::state::{NodeKind, NodeRegistry};
use crate::structured::use_tree::UsePathRewritePass;
use crate::structured::{node_handle, FieldMutation, NodeOp};
use crate::structured::{StructuredEditOptions, StructuredPass};
use compiler_capture::frontends::rustc::RustcFrontend;
use compiler_capture::multi_capture::capture_project;
use compiler_capture::project::CargoProject;
use database::graph_log::{GraphSnapshot as WireSnapshot, WireNodeId};
use database::{MemoryEngine, MemoryEngineConfig};

#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}

pub struct ProjectEditor {
    pub registry: NodeRegistry,
    pub changesets: HashMap<PathBuf, Vec<QueuedOp>>,
    pub oracle: Box<dyn StructuralEditOracle>,
    pub original_sources: HashMap<PathBuf, String>,
    pending_file_moves: Vec<(PathBuf, PathBuf)>,
    pending_file_renames: Vec<FileRename>,
    last_touched_files: HashSet<PathBuf>,
}

#[derive(Clone)]
pub struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}

impl ProjectEditor {
    pub fn load(project: &Path, oracle: Box<dyn StructuralEditOracle>) -> Result<Self> {
        let files = fs::collect_rs_files(project)?;
        let mut registry = NodeRegistry::new();
        let mut original_sources = HashMap::new();

        for file in files {
            let content = std::fs::read_to_string(&file)?;
            let ast = syn::parse_file(&content).with_context(|| format!("Failed to parse {}", file.display()))?;
            let mut builder = NodeRegistryBuilder::new(project, &file, &mut registry);
            builder.visit_file(&ast);
            registry.insert_ast(file.clone(), ast);
            original_sources.insert(file, content);
        }

        Ok(Self { registry, changesets: HashMap::new(), oracle, original_sources, pending_file_moves: Vec::new(), pending_file_renames: Vec::new(), last_touched_files: HashSet::new() })
    }

    pub fn load_with_rustc(project: &Path) -> Result<Self> {
        let cargo = CargoProject::from_entry(project)?;
        let frontend = RustcFrontend::new();
        let _artifacts = capture_project(&frontend, &cargo, &[]).with_context(|| format!("rustc capture failed for {}", project.display()))?;
        let state_dir = cargo.workspace_root().join(".rename");
        std::fs::create_dir_all(&state_dir)?;
        let tlog_path = state_dir.join("state.tlog");
        let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
        let snapshot = engine.materialized_graph()?;
        let oracle = Box::new(SnapshotOracle::from_snapshot(snapshot));
        Self::load(project, oracle)
    }

    pub fn queue(&mut self, symbol_id: &str, op: NodeOp) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = match &op {
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. }
            | NodeOp::MoveSymbol { handle, .. } => Some(handle),
            NodeOp::ReorderItems { .. } => None,
        };

        if let Some(handle) = handle {
            let exists = self.registry.handles.get(&norm);
            if exists.is_none() {
                self.registry.insert_handle(norm.clone(), handle.clone());
            }
        }

        let file = match &op {
            NodeOp::ReplaceNode { handle, .. } | NodeOp::InsertBefore { handle, .. } | NodeOp::InsertAfter { handle, .. } | NodeOp::DeleteNode { handle } | NodeOp::MutateField { handle, .. } => {
                handle.file.clone()
            }
            NodeOp::ReorderItems { file, .. } => file.clone(),
            NodeOp::MoveSymbol { handle, .. } => handle.file.clone(),
        };

        self.changesets.entry(file).or_default().push(QueuedOp { symbol_id: norm, op });
        Ok(())
    }

    pub fn queue_by_id(&mut self, symbol_id: &str, mutation: FieldMutation) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = self.registry.handles.get(&norm).cloned().with_context(|| format!("no handle found for {symbol_id}"))?;
        let op = NodeOp::MutateField { handle, mutation };
        self.queue(&norm, op)
    }

    pub fn apply(&mut self) -> Result<ChangeReport> {
        let mut touched_files: HashSet<PathBuf> = HashSet::new();
        let mut rewrites = Vec::new();
        let mut conflicts = Vec::new();
        let mut file_renames = Vec::new();
        let handle_snapshot = self.registry.handles.clone();

        // Phase 1: propagation (pure planning)
        for (_file, ops) in &self.changesets {
            for queued in ops {
                let prop = propagate(&queued.op, &queued.symbol_id, &self.registry, &*self.oracle)?;
                rewrites.extend(prop.rewrites);
                conflicts.extend(prop.conflicts);
                file_renames.extend(prop.file_renames);
            }
        }

        // Phase 2: span-based rewrites on original ASTs
        let rewrite_touched = apply_rewrites(&mut self.registry, &rewrites)?;
        touched_files.extend(rewrite_touched);

        // Phase 3: structured AST mutations
        for (file, ops) in &self.changesets {
            for queued in ops {
                let changed = {
                    let ast = self.registry.asts.get_mut(file).with_context(|| format!("missing AST for {}", file.display()))?;
                    apply_node_op(ast, &handle_snapshot, &queued.symbol_id, &queued.op).with_context(|| format!("failed to apply {}", queued.symbol_id))?
                };
                if changed {
                    touched_files.insert(file.clone());
                }
            }
        }

        let use_path_touched = run_use_path_rewrite(&mut self.registry, &self.changesets)?;
        touched_files.extend(use_path_touched);

        let mut validation = self.validate()?;
        validation.extend(conflicts);
        self.pending_file_moves = file_renames.iter().map(|r| (PathBuf::from(&r.from), PathBuf::from(&r.to))).collect();
        self.pending_file_renames = file_renames.clone();
        self.last_touched_files = touched_files.clone();
        Ok(ChangeReport { touched_files: touched_files.into_iter().collect(), conflicts: validation, file_moves: self.pending_file_moves.clone() })
    }

    pub fn validate(&self) -> Result<Vec<EditConflict>> {
        let mut conflicts = Vec::new();
        for (symbol_id, _handle) in &self.registry.handles {
            if self.oracle.is_macro_generated(symbol_id) {
                conflicts.push(EditConflict { symbol_id: symbol_id.clone(), reason: "symbol generated by macro".to_string() });
            }
        }
        for ops in self.changesets.values() {
            for queued in ops {
                if let NodeOp::MutateField { mutation, .. } = &queued.op {
                    if matches!(mutation, FieldMutation::ReplaceSignature(_)) {
                        let impacts = self.oracle.impact_of(&queued.symbol_id);
                        if !impacts.is_empty() {
                            conflicts.push(EditConflict { symbol_id: queued.symbol_id.clone(), reason: "signature change may require updating call sites".to_string() });
                        }
                    }
                }
            }
        }
        Ok(conflicts)
    }

    pub fn commit(&self) -> Result<Vec<PathBuf>> {
        let mut written = Vec::new();
        let targets: Vec<PathBuf> = if self.last_touched_files.is_empty() { self.changesets.keys().cloned().collect() } else { self.last_touched_files.iter().cloned().collect() };
        for file in targets {
            let ast = self.registry.asts.get(&file).with_context(|| format!("missing AST for {}", file.display()))?;
            let rendered = crate::structured::render_file(ast);
            std::fs::write(&file, rendered)?;
            written.push(file.clone());
        }
        for (from, to) in &self.pending_file_moves {
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::rename(from, to)?;
            written.push(to.clone());
        }
        if !self.pending_file_renames.is_empty() {
            if let Some(project_root) = find_project_root(&self.registry)? {
                let symbol_table = build_symbol_index(&project_root, &self.registry)?;
                let mut touched = HashSet::new();
                update_mod_declarations(&project_root, &symbol_table, &self.pending_file_renames, &mut touched)?;
                written.extend(touched.into_iter());
            }
        }
        Ok(written)
    }

    pub fn preview(&self) -> Result<String> {
        let mut output = String::new();
        let targets: Vec<PathBuf> = if self.last_touched_files.is_empty() { self.changesets.keys().cloned().collect() } else { self.last_touched_files.iter().cloned().collect() };
        for file in targets.into_iter().filter(|p| self.original_sources.contains_key(p)) {
            let original = &self.original_sources[&file];
            let ast = self.registry.asts.get(&file).with_context(|| format!("missing AST for {}", file.display()))?;
            let rendered = crate::structured::render_file(ast);
            if original != &rendered {
                let diff = similar::TextDiff::from_lines(original, &rendered).unified_diff().header(&format!("{} (original)", file.display()), &format!("{} (updated)", file.display())).to_string();
                output.push_str(&diff);
                output.push('\n');
            }
        }
        if output.is_empty() {
            Ok(format!("{} files touched", self.changesets.len()))
        } else {
            Ok(output)
        }
    }
}

fn run_use_path_rewrite(registry: &mut NodeRegistry, changesets: &HashMap<PathBuf, Vec<QueuedOp>>) -> Result<HashSet<PathBuf>> {
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
        let resolver = crate::resolve::ResolverContext {
            module_path: module_path_for_file(&project_root, file),
            alias_graph: alias_graph.clone(),
            symbol_table: symbol_table.clone(),
        };
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
                NodeOp::MutateField { mutation: FieldMutation::RenameIdent(new_name), .. } => {
                    let old_id = normalize_symbol_id(&queued.symbol_id);
                    if let Some(new_id) = replace_last_segment(&old_id, new_name) {
                        updates.insert(old_id, new_id);
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

fn build_symbol_index(project_root: &Path, registry: &NodeRegistry) -> Result<SymbolIndex> {
    let mut symbol_table = SymbolIndex::default();
    let mut symbols = Vec::new();
    let mut symbol_set = HashSet::new();

    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(project_root, file));
        add_file_module_symbol(&module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);
        let _ = collect_symbols(ast, &module_path, file, &mut symbol_table, &mut symbols, &mut symbol_set);
    }

    Ok(symbol_table)
}

fn find_project_root(registry: &NodeRegistry) -> Result<Option<PathBuf>> {
    let file = match registry.asts.keys().next() {
        Some(f) => f,
        None => return Ok(None),
    };
    let mut current = file.parent().unwrap_or_else(|| Path::new("/")).to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() {
            return Ok(Some(current));
        }
        if !current.pop() {
            break;
        }
    }
    Ok(None)
}

mod ops;
mod propagate;

use ops::apply_node_op;
use propagate::{apply_rewrites, propagate};

struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
}

impl<'a> NodeRegistryBuilder<'a> {
    fn new(project_root: &'a Path, file: &'a Path, registry: &'a mut NodeRegistry) -> Self {
        Self { project_root, file, registry, module_path: module_path_for_file(project_root, file), item_index: 0, parent_path: Vec::new(), current_impl: None }
    }

    fn register(&mut self, ident: &syn::Ident, kind: NodeKind) {
        let id = module_child_path(&self.module_path, ident.to_string());
        self.register_with_id(id, kind);
    }

    fn register_with_id(&mut self, id: String, kind: NodeKind) {
        let handle = node_handle(self.file, self.item_index, self.parent_path.clone(), kind);
        self.registry.insert_handle(normalize_symbol_id(&id), handle);
    }

    fn register_use_tree(&mut self, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Name(name) => {
                let id = format!("{}::use::{}", self.module_path, name.ident);
                self.register_with_id(id, NodeKind::Use);
            }
            syn::UseTree::Rename(rename) => {
                let id = format!("{}::use::{}", self.module_path, rename.rename);
                self.register_with_id(id, NodeKind::Use);
            }
            syn::UseTree::Path(path) => self.register_use_tree(&path.tree),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.register_use_tree(item);
                }
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> syn::visit::Visit<'ast> for NodeRegistryBuilder<'_> {
    fn visit_file(&mut self, i: &'ast syn::File) {
        self.module_path = module_path_for_file(self.project_root, self.file);
        for (idx, item) in i.items.iter().enumerate() {
            self.item_index = idx;
            self.visit_item(item);
        }
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.register(&i.sig.ident, NodeKind::Fn);
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.register(&i.ident, NodeKind::Struct);
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.register(&i.ident, NodeKind::Enum);
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.register(&i.ident, NodeKind::Trait);
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.register(&i.ident, NodeKind::Type);
        syn::visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.register(&i.ident, NodeKind::Const);
        syn::visit::visit_item_const(self, i);
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.register(&i.ident, NodeKind::Mod);
        if let Some((_brace, items)) = &i.content {
            let prev = self.module_path.clone();
            self.module_path = module_child_path(&prev, i.ident.to_string());
            let mod_index = self.item_index;
            self.parent_path.push(mod_index);
            for (idx, item) in items.iter().enumerate() {
                self.item_index = idx;
                self.visit_item(item);
            }
            self.parent_path.pop();
            self.module_path = prev;
        }
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        self.register_use_tree(&i.tree);
        syn::visit::visit_item_use(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let impl_index = self.item_index;
        let prev_impl = self.current_impl.clone();
        let struct_path = type_path_string(&i.self_ty, &self.module_path);
        let trait_path = i.trait_.as_ref().map(|(_, path, _)| path_to_string(path, &self.module_path));
        self.current_impl = Some(ImplContext { struct_path, trait_path });
        self.parent_path.push(impl_index);
        syn::visit::visit_item_impl(self, i);
        self.parent_path.pop();
        self.current_impl = prev_impl;
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if let Some(ctx) = &self.current_impl {
            let name = i.sig.ident.to_string();
            let id = if let Some(trait_path) = &ctx.trait_path { format!("{} as {}::{}", ctx.struct_path, trait_path, name) } else { format!("{}::{}", ctx.struct_path, name) };
            self.register_with_id(id, NodeKind::ImplFn);
        } else {
            self.register(&i.sig.ident, NodeKind::ImplFn);
        }
        syn::visit::visit_impl_item_fn(self, i);
    }
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}

#[derive(Debug, Clone)]
struct SnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}

impl SnapshotOracle {
    fn from_snapshot(snapshot: WireSnapshot) -> Self {
        let mut id_by_key = HashMap::new();
        let mut key_by_index = Vec::new();
        let mut macro_generated = HashSet::new();
        let mut crate_by_key = HashMap::new();
        let mut signature_by_key = HashMap::new();

        for node in snapshot.nodes.iter() {
            let key = node.key.to_string();
            let external_id = node.id.clone();
            id_by_key.insert(key.clone(), external_id.clone());
            key_by_index.push(key.clone());
            if is_macro_generated(&node.metadata) {
                macro_generated.insert(key.clone());
            }
            if let Some(crate_name) = node.metadata.get("crate").or_else(|| node.metadata.get("crate_name")).or_else(|| node.metadata.get("package")) {
                crate_by_key.insert(key.clone(), crate_name.clone());
            }
            if let Some(signature) = node.metadata.get("signature") {
                signature_by_key.insert(key.clone(), signature.clone());
            }
        }

        Self { snapshot, id_by_key, key_by_index, macro_generated, crate_by_key, signature_by_key }
    }
}

fn is_macro_generated(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let value = metadata.get("macro_generated").or_else(|| metadata.get("generated_by_macro")).or_else(|| metadata.get("macro")).or_else(|| metadata.get("is_macro"));
    matches!(value.map(|v| v.as_str()), Some("true") | Some("1") | Some("yes"))
}

impl StructuralEditOracle for SnapshotOracle {
    fn impact_of(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        let Some(external_id) = self.id_by_key.get(&symbol_id).cloned() else {
            return Vec::new();
        };
        let mut snapshot = self.snapshot.clone();
        let levels = snapshot.bfs_gpu(external_id);
        levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level < 0 {
                    return None;
                }
                let key = self.key_by_index.get(idx)?;
                if key == &symbol_id {
                    None
                } else {
                    Some(key.clone())
                }
            })
            .collect()
    }

    fn satisfies_bounds(&self, id: &str, new_sig: &syn::Signature) -> bool {
        let id = normalize_symbol_id(id);
        if let Some(sig) = self.signature_by_key.get(&id) {
            let new_sig = quote::quote!(# new_sig).to_string();
            return sig == &new_sig;
        }
        true
    }

    fn is_macro_generated(&self, symbol_id: &str) -> bool {
        let symbol_id = normalize_symbol_id(symbol_id);
        self.macro_generated.contains(&symbol_id)
    }

    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        let Some(symbol_crate) = self.crate_by_key.get(&symbol_id) else {
            return Vec::new();
        };
        let Some(external_id) = self.id_by_key.get(&symbol_id).cloned() else {
            return Vec::new();
        };
        let mut snapshot = self.snapshot.clone();
        let levels = snapshot.bfs_gpu(external_id);
        levels
            .iter()
            .enumerate()
            .filter_map(|(idx, level)| {
                if *level < 0 {
                    return None;
                }
                let key = self.key_by_index.get(idx)?;
                if key == &symbol_id {
                    return None;
                }
                let other_crate = self.crate_by_key.get(key)?;
                if other_crate != symbol_crate {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}
