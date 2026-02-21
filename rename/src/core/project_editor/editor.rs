use super::cross_file::{apply_cross_file_moves, collect_new_files};
use super::ops::apply_node_op;
use super::oracle::GraphSnapshotOracle;
use super::propagate::{apply_rewrites, propagate};
use super::registry_builder::NodeRegistryBuilder;
use super::use_path::run_use_path_rewrite;
use super::utils::{build_symbol_index, find_project_root};
use crate::core::mod_decls::update_mod_declarations;
use crate::core::oracle::StructuralEditOracle;
use crate::core::symbol_id::normalize_symbol_id;
use crate::fs;
use crate::model::types::FileRename;
use crate::state::NodeRegistry;
use crate::structured::{FieldMutation, NodeOp};
use anyhow::{Context, Result};
use compiler_capture::frontends::rustc::RustcFrontend;
use compiler_capture::multi_capture::capture_project;
use compiler_capture::project::CargoProject;
use database::{MemoryEngine, MemoryEngineConfig};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use syn::visit::Visit;

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
    pending_new_files: Vec<(PathBuf, String)>,
    last_touched_files: HashSet<PathBuf>,
}

#[derive(Clone)]
pub(crate) struct QueuedOp {
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
        Ok(Self {
            registry,
            changesets: HashMap::new(),
            oracle,
            original_sources,
            pending_file_moves: Vec::new(),
            pending_file_renames: Vec::new(),
            pending_new_files: Vec::new(),
            last_touched_files: HashSet::new(),
        })
    }

    pub fn load_with_rustc(project: &Path) -> Result<Self> {
        let cargo = CargoProject::from_entry(project)?;
        let frontend = RustcFrontend::new();
        let _artifacts =
            capture_project(&frontend, &cargo, &[]).with_context(|| format!("rustc capture failed for {}", project.display()))?;
        let state_dir = cargo.workspace_root().join(".rename");
        std::fs::create_dir_all(&state_dir)?;
        let tlog_path = state_dir.join("state.tlog");
        let engine = MemoryEngine::new(MemoryEngineConfig { tlog_path })?;
        let snapshot = engine.materialized_graph()?;
        let oracle = Box::new(GraphSnapshotOracle::from_snapshot(snapshot));
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
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. } => handle.file.clone(),
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
        for (_file, ops) in &self.changesets {
            for queued in ops {
                let prop = propagate(&queued.op, &queued.symbol_id, &self.registry, &*self.oracle)?;
                rewrites.extend(prop.rewrites);
                conflicts.extend(prop.conflicts);
                file_renames.extend(prop.file_renames);
            }
        }
        let rewrite_touched = apply_rewrites(&mut self.registry, &rewrites)?;
        touched_files.extend(rewrite_touched);
        for (file, ops) in &self.changesets {
            for queued in ops {
                let changed = {
                    let ast = self.registry.asts.get_mut(file).with_context(|| format!("missing AST for {}", file.display()))?;
                    apply_node_op(ast, &handle_snapshot, &queued.symbol_id, &queued.op)
                        .with_context(|| format!("failed to apply {}", queued.symbol_id))?
                };
                if changed {
                    touched_files.insert(file.clone());
                }
            }
        }
        let use_path_touched = run_use_path_rewrite(&mut self.registry, &self.changesets)?;
        touched_files.extend(use_path_touched);
        let cross_file_touched = apply_cross_file_moves(&mut self.registry, &self.changesets)?;
        touched_files.extend(cross_file_touched);
        self.pending_new_files = collect_new_files(&self.registry, &self.changesets);
        let mut validation = self.validate()?;
        validation.extend(conflicts);
        self.pending_file_moves = file_renames.iter().map(|r| (PathBuf::from(&r.from), PathBuf::from(&r.to))).collect();
        self.pending_file_renames = file_renames.clone();
        self.last_touched_files = touched_files.clone();
        Ok(ChangeReport {
            touched_files: touched_files.into_iter().collect(),
            conflicts: validation,
            file_moves: self.pending_file_moves.clone(),
        })
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
                            conflicts.push(EditConflict {
                                symbol_id: queued.symbol_id.clone(),
                                reason: "signature change may require updating call sites".to_string(),
                            });
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
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)?;
            }
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
        if !self.pending_new_files.is_empty() {
            if let Some(project_root) = find_project_root(&self.registry)? {
                let symbol_table = build_symbol_index(&project_root, &self.registry)?;
                let mut touched = HashSet::new();
                let mut synthetic_renames: Vec<FileRename> = Vec::new();
                for (new_path, new_module_id) in &self.pending_new_files {
                    if let Some(ast) = self.registry.asts.get(new_path) {
                        if let Some(parent) = new_path.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        let rendered = crate::structured::render_file(ast);
                        std::fs::write(new_path, &rendered)?;
                        written.push(new_path.clone());
                    }
                    let parts: Vec<&str> = new_module_id.split("::").collect();
                    if parts.len() >= 2 {
                        synthetic_renames.push(FileRename {
                            from: String::new(),
                            to: new_path.to_string_lossy().to_string(),
                            is_directory_move: false,
                            old_module_id: format!("crate::__new__::{}", parts.last().unwrap()),
                            new_module_id: new_module_id.clone(),
                        });
                    }
                }
                update_mod_declarations(&project_root, &symbol_table, &synthetic_renames, &mut touched)?;
                written.extend(touched.into_iter());
            }
        }
        written.sort();
        written.dedup();
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
                let diff = similar::TextDiff::from_lines(original, &rendered)
                    .unified_diff()
                    .header(&format!("{} (original)", file.display()), &format!("{} (updated)", file.display()))
                    .to_string();
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

    /// DEBUG: return all registered symbol IDs currently indexed.
    pub fn debug_list_symbol_ids(&self) -> Vec<String> {
        self.registry.handles.keys().cloned().collect()
    }
}
