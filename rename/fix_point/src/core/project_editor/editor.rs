use super::cross_file::{apply_cross_file_moves, collect_new_files};


use super::graph_pipeline::{
    apply_moves_to_snapshot, compare_snapshots, emit_plan, project_plan,
    rebuild_graph_snapshot, rollback_emission,
};


use super::model_validation::validate_model0;


use super::ops::apply_node_op;


use super::oracle::GraphSnapshotOracle;


use super::propagate::{apply_rewrites, build_symbol_index_and_occurrences, propagate};


use super::refactor::{
    run_pass1_canonical_rewrite, run_pass2_scope_rehydration, run_pass3_orphan_cleanup,
    MoveSet,
};


use super::registry_builder::{NodeRegistryBuilder, SpanLookup, SpanOverride};


use super::use_path::run_use_path_rewrite;


use super::utils::{build_symbol_index, find_project_root};


use crate::core::mod_decls::update_mod_declarations;


use crate::core::oracle::StructuralEditOracle;


use crate::core::paths::module_path_for_file;


use crate::core::symbol_id::normalize_symbol_id;


use crate::fs;


use crate::model::types::{FileRename, LineColumn, SpanRange};


use crate::state::NodeRegistry;


use crate::structured::{FieldMutation, NodeOp};


use anyhow::{Context, Result};


use compiler_capture::cargo_project::CargoProject;


use compiler_capture::multi_capture::capture_project;


use compiler_capture::rustc::RustcFrontend;


use database::graph_log::GraphSnapshot;


use kernel::kernel::{Kernel as MemoryEngine, MemoryEngineConfig};


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use std::sync::Arc;


use syn::visit::Visit;


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


pub struct ProjectEditor {
    pub registry: NodeRegistry,
    pub changesets: HashMap<PathBuf, Vec<QueuedOp>>,
    pub oracle: Box<dyn StructuralEditOracle>,
    pub original_sources: HashMap<PathBuf, String>,
    project_root: PathBuf,
    model0: Option<GraphSnapshot>,
    span_lookup: Option<SpanLookup>,
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
