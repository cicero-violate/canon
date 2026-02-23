use super::EditConflict;


use crate::alias::VisibilityLeakAnalysis;


use crate::alias::{AliasGraph, VisibilityScope};


use crate::core::collect::{add_file_module_symbol, collect_symbols};


use crate::core::oracle::StructuralEditOracle;


use crate::core::paths::module_path_for_file;


use crate::core::paths::plan_file_renames;


use crate::core::rename::apply_symbol_edits_to_ast;


use crate::core::symbol_id::normalize_symbol_id;


use crate::core::use_map::build_use_map;


use crate::model::types::{FileRename, SymbolEdit, SymbolIndex, SymbolOccurrence};


use crate::module_path::{ModuleMovePlan, ModulePath};


use crate::occurrence::OccurrenceVisitor;


use crate::state::NodeRegistry;


use crate::structured::{FieldMutation, NodeOp};


use anyhow::Result;


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use syn::visit::Visit;


pub struct PropagationResult {
    pub rewrites: Vec<SymbolEdit>,
    pub conflicts: Vec<EditConflict>,
    pub file_renames: Vec<crate::model::types::FileRename>,
}
