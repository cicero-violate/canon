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
