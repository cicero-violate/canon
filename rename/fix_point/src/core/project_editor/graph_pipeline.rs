use crate::core::project_editor::invariants::{
    assert_edge_endpoints_exist, assert_module_path_consistency,
    assert_no_duplicate_edges, assert_unique_def_paths, assert_unique_node_ids,
    assert_unique_node_keys,
};


use crate::core::project_editor::refactor::MoveSet;


use crate::core::symbol_id::normalize_symbol_id;


use crate::module_path::ModulePath;


use crate::state::NodeRegistry;


use crate::structured::ast_render::render_node;


use anyhow::{anyhow, Result};


use database::graph_log::{GraphSnapshot, WireNode, WireNodeId};


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use std::sync::Arc;


pub(crate) struct EmissionReport {
    pub written: Vec<PathBuf>,
    pub unchanged: Vec<PathBuf>,
    pub deletion_candidates: Vec<PathBuf>,
}


pub(crate) struct FilePlan {
    pub path: PathBuf,
    pub content: String,
}


pub(crate) struct Plan1 {
    pub files: Vec<FilePlan>,
}


fn strip_leading_visibility(snippet: &str) -> String {
    let s = snippet.trim_start();
    if let Some(rest) = s.strip_prefix("pub ") {
        return rest.trim_start().to_string();
    }
    if let Some(rest) = s.strip_prefix("pub(") {
        if let Some(idx) = rest.find(')') {
            let after = &rest[idx + 1..];
            return after.trim_start().to_string();
        }
    }
    s.to_string()
}
