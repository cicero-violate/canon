use super::core::{apply_rename_with_map, collect_names, SymbolIndexReport, SymbolRecord};


use super::structured::{apply_ast_rewrites, AstEdit, NodeOp};


use anyhow::{bail, Result};


use std::collections::HashMap;


use std::path::{Path, PathBuf};


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct MutationRequest {
    /// Symbol identifier → new name mapping.
    #[serde(default)]
    pub renames: HashMap<String, String>,
    /// Whether to run in dry-run mode.
    #[serde(default)]
    pub dry_run: bool,
    /// Optional preview path (used when `dry_run = true`).
    #[serde(default)]
    pub preview_path: Option<PathBuf>,
}


#[derive(serde::Serialize)]
pub struct MutationResult {
    pub renamed: usize,
    pub dry_run: bool,
    pub preview_path: Option<PathBuf>,
}


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryRequest {
    /// Optional list of symbol kinds to match (e.g., `["function","struct"]`).
    #[serde(default)]
    pub kinds: Vec<String>,
    /// Optional module prefix filter (e.g., `crate::rename`).
    #[serde(default)]
    pub module_prefix: Option<String>,
    /// Optional substring filter applied to symbol names (case-insensitive).
    #[serde(default)]
    pub name_contains: Option<String>,
}


#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
}


#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UpsertRequest {
    /// Concrete AST edits to apply.
    #[serde(default)]
    pub edits: Vec<AstEdit>,
    /// Node operations for structural edits.
    #[serde(skip, default)]
    pub node_ops: Vec<NodeOp>,
    /// Whether to run rustfmt on touched files.
    #[serde(default = "default_true")]
    pub format: bool,
}


#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct QueryRequest {
    /// Optional list of symbol kinds to match (e.g., `["function","struct"]`).
    #[serde(default)]
    pub kinds: Vec<String>,
    /// Optional module prefix filter (e.g., `crate::rename`).
    #[serde(default)]
    pub module_prefix: Option<String>,
    /// Optional substring filter applied to symbol names (case-insensitive).
    #[serde(default)]
    pub name_contains: Option<String>,
}


fn default_true() -> bool {
    true
}


pub fn execute_mutation_json(project: &Path, payload: &str) -> Result<String> {
    let request: MutationRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}


pub fn execute_query_json(project: &Path, payload: &str) -> Result<String> {
    let request: QueryRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}


pub fn execute_upsert_json(payload: &str) -> Result<String> {
    let request: UpsertRequest = serde_json::from_str(payload)?;
    let response = request.execute()?;
    Ok(serde_json::to_string_pretty(&response)?)
}
