//! High-level user API for querying and mutating Rust projects.
//!
//! This module exposes a small, GraphQL-inspired surface with three verbs:
//! - **query**: inspect the symbol graph collected by `semantic-lint names`
//! - **mutate**: rename symbols by providing an explicit `id -> new_name` map
//! - **upsert**: insert or replace AST fragments via the structured rewrite layer
//!
//! The goal is to let callers script transformations without touching the lower-level
//! rename pipeline directly.
use super::core::{apply_rename_with_map, collect_names, SymbolIndexReport, SymbolRecord};
use super::structured::{apply_ast_rewrites, AstEdit, NodeOp};
use anyhow::{bail, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
/// Serializable request for querying symbol metadata.
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
impl Default for QueryRequest {
    fn default() -> Self {
        Self { kinds: Vec::new(), module_prefix: None, name_contains: None }
    }
}
/// Result of executing a `QueryRequest`.
#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
}
impl QueryRequest {
    /// Build a new query with no filters (matches everything).
    pub fn new() -> Self {
        Self::default()
    }
    /// Restrict the query to a single symbol `kind` (e.g. "function", "struct").
    pub fn kind(mut self, kind: impl Into<String>) -> Self {
        self.kinds.push(kind.into());
        self
    }
    /// Require that matched symbols live under a module prefix (e.g. `crate::rename`).
    pub fn module_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.module_prefix = Some(prefix.into());
        self
    }
    /// Require that symbol names contain a substring (case-insensitive).
    pub fn name_contains(mut self, needle: impl Into<String>) -> Self {
        self.name_contains = Some(needle.into());
        self
    }
    /// Execute the query against a project root.
    pub fn execute(&self, project: &Path) -> Result<QueryResult> {
        let report = collect_names(project)?;
        let matches = report.symbols.iter().filter(|sym| self.matches_symbol(sym)).cloned().collect();
        Ok(QueryResult { report, matches })
    }
    fn matches_symbol(&self, symbol: &SymbolRecord) -> bool {
        if !self.kinds.is_empty() && !self.kinds.iter().any(|k| k == &symbol.kind) {
            return false;
        }
        if let Some(prefix) = &self.module_prefix {
            if !symbol.module.starts_with(prefix) {
                return false;
            }
        }
        if let Some(needle) = &self.name_contains {
            let target = needle.to_lowercase();
            if !symbol.name.to_lowercase().contains(&target) {
                return false;
            }
        }
        true
    }
}
/// Serializable request describing a rename mutation.
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
/// Serializable response describing mutation outcome.
#[derive(serde::Serialize)]
pub struct MutationResult {
    pub renamed: usize,
    pub dry_run: bool,
    pub preview_path: Option<PathBuf>,
}
impl MutationRequest {
    /// Create an empty mutation.
    pub fn new() -> Self {
        Self::default()
    }
    /// Append a rename mapping (symbol id -> new name).
    pub fn rename(mut self, id: impl Into<String>, new_name: impl Into<String>) -> Self {
        self.renames.insert(id.into(), new_name.into());
        self
    }
    /// Extend the rename mapping with an iterator of pairs.
    pub fn extend<I, K, V>(mut self, entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (k, v) in entries {
            self.renames.insert(k.into(), v.into());
        }
        self
    }
    /// Enable or disable dry-run mode (defaults to `false`).
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }
    /// Write rename previews to a custom path (only used when `dry_run` is true).
    pub fn preview_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.preview_path = Some(path.into());
        self
    }
    /// Execute the mutation against a project root.
    pub fn execute(self, project: &Path) -> Result<MutationResult> {
        if self.renames.is_empty() {
            bail!("mutation contains no rename mappings");
        }
        let out_path = self.preview_path.as_deref();
        apply_rename_with_map(project, &self.renames, self.dry_run, out_path)?;
        Ok(MutationResult { renamed: self.renames.len(), dry_run: self.dry_run, preview_path: self.preview_path })
    }
}
/// Serializable upsert request (AST edits).
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
impl Default for UpsertRequest {
    fn default() -> Self {
        Self { edits: Vec::new(), node_ops: Vec::new(), format: true }
    }
}
/// Serializable result describing applied edits.
#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}
impl UpsertRequest {
    /// Create an empty upsert request.
    pub fn new() -> Self {
        Self::default()
    }
    /// Queue an AST edit generated via `AstEdit::insert`/`replace`.
    pub fn edit(mut self, edit: AstEdit) -> Self {
        self.edits.push(edit);
        self
    }
    /// Queue a structural node operation.
    pub fn push_node_op(mut self, op: NodeOp) -> Self {
        self.node_ops.push(op);
        self
    }
    /// Configure whether `rustfmt` should run on touched files (default: true).
    pub fn format(mut self, enabled: bool) -> Self {
        self.format = enabled;
        self
    }
    /// Execute the queued edits.
    pub fn execute(self) -> Result<UpsertResult> {
        if self.edits.is_empty() && self.node_ops.is_empty() {
            bail!("no edits were queued for upsert");
        }
        let touched = if self.edits.is_empty() { Vec::new() } else { apply_ast_rewrites(&self.edits, self.format)? };
        Ok(UpsertResult { touched_files: touched })
    }
}
fn default_true() -> bool {
    true
}
/// Convenience helper: execute a serialized query request and return JSON.
pub fn execute_query_json(project: &Path, payload: &str) -> Result<String> {
    let request: QueryRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}
/// Convenience helper: execute a mutation described via JSON.
pub fn execute_mutation_json(project: &Path, payload: &str) -> Result<String> {
    let request: MutationRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}
/// Convenience helper: execute an upsert request encoded as JSON.
pub fn execute_upsert_json(payload: &str) -> Result<String> {
    let request: UpsertRequest = serde_json::from_str(payload)?;
    let response = request.execute()?;
    Ok(serde_json::to_string_pretty(&response)?)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    #[test]
    fn defaults_are_sensible() {
        let query = QueryRequest::default();
        assert!(query.kinds.is_empty());
        assert!(query.module_prefix.is_none());
        assert!(query.name_contains.is_none());
        let mutation = MutationRequest::default();
        assert!(mutation.renames.is_empty());
        assert!(!mutation.dry_run);
        assert!(mutation.preview_path.is_none());
        let upsert = UpsertRequest::default();
        assert!(upsert.edits.is_empty());
        assert!(upsert.format);
    }
    #[test]
    fn upsert_json_executes() {
        use syn::parse_quote;
        let dir = tempdir().unwrap();
        let file = dir.path().join("demo.rs");
        fs::write(&file, "fn existing() {}\n").unwrap();
        let helper: syn::ItemFn = parse_quote! {
            fn added() {}
        };
        let edit = AstEdit::insert(&file, 0, &helper);
        let request = UpsertRequest { edits: vec![edit], node_ops: Vec::new(), format: false };
        let payload = serde_json::to_string(&request).unwrap();
        let response_raw = execute_upsert_json(&payload).unwrap();
        let response: serde_json::Value = serde_json::from_str(&response_raw).unwrap();
        assert_eq!(response["touched_files"].as_array().map(|a| a.len()), Some(1));
        let contents = fs::read_to_string(&file).unwrap();
        assert!(contents.contains("added"));
    }
    #[test]
    fn query_request_serializes() {
        let request = QueryRequest::new().kind("struct").module_prefix("crate::rename").name_contains("Buffer");
        let json = serde_json::to_string(&request).unwrap();
        let parsed: QueryRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kinds, vec!["struct"]);
        assert_eq!(parsed.module_prefix.as_deref(), Some("crate::rename"));
    }
}
