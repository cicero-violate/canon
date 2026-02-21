impl Default for QueryRequest {
    fn default() -> Self {
        Self {
            kinds: Vec::new(),
            module_prefix: None,
            name_contains: None,
        }
    }
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
        Ok(MutationResult {
            renamed: self.renames.len(),
            dry_run: self.dry_run,
            preview_path: self.preview_path,
        })
    }
}


impl Default for UpsertRequest {
    fn default() -> Self {
        Self {
            edits: Vec::new(),
            node_ops: Vec::new(),
            format: true,
        }
    }
}


fn default_true() -> bool {
    true
}


pub fn execute_query_json(project: &Path, payload: &str) -> Result<String> {
    let request: QueryRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}


pub fn execute_mutation_json(project: &Path, payload: &str) -> Result<String> {
    let request: MutationRequest = serde_json::from_str(payload)?;
    let response = request.execute(project)?;
    Ok(serde_json::to_string_pretty(&response)?)
}


pub fn execute_upsert_json(payload: &str) -> Result<String> {
    let request: UpsertRequest = serde_json::from_str(payload)?;
    let response = request.execute()?;
    Ok(serde_json::to_string_pretty(&response)?)
}


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


/// Result of executing a `QueryRequest`.
#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
}


/// Result of executing a `QueryRequest`.
#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
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


/// Result of executing a `QueryRequest`.
#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
}


/// Result of executing a `QueryRequest`.
#[derive(serde::Serialize)]
pub struct QueryResult {
    /// Full names report (symbols, occurrences, alias graph, etc.)
    pub report: SymbolIndexReport,
    /// Symbols that matched the provided filters.
    pub matches: Vec<SymbolRecord>,
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


/// Serializable response describing mutation outcome.
#[derive(serde::Serialize)]
pub struct MutationResult {
    pub renamed: usize,
    pub dry_run: bool,
    pub preview_path: Option<PathBuf>,
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


/// Serializable response describing mutation outcome.
#[derive(serde::Serialize)]
pub struct MutationResult {
    pub renamed: usize,
    pub dry_run: bool,
    pub preview_path: Option<PathBuf>,
}


/// Serializable response describing mutation outcome.
#[derive(serde::Serialize)]
pub struct MutationResult {
    pub renamed: usize,
    pub dry_run: bool,
    pub preview_path: Option<PathBuf>,
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


/// Serializable result describing applied edits.
#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}


/// Serializable result describing applied edits.
#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}


/// Serializable result describing applied edits.
#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}


/// Serializable result describing applied edits.
#[derive(serde::Serialize)]
pub struct UpsertResult {
    pub touched_files: Vec<PathBuf>,
}
