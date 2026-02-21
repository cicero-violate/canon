pub mod alias;

pub mod api;

pub mod attributes;

pub mod compiler_capture;

pub mod core;

pub mod fs;

pub mod macros;

pub mod model;

pub mod module_path;

pub mod occurrence;

pub mod pattern;

pub mod resolve;

pub mod scope;

pub mod state;

pub mod structured;

pub use crate::core::{apply_rename, apply_rename_with_map, collect_names, emit_names};


#[derive(Debug, Default, Clone)]
pub struct AliasGraph {
    /// All use nodes indexed by ID
    nodes: HashMap<String, ImportNode>,
    /// Map from local name to use node ID (for resolution)
    /// Key: (module_path, local_name) -> use_node_id
    local_names: HashMap<(String, String), String>,
    /// Map from source path to all use node IDs that import it
    /// Key: source_path -> Vec<use_node_id>
    source_imports: HashMap<String, Vec<String>>,
    /// Glob imports indexed by module
    /// Key: module_path -> Vec<(source_path, use_node_id)>
    glob_imports: HashMap<String, Vec<(String, String)>>,
    /// Edges representing alias relationships
    edges: Vec<AliasEdge>,
}


#[derive(Debug, Default, Clone)]
pub struct AliasGraph {
    /// All use nodes indexed by ID
    nodes: HashMap<String, ImportNode>,
    /// Map from local name to use node ID (for resolution)
    /// Key: (module_path, local_name) -> use_node_id
    local_names: HashMap<(String, String), String>,
    /// Map from source path to all use node IDs that import it
    /// Key: source_path -> Vec<use_node_id>
    source_imports: HashMap<String, Vec<String>>,
    /// Glob imports indexed by module
    /// Key: module_path -> Vec<(source_path, use_node_id)>
    glob_imports: HashMap<String, Vec<(String, String)>>,
    /// Edges representing alias relationships
    edges: Vec<AliasEdge>,
}


#[derive(Debug, Default, Clone)]
pub struct AliasGraph {
    /// All use nodes indexed by ID
    nodes: HashMap<String, ImportNode>,
    /// Map from local name to use node ID (for resolution)
    /// Key: (module_path, local_name) -> use_node_id
    local_names: HashMap<(String, String), String>,
    /// Map from source path to all use node IDs that import it
    /// Key: source_path -> Vec<use_node_id>
    source_imports: HashMap<String, Vec<String>>,
    /// Glob imports indexed by module
    /// Key: module_path -> Vec<(source_path, use_node_id)>
    glob_imports: HashMap<String, Vec<(String, String)>>,
    /// Edges representing alias relationships
    edges: Vec<AliasEdge>,
}


#[derive(Debug, Clone, Serialize)]
pub struct AliasEdge {
    /// Source use node ID
    pub from: String,
    /// Target symbol or use node ID
    pub to: String,
    /// Type of edge
    pub kind: EdgeKind,
}


#[derive(Debug, Clone, Serialize)]
pub struct AliasEdge {
    /// Source use node ID
    pub from: String,
    /// Target symbol or use node ID
    pub to: String,
    /// Type of edge
    pub kind: EdgeKind,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum EdgeKind {
    /// Direct import: A imports B
    Import,
    /// Re-export: A re-exports B
    ReExport,
    /// Alias: A is an alias for B
    Alias,
    /// Transitive: A transitively refers to B through intermediaries
    Transitive,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum EdgeKind {
    /// Direct import: A imports B
    Import,
    /// Re-export: A re-exports B
    ReExport,
    /// Alias: A is an alias for B
    Alias,
    /// Transitive: A transitively refers to B through intermediaries
    Transitive,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum EdgeKind {
    /// Direct import: A imports B
    Import,
    /// Re-export: A re-exports B
    ReExport,
    /// Alias: A is an alias for B
    Alias,
    /// Transitive: A transitively refers to B through intermediaries
    Transitive,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum EdgeKind {
    /// Direct import: A imports B
    Import,
    /// Re-export: A re-exports B
    ReExport,
    /// Alias: A is an alias for B
    Alias,
    /// Transitive: A transitively refers to B through intermediaries
    Transitive,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum EdgeKind {
    /// Direct import: A imports B
    Import,
    /// Re-export: A re-exports B
    ReExport,
    /// Alias: A is an alias for B
    Alias,
    /// Transitive: A transitively refers to B through intermediaries
    Transitive,
}


#[derive(Debug, Clone, Serialize)]
pub struct ExposurePath {
    /// The symbol's original module
    pub origin_module: String,
    /// Modules that re-export this symbol
    pub reexport_chain: Vec<String>,
    /// Final visibility level
    pub visibility: VisibilityScope,
}


#[derive(Debug, Clone, Serialize)]
pub struct ExposurePath {
    /// The symbol's original module
    pub origin_module: String,
    /// Modules that re-export this symbol
    pub reexport_chain: Vec<String>,
    /// Final visibility level
    pub visibility: VisibilityScope,
}


#[derive(Debug, Clone, Serialize)]
pub struct ImportNode {
    /// Unique identifier for this use statement
    pub id: String,
    /// The module where this use statement appears
    pub module_path: String,
    /// Source path being imported (e.g., "std::collections::HashMap")
    pub source_path: String,
    /// Local name (after 'as' if present, otherwise last segment)
    pub local_name: String,
    /// Original name (before 'as' if present)
    pub original_name: Option<String>,
    /// Type of use statement
    pub kind: UseKind,
    /// Visibility of this use statement
    pub visibility: VisibilityScope,
    /// File where this use appears
    pub file: String,
}


#[derive(Debug, Clone, Serialize)]
pub struct ImportNode {
    /// Unique identifier for this use statement
    pub id: String,
    /// The module where this use statement appears
    pub module_path: String,
    /// Source path being imported (e.g., "std::collections::HashMap")
    pub source_path: String,
    /// Local name (after 'as' if present, otherwise last segment)
    pub local_name: String,
    /// Original name (before 'as' if present)
    pub original_name: Option<String>,
    /// Type of use statement
    pub kind: UseKind,
    /// Visibility of this use statement
    pub visibility: VisibilityScope,
    /// File where this use appears
    pub file: String,
}


#[derive(Debug, Clone, Serialize)]
pub struct LeakedSymbol {
    /// Symbol identifier
    pub symbol_id: String,
    /// Original visibility
    pub original_visibility: VisibilityScope,
    /// Module where it's leaked to
    pub leaked_to: String,
    /// How it was leaked (re-export chain)
    pub leak_chain: Vec<String>,
}


#[derive(Debug, Clone, Serialize)]
pub struct LeakedSymbol {
    /// Symbol identifier
    pub symbol_id: String,
    /// Original visibility
    pub original_visibility: VisibilityScope,
    /// Module where it's leaked to
    pub leaked_to: String,
    /// How it was leaked (re-export chain)
    pub leak_chain: Vec<String>,
}


#[derive(Debug, Clone)]
pub struct ResolutionChain {
    /// Starting identifier (as used in code)
    pub start_name: String,
    /// Module where the identifier is used
    pub start_module: String,
    /// Steps in the resolution chain
    pub steps: Vec<ResolutionStep>,
    /// Final resolved symbol ID
    pub resolved_symbol: Option<String>,
}


#[derive(Debug, Clone)]
pub struct ResolutionChain {
    /// Starting identifier (as used in code)
    pub start_name: String,
    /// Module where the identifier is used
    pub start_module: String,
    /// Steps in the resolution chain
    pub steps: Vec<ResolutionStep>,
    /// Final resolved symbol ID
    pub resolved_symbol: Option<String>,
}


#[derive(Debug, Clone)]
pub struct ResolutionStep {
    /// Type of resolution step
    pub kind: StepKind,
    /// Name at this step
    pub name: String,
    /// Module context at this step
    pub module: String,
    /// Associated use node if applicable
    pub use_node_id: Option<String>,
}


#[derive(Debug, Clone)]
pub struct ResolutionStep {
    /// Type of resolution step
    pub kind: StepKind,
    /// Name at this step
    pub name: String,
    /// Module context at this step
    pub module: String,
    /// Associated use node if applicable
    pub use_node_id: Option<String>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Starting point
    Start,
    /// Local use statement
    LocalUse,
    /// Re-export traversal
    ReExport,
    /// Glob import resolution
    GlobImport,
    /// Direct symbol lookup
    DirectLookup,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Starting point
    Start,
    /// Local use statement
    LocalUse,
    /// Re-export traversal
    ReExport,
    /// Glob import resolution
    GlobImport,
    /// Direct symbol lookup
    DirectLookup,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Starting point
    Start,
    /// Local use statement
    LocalUse,
    /// Re-export traversal
    ReExport,
    /// Glob import resolution
    GlobImport,
    /// Direct symbol lookup
    DirectLookup,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Starting point
    Start,
    /// Local use statement
    LocalUse,
    /// Re-export traversal
    ReExport,
    /// Glob import resolution
    GlobImport,
    /// Direct symbol lookup
    DirectLookup,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepKind {
    /// Starting point
    Start,
    /// Local use statement
    LocalUse,
    /// Re-export traversal
    ReExport,
    /// Glob import resolution
    GlobImport,
    /// Direct symbol lookup
    DirectLookup,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum UseKind {
    /// Simple import: use foo::Bar;
    Simple,
    /// Aliased import: use foo::Bar as Baz;
    Aliased,
    /// Glob import: use foo::*;
    Glob,
    /// Re-export: pub use foo::Bar;
    ReExport,
    /// Aliased re-export: pub use foo::Bar as Baz;
    ReExportAliased,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum UseKind {
    /// Simple import: use foo::Bar;
    Simple,
    /// Aliased import: use foo::Bar as Baz;
    Aliased,
    /// Glob import: use foo::*;
    Glob,
    /// Re-export: pub use foo::Bar;
    ReExport,
    /// Aliased re-export: pub use foo::Bar as Baz;
    ReExportAliased,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum UseKind {
    /// Simple import: use foo::Bar;
    Simple,
    /// Aliased import: use foo::Bar as Baz;
    Aliased,
    /// Glob import: use foo::*;
    Glob,
    /// Re-export: pub use foo::Bar;
    ReExport,
    /// Aliased re-export: pub use foo::Bar as Baz;
    ReExportAliased,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum UseKind {
    /// Simple import: use foo::Bar;
    Simple,
    /// Aliased import: use foo::Bar as Baz;
    Aliased,
    /// Glob import: use foo::*;
    Glob,
    /// Re-export: pub use foo::Bar;
    ReExport,
    /// Aliased re-export: pub use foo::Bar as Baz;
    ReExportAliased,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum UseKind {
    /// Simple import: use foo::Bar;
    Simple,
    /// Aliased import: use foo::Bar as Baz;
    Aliased,
    /// Glob import: use foo::*;
    Glob,
    /// Re-export: pub use foo::Bar;
    ReExport,
    /// Aliased re-export: pub use foo::Bar as Baz;
    ReExportAliased,
}


#[derive(Debug, Clone, Serialize)]
pub struct VisibilityLeakAnalysis {
    /// Symbols that are publicly exposed from each module
    /// Key: module_path -> Vec<(symbol_name, exposure_path)>
    pub public_symbols: HashMap<String, Vec<(String, ExposurePath)>>,
    /// Symbols with restricted visibility
    pub restricted_symbols: HashMap<String, Vec<(String, VisibilityScope)>>,
    /// Re-export chains that leak private symbols
    pub leaked_private_symbols: Vec<LeakedSymbol>,
}


#[derive(Debug, Clone, Serialize)]
pub struct VisibilityLeakAnalysis {
    /// Symbols that are publicly exposed from each module
    /// Key: module_path -> Vec<(symbol_name, exposure_path)>
    pub public_symbols: HashMap<String, Vec<(String, ExposurePath)>>,
    /// Symbols with restricted visibility
    pub restricted_symbols: HashMap<String, Vec<(String, VisibilityScope)>>,
    /// Re-export chains that leak private symbols
    pub leaked_private_symbols: Vec<LeakedSymbol>,
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VisibilityScope {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or private
    Private,
    /// pub(in path)
    Restricted(String),
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VisibilityScope {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or private
    Private,
    /// pub(in path)
    Restricted(String),
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VisibilityScope {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or private
    Private,
    /// pub(in path)
    Restricted(String),
}


impl From<&Visibility> for VisibilityScope {
    fn from(vis: &Visibility) -> Self {
        match vis {
            Visibility::Public(_) => VisibilityScope::Public,
            Visibility::Restricted(restricted) => {
                if restricted.path.is_ident("crate") {
                    VisibilityScope::Crate
                } else if restricted.path.is_ident("super") {
                    VisibilityScope::Super
                } else if restricted.path.is_ident("self") {
                    VisibilityScope::Private
                } else {
                    VisibilityScope::Restricted(
                        restricted
                            .path
                            .segments
                            .iter()
                            .map(|s| s.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::"),
                    )
                }
            }
            Visibility::Inherited => VisibilityScope::Private,
        }
    }
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VisibilityScope {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or private
    Private,
    /// pub(in path)
    Restricted(String),
}


#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum VisibilityScope {
    /// pub
    Public,
    /// pub(crate)
    Crate,
    /// pub(super)
    Super,
    /// pub(self) or private
    Private,
    /// pub(in path)
    Restricted(String),
}


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
        Self {
            kinds: Vec::new(),
            module_prefix: None,
            name_contains: None,
        }
    }
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
        Self {
            edits: Vec::new(),
            node_ops: Vec::new(),
            format: true,
        }
    }
}


#[derive(Clone)]
pub struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


#[derive(Debug)]
pub struct ModuleRenamePlan {
    old_name: String,
    new_name: String,
    old_parent: String,
    new_parent: String,
}


#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
    pub file_moves: Vec<(PathBuf, PathBuf)>,
}


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


#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}


#[derive(Clone)]
pub struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}


impl StructuralEditOracle for GraphSnapshotOracle {
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
                if key == &symbol_id { None } else { Some(key.clone()) }
            })
            .collect()
    }
    fn satisfies_bounds(&self, id: &str, new_sig: &Signature) -> bool {
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
                if other_crate != symbol_crate { Some(key.clone()) } else { None }
            })
            .collect()
    }
}


#[derive(Debug, Clone)]
pub struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


impl StructuralEditOracle for NullOracle {
    fn impact_of(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
    fn satisfies_bounds(&self, _id: &str, _new_sig: &Signature) -> bool {
        true
    }
    fn is_macro_generated(&self, _symbol_id: &str) -> bool {
        false
    }
    fn cross_crate_users(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
}


#[derive(Debug, Clone, Default)]
pub struct NullOracle;


#[derive(Debug, Clone, Default)]
pub struct NullOracle;


#[derive(Debug, Clone, Default)]
pub struct NullOracle;


impl VisitMut for CanonicalRewriteVisitor {
    fn visit_path_mut(&mut self, node: &mut syn::Path) {
        self.rewrite_path(node);
        syn::visit_mut::visit_path_mut(self, node);
    }
    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        self.rewrite_use_tree(&mut node.tree, &Vec::new());
        syn::visit_mut::visit_item_use_mut(self, node);
    }
    fn visit_macro_mut(&mut self, node: &mut syn::Macro) {
        self.rewrite_path(&mut node.path);
        syn::visit_mut::visit_macro_mut(self, node);
    }
}


#[derive(Debug, Default, Clone)]
pub struct MoveSet {
    pub entries: HashMap<String, (String, String)>,
}


#[derive(Debug, Default, Clone)]
pub struct MoveSet {
    pub entries: HashMap<String, (String, String)>,
}


#[derive(Debug, Default, Clone)]
pub struct MoveSet {
    pub entries: HashMap<String, (String, String)>,
}


#[derive(Clone)]
pub struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


impl VisitMut for SpanRangeRenamer {
    fn visit_ident_mut(&mut self, ident: &mut syn::Ident) {
        let key = SpanRangeKey::from_span(ident.span());
        if let Some(new_name) = self.map.get(&key) {
            if ident.to_string() != *new_name {
                *ident = syn::Ident::new(new_name, ident.span());
                self.changed = true;
            }
        }
        syn::visit_mut::visit_ident_mut(self, ident);
    }
    fn visit_macro_mut(&mut self, mac: &mut syn::Macro) {
        mac.tokens = rewrite_token_stream(
            mac.tokens.clone(),
            &self.map,
            &mut self.changed,
        );
        syn::visit_mut::visit_macro_mut(self, mac);
    }
}


#[derive(Default)]
pub struct EditSessionTracker {
    files: HashSet<String>,
    pub(crate) doc_files: HashSet<String>,
    pub(crate) attr_files: HashSet<String>,
    pub(crate) use_files: HashSet<String>,
}


#[derive(Debug, Clone)]
pub struct AuxiliaryFile {
    pub path: PathBuf,
    pub kind: AuxiliaryKind,
}


#[derive(Debug, Clone)]
pub struct AuxiliaryFile {
    pub path: PathBuf,
    pub kind: AuxiliaryKind,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}


#[derive(Debug, Clone)]
pub struct DiscoveredFiles {
    pub rust_files: Vec<PathBuf>,
    pub auxiliary_files: Vec<AuxiliaryFile>,
}


#[derive(Debug, Clone)]
pub struct DiscoveredFiles {
    pub rust_files: Vec<PathBuf>,
    pub auxiliary_files: Vec<AuxiliaryFile>,
}


#[derive(Debug, Clone)]
pub struct MacroHandlingReport {
    pub supported_macros: usize,
    pub unsupported_macros: Vec<String>,
    pub extracted_identifiers: usize,
    pub flagged_for_review: Vec<String>,
}


#[derive(Debug, Clone)]
pub struct MacroHandlingReport {
    pub supported_macros: usize,
    pub unsupported_macros: Vec<String>,
    pub extracted_identifiers: usize,
    pub flagged_for_review: Vec<String>,
}


#[derive(Clone, Serialize)]
pub struct FileRename {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) is_directory_move: bool,
    pub(crate) old_module_id: String,
    pub(crate) new_module_id: String,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}


#[derive(Clone, Serialize)]
pub struct SymbolEdit {
    pub(crate) id: String,
    pub(crate) file: String,
    pub(crate) kind: String,
    pub(crate) start: LineColumn,
    pub(crate) end: LineColumn,
    pub(crate) new_name: String,
}


#[derive(Default, Clone)]
pub struct SymbolIndex {
    pub symbols: HashMap<String, SymbolRecord>,
}


#[derive(Default, Clone)]
pub struct SymbolIndex {
    pub symbols: HashMap<String, SymbolRecord>,
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


#[derive(Debug, Clone)]
pub enum LayoutChange {
    /// Convert inline module to file
    InlineToFile,
    /// Convert file module to inline
    FileToInline,
    /// Convert between directory layouts
    DirectoryLayoutChange { from: ModuleLayout, to: ModuleLayout },
}


#[derive(Debug, Clone)]
pub enum LayoutChange {
    /// Convert inline module to file
    InlineToFile,
    /// Convert file module to inline
    FileToInline,
    /// Convert between directory layouts
    DirectoryLayoutChange { from: ModuleLayout, to: ModuleLayout },
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleLayout {
    /// Inline module: `mod foo { ... }`
    Inline,
    /// File module: `foo.rs`
    File(PathBuf),
    /// Directory module with mod.rs: `foo/mod.rs`
    DirectoryModRs(PathBuf),
    /// Directory module with named file: `foo.rs` (where foo/ exists)
    DirectoryNamed(PathBuf),
}


#[derive(Debug, Clone)]
pub struct ModuleMovePlan {
    /// Original module path
    pub from_path: ModulePath,
    /// New module path
    pub to_path: ModulePath,
    /// Original file location
    pub from_file: PathBuf,
    /// New file location
    pub to_file: PathBuf,
    /// Whether this requires creating a new directory
    pub create_directory: bool,
    /// Whether this converts between inline and file module
    pub layout_change: Option<LayoutChange>,
}


#[derive(Debug, Clone)]
pub struct ModuleMovePlan {
    /// Original module path
    pub from_path: ModulePath,
    /// New module path
    pub to_path: ModulePath,
    /// Original file location
    pub from_file: PathBuf,
    /// New file location
    pub to_file: PathBuf,
    /// Whether this requires creating a new directory
    pub create_directory: bool,
    /// Whether this converts between inline and file module
    pub layout_change: Option<LayoutChange>,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModulePath {
    /// Segments of the module path (e.g., ["crate", "foo", "bar"])
    pub segments: Vec<String>,
    /// Whether this is an inline module
    pub is_inline: bool,
}


#[derive(Clone)]
pub struct ImplCtx {
    type_name: String,
}


#[derive(Clone)]
pub struct ResolverContext {
    pub module_path: String,
    pub alias_graph: Arc<AliasGraph>,
    pub symbol_table: Arc<SymbolIndex>,
}


#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}


#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}


#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Default)]
pub struct KernelGraphBuilder {
    materializer: GraphMaterializer,
}


#[derive(Debug, Default)]
pub struct KernelGraphBuilder {
    materializer: GraphMaterializer,
}


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub enum LinuxFact {
    Exists(PathBuf),
    File(PathBuf),
    Dir(PathBuf),
    ProcessRunning(String),
    BinaryInstalled(String),
}


#[derive(Debug, Clone)]
pub enum LinuxFact {
    Exists(PathBuf),
    File(PathBuf),
    Dir(PathBuf),
    ProcessRunning(String),
    BinaryInstalled(String),
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}


#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub enum GraphDelta {
    AddNode(NodeRecord),
    AddEdge(EdgeRecord),
}


#[derive(Debug, Clone)]
pub enum GraphDelta {
    AddNode(NodeRecord),
    AddEdge(EdgeRecord),
}


#[derive(Debug, Clone)]
pub enum GraphDeltaError {
    NodeExists(NodeId),
    EdgeExists(EdgeId),
    NodeMissing(NodeId),
    Persistence(String),
}


impl std::error::Error for GraphDeltaError {}


#[derive(Debug, Clone)]
pub enum GraphDeltaError {
    NodeExists(NodeId),
    EdgeExists(EdgeId),
    NodeMissing(NodeId),
    Persistence(String),
}


#[derive(Debug, Default)]
pub struct GraphMaterializer {
    pub(crate) nodes: HashMap<NodeId, NodeRecord>,
    pub(crate) edges: HashMap<EdgeId, EdgeRecord>,
}


#[derive(Debug, Default)]
pub struct GraphMaterializer {
    pub(crate) nodes: HashMap<NodeId, NodeRecord>,
    pub(crate) edges: HashMap<EdgeId, EdgeRecord>,
}


#[derive(Debug, Default, Clone)]
pub struct GraphSnapshot {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    hash: u64,
}


#[derive(Debug, Default, Clone)]
pub struct GraphSnapshot {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    hash: u64,
}


#[derive(Debug, Default, Clone)]
pub struct GraphSnapshot {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    hash: u64,
}


#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: NodeId,
    pub key: Arc<str>,
    pub label: Arc<str>,
    pub metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: NodeId,
    pub key: Arc<str>,
    pub label: Arc<str>,
    pub metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EdgeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId([u8; 16]);


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}


#[derive(Debug, Clone)]
pub struct GraphWorkspace {
    hash: u64,
}


#[derive(Debug, Clone)]
pub struct GraphWorkspace {
    hash: u64,
}


#[derive(Debug, Clone)]
pub struct WorkspaceBuilder {
    hash: u64,
}


#[derive(Debug, Clone)]
pub struct WorkspaceBuilder {
    hash: u64,
}


#[derive(Clone, Debug)]
pub struct StructuredEditOptions {
    doc_literals: bool,
    attr_literals: bool,
    use_statements: bool,
}


#[derive(Clone, Debug)]
pub struct StructuredEditOptions {
    doc_literals: bool,
    attr_literals: bool,
    use_statements: bool,
}


impl VisitMut for AttributeRewriteVisitor {
    fn visit_attribute_mut(&mut self, attr: &mut syn::Attribute) {
        self.process_attribute(attr);
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}


#[derive(Clone)]
pub enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}


#[derive(Clone)]
pub enum NodeOp {
    ReplaceNode { handle: NodeHandle, new_node: syn::Item },
    InsertBefore { handle: NodeHandle, new_node: syn::Item },
    InsertAfter { handle: NodeHandle, new_node: syn::Item },
    DeleteNode { handle: NodeHandle },
    MutateField { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems { file: PathBuf, new_order: Vec<String> },
    MoveSymbol {
        handle: NodeHandle,
        symbol_id: String,
        new_module_path: String,
        new_crate: Option<String>,
    },
}


impl Default for StructuredPassRunner {
    fn default() -> Self {
        Self::new()
    }
}


impl<'a> VisitMut for UseAstRewriter<'a> {
    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        let mut current_path = Vec::new();
        if node.leading_colon.is_some() {
            current_path.push("crate".to_string());
        }
        rewrite_use_tree_mut(
            &mut node.tree,
            self.updates,
            &mut self.changed,
            &mut current_path,
            self.resolver,
        );
        syn::visit_mut::visit_item_use_mut(self, node);
    }
}


impl AliasGraph {
    pub fn analyze_visibility_leaks(
        &self,
        symbols: &HashMap<String, VisibilityScope>,
    ) -> VisibilityLeakAnalysis {
        let mut analysis = VisibilityLeakAnalysis {
            public_symbols: HashMap::new(),
            restricted_symbols: HashMap::new(),
            leaked_private_symbols: Vec::new(),
        };
        for (symbol_id, visibility) in symbols {
            let module = extract_module_from_path(symbol_id);
            let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
            match visibility {
                VisibilityScope::Public => {
                    let chains = self.find_reexport_chains(symbol_id);
                    let has_chains = !chains.is_empty();
                    for chain in chains {
                        let reexport_modules: Vec<String> = chain
                            .iter()
                            .map(|node| node.module_path.clone())
                            .collect();
                        let exposure = ExposurePath {
                            origin_module: module.clone(),
                            reexport_chain: reexport_modules,
                            visibility: VisibilityScope::Public,
                        };
                        analysis
                            .public_symbols
                            .entry(module.clone())
                            .or_insert_with(Vec::new)
                            .push((name.to_string(), exposure));
                    }
                    if !has_chains {
                        let exposure = ExposurePath {
                            origin_module: module.clone(),
                            reexport_chain: Vec::new(),
                            visibility: VisibilityScope::Public,
                        };
                        analysis
                            .public_symbols
                            .entry(module.clone())
                            .or_insert_with(Vec::new)
                            .push((name.to_string(), exposure));
                    }
                }
                VisibilityScope::Private
                | VisibilityScope::Crate
                | VisibilityScope::Super
                | VisibilityScope::Restricted(_) => {
                    analysis
                        .restricted_symbols
                        .entry(module.clone())
                        .or_insert_with(Vec::new)
                        .push((name.to_string(), visibility.clone()));
                    self.detect_visibility_leak(
                        symbol_id,
                        visibility,
                        &module,
                        &mut analysis.leaked_private_symbols,
                    );
                }
            }
        }
        analysis
    }
    fn detect_visibility_leak(
        &self,
        symbol_id: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        leaked: &mut Vec<LeakedSymbol>,
    ) {
        let importers = self.get_importers(symbol_id);
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(symbol_id.to_string());
        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                if !self
                    .is_visible(
                        origin_module,
                        &importer.module_path,
                        original_visibility,
                    )
                {
                    leaked
                        .push(LeakedSymbol {
                            symbol_id: symbol_id.to_string(),
                            original_visibility: original_visibility.clone(),
                            leaked_to: importer.module_path.clone(),
                            leak_chain: vec![importer.id.clone()],
                        });
                }
                let reexport_path = format!(
                    "{}::{}", importer.module_path, importer.local_name
                );
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    vec![importer.id.clone()],
                    leaked,
                    &mut visited,
                );
            }
        }
    }
    fn detect_visibility_leak_recursive(
        &self,
        current_path: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        chain: Vec<String>,
        leaked: &mut Vec<LeakedSymbol>,
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(current_path.to_string()) {
            return;
        }
        let importers = self.get_importers(current_path);
        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                let mut new_chain = chain.clone();
                new_chain.push(importer.id.clone());
                if !self
                    .is_visible(
                        origin_module,
                        &importer.module_path,
                        original_visibility,
                    )
                {
                    leaked
                        .push(LeakedSymbol {
                            symbol_id: current_path.to_string(),
                            original_visibility: original_visibility.clone(),
                            leaked_to: importer.module_path.clone(),
                            leak_chain: new_chain.clone(),
                        });
                }
                let reexport_path = format!(
                    "{}::{}", importer.module_path, importer.local_name
                );
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    new_chain,
                    leaked,
                    visited,
                );
            }
        }
    }
}


impl PatternBindingCollector {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }
    /// Collect bindings from a pattern
    pub fn collect_from_pattern(pat: &Pat) -> Vec<(String, Option<String>)> {
        let mut collector = Self::new();
        collector.visit_pat(pat);
        collector.bindings
    }
}
