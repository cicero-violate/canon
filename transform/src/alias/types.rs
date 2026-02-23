use serde::Serialize;
use std::collections::HashMap;
use syn::Visibility;

/// Represents a use statement with full context
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
                    VisibilityScope::Restricted(restricted.path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::"))
                }
            }
            Visibility::Inherited => VisibilityScope::Private,
        }
    }
}

/// Represents an edge in the alias resolution graph
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

/// Represents a complete resolution chain for an identifier
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

/// Analysis of which symbols are publicly exposed (leak analysis)
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

/// Describes how a symbol is exposed
#[derive(Debug, Clone, Serialize)]
pub struct ExposurePath {
    /// The symbol's original module
    pub origin_module: String,

    /// Modules that re-export this symbol
    pub reexport_chain: Vec<String>,

    /// Final visibility level
    pub visibility: VisibilityScope,
}

/// A symbol that is leaked beyond its intended visibility
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
