//! semantic: domain=tooling
//! Alias and re-export modeling for comprehensive rename tracking

use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
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

/// Graph tracking all use statements and their relationships
#[derive(Debug, Default)]
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

impl AliasGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a use node to the graph
    pub fn add_use_node(&mut self, node: ImportNode) {
        let id = node.id.clone();
        let module_path = node.module_path.clone();
        let local_name = node.local_name.clone();
        let source_path = node.source_path.clone();

        // Track local name resolution
        self.local_names
            .insert((module_path.clone(), local_name), id.clone());

        // Track source imports
        self.source_imports
            .entry(source_path.clone())
            .or_insert_with(Vec::new)
            .push(id.clone());

        // Track glob imports separately
        if node.kind == UseKind::Glob {
            self.glob_imports
                .entry(module_path)
                .or_insert_with(Vec::new)
                .push((source_path, id.clone()));
        }

        self.nodes.insert(id, node);
    }

    /// Resolve a local name in a given module to its source path
    pub fn resolve_local(&self, module_path: &str, local_name: &str) -> Option<&str> {
        self.local_names
            .get(&(module_path.to_string(), local_name.to_string()))
            .and_then(|id| self.nodes.get(id))
            .map(|node| node.source_path.as_str())
    }

    /// Get all use nodes that import from a given source path
    pub fn get_importers(&self, source_path: &str) -> Vec<&ImportNode> {
        self.source_imports
            .get(source_path)
            .map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get all glob imports in a module
    pub fn get_glob_imports(&self, module_path: &str) -> Vec<&ImportNode> {
        self.glob_imports
            .get(module_path)
            .map(|globs| {
                globs
                    .iter()
                    .filter_map(|(_, id)| self.nodes.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a symbol is visible from a given module
    pub fn is_visible(
        &self,
        symbol_module: &str,
        from_module: &str,
        visibility: &VisibilityScope,
    ) -> bool {
        match visibility {
            VisibilityScope::Public => true,
            VisibilityScope::Crate => {
                // Visible within the same crate
                let symbol_crate = symbol_module.split("::").next().unwrap_or("");
                let from_crate = from_module.split("::").next().unwrap_or("");
                symbol_crate == from_crate
            }
            VisibilityScope::Super => {
                // Visible to parent module
                from_module.starts_with(symbol_module)
            }
            VisibilityScope::Private => {
                // Only visible in the same module
                symbol_module == from_module
            }
            VisibilityScope::Restricted(path) => {
                // Visible within the specified path
                from_module.starts_with(path)
            }
        }
    }

    /// Get all use nodes
    pub fn all_nodes(&self) -> Vec<&ImportNode> {
        self.nodes.values().collect()
    }

    /// Get all use nodes that originate from a specific file path
    pub fn nodes_in_file(&self, file_path: &str) -> Vec<&ImportNode> {
        self.nodes
            .values()
            .filter(|node| node.file == file_path)
            .collect()
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: String, to: String, kind: EdgeKind) {
        self.edges.push(AliasEdge { from, to, kind });
    }

    /// Build edges for all use nodes
    pub fn build_edges(&mut self) {
        let nodes: Vec<_> = self.nodes.values().cloned().collect();

        for node in nodes {
            match node.kind {
                UseKind::Simple => {
                    // Direct import edge
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Import);
                }
                UseKind::Aliased => {
                    // Alias edge
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Alias);
                }
                UseKind::ReExport | UseKind::ReExportAliased => {
                    // Re-export edge
                    self.add_edge(
                        node.id.clone(),
                        node.source_path.clone(),
                        EdgeKind::ReExport,
                    );
                }
                UseKind::Glob => {
                    // Glob import - edges will be resolved dynamically
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Import);
                }
            }
        }
    }

    /// Resolve an identifier to its ultimate symbol through the alias chain
    pub fn resolve_chain(&self, module_path: &str, name: &str) -> ResolutionChain {
        let mut chain = ResolutionChain {
            start_name: name.to_string(),
            start_module: module_path.to_string(),
            steps: Vec::new(),
            resolved_symbol: None,
        };

        // Add starting step
        chain.steps.push(ResolutionStep {
            kind: StepKind::Start,
            name: name.to_string(),
            module: module_path.to_string(),
            use_node_id: None,
        });

        // Try to resolve through local use statements
        if let Some(source) = self.resolve_local(module_path, name) {
            let use_node_id = self
                .local_names
                .get(&(module_path.to_string(), name.to_string()))
                .cloned();

            chain.steps.push(ResolutionStep {
                kind: StepKind::LocalUse,
                name: source.to_string(),
                module: extract_module_from_path(source),
                use_node_id,
            });

            // Follow the chain if this is a re-export
            self.follow_reexport_chain(&mut chain, source);

            chain.resolved_symbol = Some(source.to_string());
        } else {
            // Try glob imports
            if let Some(resolved) = self.resolve_through_glob(module_path, name) {
                chain.steps.push(ResolutionStep {
                    kind: StepKind::GlobImport,
                    name: resolved.clone(),
                    module: extract_module_from_path(&resolved),
                    use_node_id: None,
                });

                chain.resolved_symbol = Some(resolved);
            } else {
                // Direct lookup in current module
                let direct_path = format!("{}::{}", module_path, name);
                chain.steps.push(ResolutionStep {
                    kind: StepKind::DirectLookup,
                    name: direct_path.clone(),
                    module: module_path.to_string(),
                    use_node_id: None,
                });

                chain.resolved_symbol = Some(direct_path);
            }
        }

        chain
    }

    /// Follow re-export chain to find ultimate source
    fn follow_reexport_chain(&self, chain: &mut ResolutionChain, current_path: &str) {
        let mut current = current_path.to_string();
        let mut visited = HashSet::new();

        // Prevent infinite loops
        while visited.insert(current.clone()) {
            // Check if current path is re-exported
            let importers = self.get_importers(&current);
            let reexport = importers
                .iter()
                .find(|node| matches!(node.kind, UseKind::ReExport | UseKind::ReExportAliased));

            if let Some(reexport_node) = reexport {
                chain.steps.push(ResolutionStep {
                    kind: StepKind::ReExport,
                    name: reexport_node.source_path.clone(),
                    module: reexport_node.module_path.clone(),
                    use_node_id: Some(reexport_node.id.clone()),
                });

                current = reexport_node.source_path.clone();
            } else {
                break;
            }
        }
    }

    /// Resolve a name through glob imports
    fn resolve_through_glob(&self, module_path: &str, name: &str) -> Option<String> {
        let globs = self.get_glob_imports(module_path);

        for glob_node in globs {
            // Try to resolve name in the glob source module
            let potential_path = format!("{}::{}", glob_node.source_path, name);
            // This is a heuristic - would need symbol table to verify
            return Some(potential_path);
        }

        None
    }

    /// Get all edges
    pub fn all_edges(&self) -> &[AliasEdge] {
        &self.edges
    }

    /// Find all re-export chains for a symbol
    pub fn find_reexport_chains(&self, symbol_id: &str) -> Vec<Vec<ImportNode>> {
        let mut chains = Vec::new();
        let mut current_chains = vec![vec![]];

        self.find_reexport_chains_recursive(symbol_id, &mut current_chains, &mut chains);

        chains
    }

    fn find_reexport_chains_recursive(
        &self,
        symbol_id: &str,
        current_chains: &mut Vec<Vec<ImportNode>>,
        result: &mut Vec<Vec<ImportNode>>,
    ) {
        let importers = self.get_importers(symbol_id);
        let reexports: Vec<ImportNode> = importers
            .into_iter()
            .cloned()
            .filter(|node| matches!(node.kind, UseKind::ReExport | UseKind::ReExportAliased))
            .collect();

        if reexports.is_empty() {
            // End of chain
            for chain in current_chains.iter() {
                if !chain.is_empty() {
                    result.push(chain.clone());
                }
            }
            return;
        }

        for reexport in reexports {
            let mut new_chain = current_chains[0].clone();
            new_chain.push(reexport.clone());

            let reexport_path = format!("{}::{}", reexport.module_path, reexport.local_name);
            let mut new_chains = vec![new_chain];
            self.find_reexport_chains_recursive(&reexport_path, &mut new_chains, result);
        }
    }
}

/// Extract module path from a full symbol path
fn extract_module_from_path(path: &str) -> String {
    if let Some(last_sep) = path.rfind("::") {
        path[..last_sep].to_string()
    } else {
        "crate".to_string()
    }
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

impl AliasGraph {
    /// Analyze visibility leaks in the graph
    pub fn analyze_visibility_leaks(
        &self,
        symbols: &HashMap<String, VisibilityScope>,
    ) -> VisibilityLeakAnalysis {
        let mut analysis = VisibilityLeakAnalysis {
            public_symbols: HashMap::new(),
            restricted_symbols: HashMap::new(),
            leaked_private_symbols: Vec::new(),
        };

        // Analyze each symbol
        for (symbol_id, visibility) in symbols {
            let module = extract_module_from_path(symbol_id);
            let name = symbol_id.rsplit("::").next().unwrap_or(symbol_id);

            match visibility {
                VisibilityScope::Public => {
                    // Find all re-export chains for this symbol
                    let chains = self.find_reexport_chains(symbol_id);
                    let has_chains = !chains.is_empty();

                    for chain in chains {
                        let reexport_modules: Vec<String> =
                            chain.iter().map(|node| node.module_path.clone()).collect();

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

                    // Also add direct exposure if no re-exports
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
                    // Track restricted symbols
                    analysis
                        .restricted_symbols
                        .entry(module.clone())
                        .or_insert_with(Vec::new)
                        .push((name.to_string(), visibility.clone()));

                    // Check if this symbol is leaked through public re-exports
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

    /// Detect if a restricted symbol is leaked through re-exports
    fn detect_visibility_leak(
        &self,
        symbol_id: &str,
        original_visibility: &VisibilityScope,
        origin_module: &str,
        leaked: &mut Vec<LeakedSymbol>,
    ) {
        let importers = self.get_importers(symbol_id);

        for importer in importers {
            // Check if this is a public re-export
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                // This is a leak if the original visibility doesn't allow it
                if !self.is_visible(origin_module, &importer.module_path, original_visibility) {
                    leaked.push(LeakedSymbol {
                        symbol_id: symbol_id.to_string(),
                        original_visibility: original_visibility.clone(),
                        leaked_to: importer.module_path.clone(),
                        leak_chain: vec![importer.id.clone()],
                    });
                }

                // Recursively check if this re-export is further leaked
                let reexport_path = format!("{}::{}", importer.module_path, importer.local_name);
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    vec![importer.id.clone()],
                    leaked,
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
    ) {
        let importers = self.get_importers(current_path);

        for importer in importers {
            if matches!(importer.kind, UseKind::ReExport | UseKind::ReExportAliased)
                && importer.visibility == VisibilityScope::Public
            {
                let mut new_chain = chain.clone();
                new_chain.push(importer.id.clone());

                if !self.is_visible(origin_module, &importer.module_path, original_visibility) {
                    leaked.push(LeakedSymbol {
                        symbol_id: current_path.to_string(),
                        original_visibility: original_visibility.clone(),
                        leaked_to: importer.module_path.clone(),
                        leak_chain: new_chain.clone(),
                    });
                }

                let reexport_path = format!("{}::{}", importer.module_path, importer.local_name);
                self.detect_visibility_leak_recursive(
                    &reexport_path,
                    original_visibility,
                    origin_module,
                    new_chain,
                    leaked,
                );
            }
        }
    }
}
