use super::helpers::extract_module_from_path;
use super::types::{AliasEdge, EdgeKind, ImportNode, ResolutionChain, ResolutionStep, StepKind, UseKind, VisibilityScope};
use std::collections::{HashMap, HashSet};

#[cfg(feature = "cuda")]
use algorithms::graph::{csr::Csr, gpu::bfs_gpu};
/// Graph tracking all use statements and their relationships
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
        self.local_names.insert((module_path.clone(), local_name), id.clone());
        self.source_imports.entry(source_path.clone()).or_insert_with(Vec::new).push(id.clone());
        if node.kind == UseKind::Glob {
            self.glob_imports.entry(module_path).or_insert_with(Vec::new).push((source_path, id.clone()));
        }
        self.nodes.insert(id, node);
    }
    /// Resolve a local name in a given module to its source path
    pub fn resolve_local(&self, module_path: &str, local_name: &str) -> Option<&str> {
        self.local_names.get(&(module_path.to_string(), local_name.to_string())).and_then(|id| self.nodes.get(id)).map(|node| node.source_path.as_str())
    }
    /// Get all use nodes that import from a given source path
    pub fn get_importers(&self, source_path: &str) -> Vec<&ImportNode> {
        self.source_imports.get(source_path).map(|ids| ids.iter().filter_map(|id| self.nodes.get(id)).collect()).unwrap_or_default()
    }
    /// Get all glob imports in a module
    pub fn get_glob_imports(&self, module_path: &str) -> Vec<&ImportNode> {
        self.glob_imports.get(module_path).map(|globs| globs.iter().filter_map(|(_, id)| self.nodes.get(id)).collect()).unwrap_or_default()
    }
    /// Check if a symbol is visible from a given module
    pub fn is_visible(&self, symbol_module: &str, from_module: &str, visibility: &VisibilityScope) -> bool {
        match visibility {
            VisibilityScope::Public => true,
            VisibilityScope::Crate => {
                let symbol_crate = symbol_module.split("::").next().unwrap_or("");
                let from_crate = from_module.split("::").next().unwrap_or("");
                symbol_crate == from_crate
            }
            VisibilityScope::Super => from_module.starts_with(symbol_module),
            VisibilityScope::Private => symbol_module == from_module,
            VisibilityScope::Restricted(path) => from_module.starts_with(path),
        }
    }
    /// Get all use nodes
    pub fn all_nodes(&self) -> Vec<&ImportNode> {
        self.nodes.values().collect()
    }
    /// Get all use nodes that originate from a specific file path
    pub fn nodes_in_file(&self, file_path: &str) -> Vec<&ImportNode> {
        self.nodes.values().filter(|node| node.file == file_path).collect()
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
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Import);
                }
                UseKind::Aliased => {
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Alias);
                }
                UseKind::ReExport | UseKind::ReExportAliased => {
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::ReExport);
                }
                UseKind::Glob => {
                    self.add_edge(node.id.clone(), node.source_path.clone(), EdgeKind::Import);
                }
            }
        }
    }
    /// Resolve an identifier to its ultimate symbol through the alias chain
    pub fn resolve_alias_chain(&self, module_path: &str, name: &str) -> ResolutionChain {
        let mut chain = ResolutionChain { start_name: name.to_string(), start_module: module_path.to_string(), steps: Vec::new(), resolved_symbol: None };
        chain.steps.push(ResolutionStep { kind: StepKind::Start, name: name.to_string(), module: module_path.to_string(), use_node_id: None });
        let mut visited = HashSet::new();
        if let Some(source) = self.resolve_local(module_path, name) {
            let use_node_id = self.local_names.get(&(module_path.to_string(), name.to_string())).cloned();
            chain.steps.push(ResolutionStep { kind: StepKind::LocalUse, name: source.to_string(), module: extract_module_from_path(source), use_node_id });
            self.follow_reexport_chain_safe(&mut chain, source, &mut visited);
            chain.resolved_symbol = Some(source.to_string());
        } else {
            if let Some(resolved) = self.resolve_through_glob(module_path, name) {
                chain.steps.push(ResolutionStep { kind: StepKind::GlobImport, name: resolved.clone(), module: extract_module_from_path(&resolved), use_node_id: None });
                chain.resolved_symbol = Some(resolved);
            } else {
                let direct_path = format!("{}::{}", module_path, name);
                chain.steps.push(ResolutionStep { kind: StepKind::DirectLookup, name: direct_path.clone(), module: module_path.to_string(), use_node_id: None });
                chain.resolved_symbol = Some(direct_path);
            }
        }
        chain
    }
    /// Follow re-export chain to find ultimate source
    fn follow_reexport_chain_safe(
        &self,
        chain: &mut ResolutionChain,
        current_path: &str,
        visited: &mut HashSet<String>,
    ) {
        let mut current = current_path.to_string();
        while visited.insert(current.clone()) {
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
            let potential_path = format!("{}::{}", glob_node.source_path, name);
            return Some(potential_path);
        }
        None
    }
    /// Get all edges
    pub fn all_edges(&self) -> &[AliasEdge] {
        &self.edges
    }

    /// Return all node IDs reachable from `start_id` via directed alias edges, using GPU BFS.
    /// Assigns a dense integer index to each node, builds a Csr, runs bfs_gpu,
    /// then maps level >= 0 entries back to node IDs.
    #[cfg(feature = "cuda")]
    pub fn reachable_nodes_from(&self, start_id: &str) -> Vec<String> {
        // Build dense index: node_id -> usize
        let mut id_to_idx: HashMap<&str, usize> = HashMap::new();
        let mut idx_to_id: Vec<&str> = Vec::new();
        for id in self.nodes.keys() {
            id_to_idx.insert(id.as_str(), idx_to_id.len());
            idx_to_id.push(id.as_str());
        }
        let v = idx_to_id.len();
        if v == 0 {
            return vec![];
        }
        let Some(&start_idx) = id_to_idx.get(start_id) else {
            return vec![];
        };
        // Build adjacency list from alias edges (from -> to by index)
        let mut adj: Vec<Vec<usize>> = vec![vec![]; v];
        for edge in &self.edges {
            if let (Some(&fi), Some(&ti)) = (id_to_idx.get(edge.from.as_str()), id_to_idx.get(edge.to.as_str())) {
                adj[fi].push(ti);
            }
        }
        let csr = Csr::from_adj(&adj);
        let levels = bfs_gpu(&csr, start_idx);
        levels.into_iter().enumerate().filter(|(_, lvl)| *lvl >= 0).map(|(i, _)| idx_to_id[i].to_string()).collect()
    }
    /// Find all re-export chains for a symbol
    pub fn find_reexport_chains(&self, symbol_id: &str) -> Vec<Vec<ImportNode>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();
        self.find_reexport_chains_recursive(
            symbol_id,
            &mut vec![vec![]],
            &mut chains,
            &mut visited,
        );
        chains
    }
    fn find_reexport_chains_recursive(
        &self,
        symbol_id: &str,
        current_chains: &mut Vec<Vec<ImportNode>>,
        result: &mut Vec<Vec<ImportNode>>,
        visited: &mut HashSet<String>,
    ) {
        if !visited.insert(symbol_id.to_string()) {
            return;
        }

        let importers = self.get_importers(symbol_id);
        let reexports: Vec<ImportNode> = importers
            .into_iter()
            .cloned()
            .filter(|node| matches!(node.kind, UseKind::ReExport | UseKind::ReExportAliased))
            .collect();
        if reexports.is_empty() {
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
            self.find_reexport_chains_recursive(
                &reexport_path,
                &mut new_chains,
                result,
                visited,
            );
        }
    }
}
