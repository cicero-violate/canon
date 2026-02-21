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
