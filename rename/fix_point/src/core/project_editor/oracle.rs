#[derive(Debug, Clone)]
struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


#[derive(Debug, Clone)]
struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;


/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;


#[derive(Debug, Clone)]
struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


#[derive(Debug, Clone)]
struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;


/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;


impl GraphSnapshotOracle {
    pub(super) fn from_snapshot(snapshot: WireSnapshot) -> Self {
        let mut id_by_key = HashMap::new();
        let mut key_by_index = Vec::new();
        let mut macro_generated = HashSet::new();
        let mut crate_by_key = HashMap::new();
        let mut signature_by_key = HashMap::new();
        for node in snapshot.nodes.iter() {
            let key = node.key.to_string();
            let external_id = node.id.clone();
            id_by_key.insert(key.clone(), external_id.clone());
            key_by_index.push(key.clone());
            if is_macro_generated(&node.metadata) {
                macro_generated.insert(key.clone());
            }
            if let Some(crate_name) = node
                .metadata
                .get("crate")
                .or_else(|| node.metadata.get("crate_name"))
                .or_else(|| node.metadata.get("package"))
            {
                crate_by_key.insert(key.clone(), crate_name.clone());
            }
            if let Some(signature) = node.metadata.get("signature") {
                signature_by_key.insert(key.clone(), signature.clone());
            }
        }
        Self {
            snapshot,
            id_by_key,
            key_by_index,
            macro_generated,
            crate_by_key,
            signature_by_key,
        }
    }
}


fn is_macro_generated(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let value = metadata
        .get("macro_generated")
        .or_else(|| metadata.get("generated_by_macro"))
        .or_else(|| metadata.get("macro"))
        .or_else(|| metadata.get("is_macro"));
    matches!(value.map(| v | v.as_str()), Some("true") | Some("1") | Some("yes"))
}
