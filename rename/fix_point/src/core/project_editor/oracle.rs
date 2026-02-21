struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}


pub struct NullOracle;


fn is_macro_generated(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let value = metadata
        .get("macro_generated")
        .or_else(|| metadata.get("generated_by_macro"))
        .or_else(|| metadata.get("macro"))
        .or_else(|| metadata.get("is_macro"));
    matches!(value.map(| v | v.as_str()), Some("true") | Some("1") | Some("yes"))
}
