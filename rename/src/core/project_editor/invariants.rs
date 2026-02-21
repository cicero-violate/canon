use database::graph_log::GraphSnapshot;
use std::collections::{HashMap, HashSet};

pub fn assert_unique_node_keys(snapshot: &GraphSnapshot) {
    let mut seen = HashSet::new();
    for node in &snapshot.nodes {
        if !seen.insert(node.key.clone()) {
            panic!("Invariant violation: duplicate node.key detected: {}", node.key);
        }
    }
}

pub fn assert_unique_def_paths(snapshot: &GraphSnapshot) {
    let mut seen = HashSet::new();
    for node in &snapshot.nodes {
        if let Some(def_path) = node.metadata.get("def_path") {
            if !seen.insert(def_path.clone()) {
                panic!("Invariant violation: duplicate def_path detected: {}", def_path);
            }
        }
    }
}

pub fn assert_unique_node_ids(snapshot: &GraphSnapshot) {
    let mut seen = HashSet::new();
    for node in &snapshot.nodes {
        if !seen.insert(node.id.clone()) {
            panic!("Invariant violation: duplicate node.id detected: {:?}", node.id);
        }
    }
}

pub fn assert_edge_endpoints_exist(snapshot: &GraphSnapshot) {
    let node_ids: HashSet<_> = snapshot.nodes.iter().map(|n| n.id.clone()).collect();
    for edge in &snapshot.edges {
        if !node_ids.contains(&edge.from) {
            panic!("Invariant violation: edge.from references missing node id: {:?}", edge.from);
        }
        if !node_ids.contains(&edge.to) {
            panic!("Invariant violation: edge.to references missing node id: {:?}", edge.to);
        }
    }
}

pub fn assert_no_duplicate_edges(snapshot: &GraphSnapshot) {
    let mut seen = HashSet::new();
    for edge in &snapshot.edges {
        let key = (edge.from.clone(), edge.to.clone(), edge.kind.clone());
        if !seen.insert(key.clone()) {
            panic!("Invariant violation: duplicate edge detected: from={:?} to={:?} kind={}", key.0, key.1, key.2);
        }
    }
}

pub fn assert_module_path_consistency(snapshot: &GraphSnapshot) {
    let mut def_path_map: HashMap<String, String> = HashMap::new();
    for node in &snapshot.nodes {
        if let (Some(def_path), Some(module_path)) = (node.metadata.get("def_path"), node.metadata.get("module_path")) {
            if let Some(existing) = def_path_map.get(def_path) {
                if existing != module_path {
                    panic!("Invariant violation: def_path {} appears in multiple modules: {} vs {}", def_path, existing, module_path);
                }
            } else {
                def_path_map.insert(def_path.clone(), module_path.clone());
            }
        }
    }
}
