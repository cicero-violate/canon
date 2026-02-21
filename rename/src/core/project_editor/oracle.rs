use crate::core::oracle::StructuralEditOracle;
use crate::core::symbol_id::normalize_symbol_id;
use database::graph_log::{GraphSnapshot as WireSnapshot, WireNodeId};
use std::collections::{HashMap, HashSet};
use syn::Signature;

#[derive(Debug, Clone)]
pub(super) struct GraphSnapshotOracle {
    snapshot: WireSnapshot,
    id_by_key: HashMap<String, WireNodeId>,
    key_by_index: Vec<String>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}

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
            if let Some(crate_name) = node.metadata.get("crate").or_else(|| node.metadata.get("crate_name")).or_else(|| node.metadata.get("package")) {
                crate_by_key.insert(key.clone(), crate_name.clone());
            }
            if let Some(signature) = node.metadata.get("signature") {
                signature_by_key.insert(key.clone(), signature.clone());
            }
        }
        Self { snapshot, id_by_key, key_by_index, macro_generated, crate_by_key, signature_by_key }
    }
}

fn is_macro_generated(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let value = metadata.get("macro_generated").or_else(|| metadata.get("generated_by_macro")).or_else(|| metadata.get("macro")).or_else(|| metadata.get("is_macro"));
    matches!(value.map(|v| v.as_str()), Some("true") | Some("1") | Some("yes"))
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
                if key == &symbol_id {
                    None
                } else {
                    Some(key.clone())
                }
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
                if other_crate != symbol_crate {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;

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
