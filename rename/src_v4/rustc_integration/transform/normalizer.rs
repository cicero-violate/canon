//! Converts captured items into [`GraphSnapshot`] instances.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::rustc_integration::CapturedItem;
use crate::state::graph::{
    EdgeKind, EdgeRecord, GraphDelta, GraphDeltaError, GraphMaterializer, GraphSnapshot, NodeRecord,
};
use crate::state::ids::{EdgeId, NodeId};

/// Normalizes captured items into the kernel's [`GraphSnapshot`].
#[derive(Debug, Default)]
pub struct GraphNormalizer;

impl GraphNormalizer {
    /// Creates a new normalizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a graph snapshot from captured items.
    pub fn normalize(&mut self, items: &[CapturedItem]) -> GraphSnapshot {
        let mut materializer = GraphMaterializer::new();
        let mut node_lookup: HashMap<String, NodeId> = HashMap::new();
        let mut pending_edges: Vec<(String, NodeId)> = Vec::new();

        for item in items {
            let deltas = self.emit_deltas(item);
            for delta in deltas {
                if let GraphDelta::AddNode(ref node) = delta {
                    node_lookup.insert(node.key.to_string(), node.id);
                    if let Some(parent) = parent_key(node.key.as_ref()) {
                        pending_edges.push((parent, node.id));
                    }
                }
                if let Err(err) = materializer.apply(delta) {
                    if matches!(err, GraphDeltaError::NodeExists(_)) {
                        // continue
                    } else {
                        eprintln!("graph delta error: {err:?}");
                    }
                }
            }
        }

        for (parent_key, child_id) in pending_edges {
            if let Some(parent_id) = node_lookup.get(&parent_key) {
                let edge = EdgeRecord {
                    id: EdgeId::from_components(parent_id, &child_id, EdgeKind::Contains.as_str()),
                    from: *parent_id,
                    to: child_id,
                    kind: EdgeKind::Contains,
                    metadata: BTreeMap::new(),
                };
                let _ = materializer.apply(GraphDelta::AddEdge(edge));
            }
        }

        materializer.snapshot()
    }

    fn emit_deltas(&self, item: &CapturedItem) -> Vec<GraphDelta> {
        match item {
            CapturedItem::Function(func) => {
                vec![GraphDelta::AddNode(make_node_record(
                    &func.path,
                    &func.name,
                    "function",
                    func.metadata.clone(),
                    func.signature.clone(),
                ))]
            }
            CapturedItem::Struct(strct) => vec![GraphDelta::AddNode(make_node_record(
                &strct.path,
                &strct.name,
                "struct",
                strct.metadata.clone(),
                None,
            ))],
            CapturedItem::Enum(enm) => vec![GraphDelta::AddNode(make_node_record(
                &enm.path,
                &enm.name,
                "enum",
                enm.metadata.clone(),
                None,
            ))],
            CapturedItem::Trait(trt) => vec![GraphDelta::AddNode(make_node_record(
                &trt.path,
                &trt.name,
                "trait",
                trt.metadata.clone(),
                None,
            ))],
            CapturedItem::Impl(imp) => vec![GraphDelta::AddNode(make_node_record(
                &imp.path,
                &imp.name,
                "impl",
                imp.metadata.clone(),
                None,
            ))],
            CapturedItem::Module(module) => vec![GraphDelta::AddNode(make_node_record(
                &module.path,
                &module.name,
                "module",
                module.metadata.clone(),
                None,
            ))],
            CapturedItem::TypeAlias(alias) => vec![GraphDelta::AddNode(make_node_record(
                &alias.path,
                &alias.name,
                "type_alias",
                alias.metadata.clone(),
                None,
            ))],
            CapturedItem::Const(konst) => vec![GraphDelta::AddNode(make_node_record(
                &konst.path,
                &konst.name,
                "const",
                konst.metadata.clone(),
                None,
            ))],
            CapturedItem::Static(stat) => vec![GraphDelta::AddNode(make_node_record(
                &stat.path,
                &stat.name,
                "static",
                stat.metadata.clone(),
                None,
            ))],
        }
    }
}

fn make_node_record(
    path: &str,
    label: &str,
    kind: &str,
    metadata: std::collections::HashMap<String, String>,
    signature: Option<String>,
) -> NodeRecord {
    let mut meta = BTreeMap::new();
    meta.insert("kind".into(), kind.into());
    if let Some(sig) = signature {
        meta.insert("signature".into(), sig);
    }
    for (k, v) in metadata {
        meta.insert(k, v);
    }
    let id = NodeId::from_key(path);
    NodeRecord {
        id,
        key: Arc::<str>::from(path.to_string()),
        label: Arc::<str>::from(label.to_string()),
        metadata: meta,
    }
}

fn parent_key(path: &str) -> Option<String> {
    let mut segments: Vec<&str> = path.split("::").collect();
    if segments.len() <= 1 {
        return None;
    }
    segments.pop();
    Some(segments.join("::"))
}
