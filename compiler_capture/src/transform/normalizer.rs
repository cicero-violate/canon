//! Converts captured items into graph deltas.

use std::collections::{BTreeMap, HashMap};

use crate::compiler_capture::CapturedItem;
use crate::compiler_capture::graph::{
    DeltaCollector, EdgeKind, EdgePayload, GraphDelta, NodeId, NodePayload,
};
use database::graph_log::{WireEdge, WireEdgeId, WireNode, WireNodeId};

/// Normalizes captured items into graph deltas.
#[derive(Debug, Default)]
pub struct GraphNormalizer;

impl GraphNormalizer {
    /// Creates a new normalizer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds graph deltas from captured items.
    pub fn normalize(&mut self, items: &[CapturedItem]) -> Vec<GraphDelta> {
        let mut materializer = DeltaCollector::new();
        let mut node_lookup: HashMap<String, NodeId> = HashMap::new();
        let mut pending_edges: Vec<(String, NodeId)> = Vec::new();

        for item in items {
            let deltas = self.emit_deltas(item);
            for delta in deltas {
                if let GraphDelta::AddNode(ref node) = delta {
                    node_lookup.insert(node.key.to_string(), node.id.clone());
                    if let Some(parent) = parent_key(node.key.as_ref()) {
                        pending_edges.push((parent, node.id.clone()));
                    }
                }
                match delta {
                    GraphDelta::AddNode(node) => {
                        let id = node.id.clone();
                        let mut payload = NodePayload::new(node.key.clone(), node.label.clone());
                        for (k, v) in node.metadata.clone() {
                            payload = payload.with_metadata(k, v);
                        }
                        let _ = materializer.add_node(payload);
                        materializer.merge_node_metadata(&id, node.metadata.into_iter());
                    }
                    GraphDelta::AddEdge(edge) => {
                        let mut payload = EdgePayload::new(
                            edge.from.clone(),
                            edge.to.clone(),
                            EdgeKind::from_str(edge.kind.as_str()),
                        );
                        for (k, v) in edge.metadata.clone() {
                            payload = payload.with_metadata(k, v);
                        }
                        let _ = materializer.add_edge(payload);
                    }
                }
            }
        }

        for (parent_key, child_id) in pending_edges {
            if let Some(parent_id) = node_lookup.get(&parent_key) {
                let edge = WireEdge {
                    id: WireEdgeId::from_components(parent_id, &child_id, EdgeKind::Contains.as_str()),
                    from: parent_id.clone(),
                    to: child_id.clone(),
                    kind: EdgeKind::Contains.as_str().to_string(),
                    metadata: BTreeMap::new(),
                };
                let mut payload = EdgePayload::new(
                    edge.from.clone(),
                    edge.to.clone(),
                    EdgeKind::from_str(edge.kind.as_str()),
                );
                for (k, v) in edge.metadata.clone() {
                    payload = payload.with_metadata(k, v);
                }
                let _ = materializer.add_edge(payload);
            }
        }

        materializer.into_deltas()
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
) -> WireNode {
    let mut meta = BTreeMap::new();
    meta.insert("kind".into(), kind.into());
    if let Some(sig) = signature {
        meta.insert("signature".into(), sig);
    }
    for (k, v) in metadata {
        meta.insert(k, v);
    }
    let id = WireNodeId::from_key(path);
    WireNode {
        id,
        key: path.to_string(),
        label: label.to_string(),
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
