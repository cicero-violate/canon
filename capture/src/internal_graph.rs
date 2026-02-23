use std::collections::{BTreeMap, HashMap};

pub use database::graph_log::{
    GraphDelta, NodeId, WireEdge, WireEdgeId, WireNode, WireNodeId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeKind::Contains => "contains",
            EdgeKind::Call => "call",
            EdgeKind::ControlFlow => "control_flow",
            EdgeKind::Reference => "reference",
        }
    }

    pub fn from_str(value: &str) -> Self {
        match value {
            "call" => EdgeKind::Call,
            "control_flow" => EdgeKind::ControlFlow,
            "reference" => EdgeKind::Reference,
            _ => EdgeKind::Contains,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}

impl NodePayload {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self { key: key.into(), label: label.into(), metadata: BTreeMap::new() }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: WireNodeId,
    to: WireNodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}

impl EdgePayload {
    pub fn new(from: WireNodeId, to: WireNodeId, kind: EdgeKind) -> Self {
        Self { from, to, kind, metadata: BTreeMap::new() }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Default)]
pub struct DeltaCollector {
    nodes: HashMap<WireNodeId, WireNode>,
    edges: HashMap<WireEdgeId, WireEdge>,
}

impl DeltaCollector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, payload: NodePayload) -> WireNodeId {
        let node_id = WireNodeId::from_key(&payload.key);
        let node = WireNode { id: node_id.clone(), key: payload.key, label: payload.label, metadata: payload.metadata };
        self.nodes.entry(node_id.clone()).or_insert(node);
        node_id
    }

    pub fn add_edge(&mut self, payload: EdgePayload) -> WireEdgeId {
        let edge_id = WireEdgeId::from_components(&payload.from, &payload.to, payload.kind.as_str());
        let edge = WireEdge { id: edge_id.clone(), from: payload.from, to: payload.to, kind: payload.kind.as_str().to_string(), metadata: payload.metadata };
        self.edges.entry(edge_id.clone()).or_insert(edge);
        edge_id
    }

    pub fn merge_node_metadata<I>(&mut self, node_id: &WireNodeId, updates: I)
    where I: IntoIterator<Item = (String, String)> {
        if let Some(node) = self.nodes.get_mut(node_id) {
            for (k, v) in updates {
                node.metadata.insert(k, v);
            }
        }
    }

    pub fn into_deltas(self) -> Vec<GraphDelta> {
        let mut out = Vec::with_capacity(self.nodes.len() + self.edges.len());
        for node in self.nodes.into_values() {
            out.push(GraphDelta::AddNode(node));
        }
        for edge in self.edges.into_values() {
            out.push(GraphDelta::AddEdge(edge));
        }
        out
    }
}
