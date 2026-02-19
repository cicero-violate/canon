use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::ids::{EdgeId, NodeId};
use graph_gpu::csr::{CsrGraph, EdgeKind as GpuEdgeKind, InputEdge};
use memory_engine::graph_log as wire;

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
pub struct NodeRecord {
    pub id: NodeId,
    pub key: Arc<str>,
    pub label: Arc<str>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub id: EdgeId,
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum GraphDelta {
    AddNode(NodeRecord),
    AddEdge(EdgeRecord),
}

#[derive(Debug, Clone)]
pub enum GraphDeltaError {
    NodeExists(NodeId),
    EdgeExists(EdgeId),
    NodeMissing(NodeId),
    Persistence(String),
}

impl std::fmt::Display for GraphDeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphDeltaError::NodeExists(id) => write!(f, "node already exists: {id:?}"),
            GraphDeltaError::EdgeExists(id) => write!(f, "edge already exists: {id:?}"),
            GraphDeltaError::NodeMissing(id) => write!(f, "node missing: {id:?}"),
            GraphDeltaError::Persistence(msg) => write!(f, "graph persistence failed: {msg}"),
        }
    }
}

impl std::error::Error for GraphDeltaError {}

#[derive(Debug, Default, Clone)]
pub struct GraphSnapshot {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    hash: u64,
}

impl GraphSnapshot {
    pub fn new(nodes: Vec<NodeRecord>, edges: Vec<EdgeRecord>) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for node in &nodes {
            node.id.hash(&mut hasher);
        }
        for edge in &edges {
            edge.id.hash(&mut hasher);
        }
        let hash = hasher.finish();
        Self { nodes, edges, hash }
    }

    pub fn nodes(&self) -> &[NodeRecord] {
        &self.nodes
    }

    pub fn edges(&self) -> &[EdgeRecord] {
        &self.edges
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn from_wire(snapshot: wire::GraphSnapshot) -> Self {
        let mut nodes_by_id: HashMap<NodeId, NodeRecord> = HashMap::new();
        for node in snapshot.nodes {
            let id = NodeId::from_bytes(node.id.0);
            nodes_by_id.insert(id, NodeRecord {
                id,
                key: Arc::<str>::from(node.key),
                label: Arc::<str>::from(node.label),
                metadata: node.metadata,
            });
        }

        let mut edges_by_id: HashMap<EdgeId, EdgeRecord> = HashMap::new();
        for edge in snapshot.edges {
            let id = EdgeId::from_bytes(edge.id.0);
            let from = NodeId::from_bytes(edge.from.0);
            let to = NodeId::from_bytes(edge.to.0);
            edges_by_id.insert(id, EdgeRecord {
                id,
                from,
                to,
                kind: EdgeKind::from_str(edge.kind.as_str()),
                metadata: edge.metadata,
            });
        }

        GraphSnapshot::new(
            nodes_by_id.into_values().collect(),
            edges_by_id.into_values().collect(),
        )
    }

    pub fn diff_nodes(&self, other: &GraphSnapshot) -> Vec<NodeId> {
        let mut changed: BTreeSet<NodeId> = BTreeSet::new();

        let mut left_nodes: HashMap<NodeId, NodeRecord> = HashMap::new();
        for node in &self.nodes {
            left_nodes.insert(node.id, node.clone());
        }
        let mut right_nodes: HashMap<NodeId, NodeRecord> = HashMap::new();
        for node in &other.nodes {
            right_nodes.insert(node.id, node.clone());
        }

        for id in left_nodes.keys().chain(right_nodes.keys()) {
            let left = left_nodes.get(id);
            let right = right_nodes.get(id);
            if left.is_none() || right.is_none() {
                changed.insert(*id);
                continue;
            }
            let left = left.unwrap();
            let right = right.unwrap();
            if left.key != right.key
                || left.label != right.label
                || left.metadata != right.metadata
            {
                changed.insert(*id);
            }
        }

        let mut left_edges: HashMap<EdgeId, EdgeRecord> = HashMap::new();
        for edge in &self.edges {
            left_edges.insert(edge.id, edge.clone());
        }
        let mut right_edges: HashMap<EdgeId, EdgeRecord> = HashMap::new();
        for edge in &other.edges {
            right_edges.insert(edge.id, edge.clone());
        }

        for id in left_edges.keys().chain(right_edges.keys()) {
            let left = left_edges.get(id);
            let right = right_edges.get(id);
            if left.is_none() || right.is_none() {
                if let Some(edge) = left.or(right) {
                    changed.insert(edge.from);
                    changed.insert(edge.to);
                }
                continue;
            }
            let left = left.unwrap();
            let right = right.unwrap();
            if left.from != right.from
                || left.to != right.to
                || left.kind != right.kind
                || left.metadata != right.metadata
            {
                changed.insert(left.from);
                changed.insert(left.to);
            }
        }

        changed.into_iter().collect()
    }

    /// Convert this snapshot into a zero-copy CSR graph for GPU traversal.
    pub fn to_csr(&self) -> CsrGraph {
        let node_ids: Vec<u64> = self.nodes
            .iter()
            .map(|n| n.id.low_u64_le())
            .collect();

        let edges: Vec<InputEdge> = self.edges
            .iter()
            .filter_map(|e| {
                let from = e.from.low_u64_le();
                let to   = e.to.low_u64_le();
                let kind = match e.kind {
                    EdgeKind::Contains    => GpuEdgeKind::Contains,
                    EdgeKind::Call        => GpuEdgeKind::Call,
                    EdgeKind::ControlFlow => GpuEdgeKind::ControlFlow,
                    EdgeKind::Reference   => GpuEdgeKind::Reference,
                };
                Some(InputEdge { from, to, kind })
            })
            .collect();

        CsrGraph::build(&node_ids, &edges)
    }
}

impl GraphDelta {
    pub fn to_wire(&self) -> wire::GraphDelta {
        match self {
            GraphDelta::AddNode(node) => wire::GraphDelta::AddNode(wire::WireNode {
                id: wire::WireNodeId(node.id.as_bytes()),
                key: node.key.to_string(),
                label: node.label.to_string(),
                metadata: node.metadata.clone(),
            }),
            GraphDelta::AddEdge(edge) => wire::GraphDelta::AddEdge(wire::WireEdge {
                id: wire::WireEdgeId(edge.id.as_bytes()),
                from: wire::WireNodeId(edge.from.as_bytes()),
                to: wire::WireNodeId(edge.to.as_bytes()),
                kind: edge.kind.as_str().to_string(),
                metadata: edge.metadata.clone(),
            }),
        }
    }
}

#[derive(Debug, Default)]
pub struct GraphMaterializer {
    pub(crate) nodes: HashMap<NodeId, NodeRecord>,
    pub(crate) edges: HashMap<EdgeId, EdgeRecord>,
}

impl GraphMaterializer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, delta: GraphDelta) -> Result<(), GraphDeltaError> {
        match delta {
            GraphDelta::AddNode(node) => {
                if self.nodes.contains_key(&node.id) {
                    return Err(GraphDeltaError::NodeExists(node.id));
                }
                self.nodes.insert(node.id, node);
            }
            GraphDelta::AddEdge(edge) => {
                if self.edges.contains_key(&edge.id) {
                    return Err(GraphDeltaError::EdgeExists(edge.id));
                }
                if !self.nodes.contains_key(&edge.from) {
                    return Err(GraphDeltaError::NodeMissing(edge.from));
                }
                if !self.nodes.contains_key(&edge.to) {
                    return Err(GraphDeltaError::NodeMissing(edge.to));
                }
                self.edges.insert(edge.id, edge);
            }
        }
        Ok(())
    }

    pub fn snapshot(&self) -> GraphSnapshot {
        let nodes = self.nodes.values().cloned().collect::<Vec<_>>();
        let edges = self.edges.values().cloned().collect::<Vec<_>>();
        GraphSnapshot::new(nodes, edges)
    }
}
