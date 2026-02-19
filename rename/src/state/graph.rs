use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use super::ids::{EdgeId, NodeId};

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
}

impl std::fmt::Display for GraphDeltaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphDeltaError::NodeExists(id) => write!(f, "node already exists: {id:?}"),
            GraphDeltaError::EdgeExists(id) => write!(f, "edge already exists: {id:?}"),
            GraphDeltaError::NodeMissing(id) => write!(f, "node missing: {id:?}"),
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
