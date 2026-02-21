use std::collections::{BTreeMap, BTreeSet, HashMap};


use std::hash::{Hash, Hasher};


use std::sync::Arc;


use super::ids::{EdgeId, NodeId};


use database::graph_log as wire;


#[cfg(feature = "cuda")]
use algorithms::graph::{csr::Csr, gpu::bfs_gpu};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeKind {
    Contains,
    Call,
    ControlFlow,
    Reference,
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


#[derive(Debug, Default)]
pub struct GraphMaterializer {
    pub(crate) nodes: HashMap<NodeId, NodeRecord>,
    pub(crate) edges: HashMap<EdgeId, EdgeRecord>,
}


#[derive(Debug, Default, Clone)]
pub struct GraphSnapshot {
    nodes: Vec<NodeRecord>,
    edges: Vec<EdgeRecord>,
    hash: u64,
}


#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: NodeId,
    pub key: Arc<str>,
    pub label: Arc<str>,
    pub metadata: BTreeMap<String, String>,
}
