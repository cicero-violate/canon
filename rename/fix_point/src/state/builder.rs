use std::collections::BTreeMap;


use super::graph::{
    EdgeKind, EdgeRecord, GraphDelta, GraphDeltaError, GraphMaterializer, NodeRecord,
};


use super::ids::{EdgeId, NodeId};


use std::sync::Arc;


#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Default)]
pub struct KernelGraphBuilder {
    materializer: GraphMaterializer,
}


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}
