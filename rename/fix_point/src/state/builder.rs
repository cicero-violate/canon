pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


pub struct KernelGraphBuilder {
    materializer: GraphMaterializer,
}


pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}
