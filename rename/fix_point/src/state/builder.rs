impl NodePayload {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl NodePayload {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl EdgePayload {
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind) -> Self {
        Self {
            from,
            to,
            kind,
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl EdgePayload {
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind) -> Self {
        Self {
            from,
            to,
            kind,
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl KernelGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_node(&mut self, payload: NodePayload) -> Result<NodeId, GraphDeltaError> {
        let node_id = NodeId::from_key(&payload.key);
        let node = NodeRecord {
            id: node_id,
            key: Arc::<str>::from(payload.key),
            label: Arc::<str>::from(payload.label),
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddNode(node))?;
        Ok(node_id)
    }
    pub fn add_edge(&mut self, payload: EdgePayload) -> Result<EdgeId, GraphDeltaError> {
        let edge_id = EdgeId::from_components(
            &payload.from,
            &payload.to,
            payload.kind.as_str(),
        );
        let edge = EdgeRecord {
            id: edge_id,
            from: payload.from,
            to: payload.to,
            kind: payload.kind,
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddEdge(edge))?;
        Ok(edge_id)
    }
    pub fn merge_node_metadata<I>(
        &mut self,
        node_id: NodeId,
        updates: I,
    ) -> Result<(), GraphDeltaError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let Some(node) = self.materializer.nodes.get_mut(&node_id) else {
            return Err(GraphDeltaError::NodeMissing(node_id));
        };
        for (k, v) in updates {
            node.metadata.insert(k, v);
        }
        Ok(())
    }
    pub fn snapshot(&self) -> super::graph::GraphSnapshot {
        self.materializer.snapshot()
    }
    pub fn finalize(&self) -> super::graph::GraphSnapshot {
        self.snapshot()
    }
}


impl KernelGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_node(&mut self, payload: NodePayload) -> Result<NodeId, GraphDeltaError> {
        let node_id = NodeId::from_key(&payload.key);
        let node = NodeRecord {
            id: node_id,
            key: Arc::<str>::from(payload.key),
            label: Arc::<str>::from(payload.label),
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddNode(node))?;
        Ok(node_id)
    }
    pub fn add_edge(&mut self, payload: EdgePayload) -> Result<EdgeId, GraphDeltaError> {
        let edge_id = EdgeId::from_components(
            &payload.from,
            &payload.to,
            payload.kind.as_str(),
        );
        let edge = EdgeRecord {
            id: edge_id,
            from: payload.from,
            to: payload.to,
            kind: payload.kind,
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddEdge(edge))?;
        Ok(edge_id)
    }
    pub fn merge_node_metadata<I>(
        &mut self,
        node_id: NodeId,
        updates: I,
    ) -> Result<(), GraphDeltaError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let Some(node) = self.materializer.nodes.get_mut(&node_id) else {
            return Err(GraphDeltaError::NodeMissing(node_id));
        };
        for (k, v) in updates {
            node.metadata.insert(k, v);
        }
        Ok(())
    }
    pub fn snapshot(&self) -> super::graph::GraphSnapshot {
        self.materializer.snapshot()
    }
    pub fn finalize(&self) -> super::graph::GraphSnapshot {
        self.snapshot()
    }
}


impl NodePayload {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl NodePayload {
    pub fn new(key: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl EdgePayload {
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind) -> Self {
        Self {
            from,
            to,
            kind,
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl EdgePayload {
    pub fn new(from: NodeId, to: NodeId, kind: EdgeKind) -> Self {
        Self {
            from,
            to,
            kind,
            metadata: BTreeMap::new(),
        }
    }
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}


impl KernelGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_node(&mut self, payload: NodePayload) -> Result<NodeId, GraphDeltaError> {
        let node_id = NodeId::from_key(&payload.key);
        let node = NodeRecord {
            id: node_id,
            key: Arc::<str>::from(payload.key),
            label: Arc::<str>::from(payload.label),
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddNode(node))?;
        Ok(node_id)
    }
    pub fn add_edge(&mut self, payload: EdgePayload) -> Result<EdgeId, GraphDeltaError> {
        let edge_id = EdgeId::from_components(
            &payload.from,
            &payload.to,
            payload.kind.as_str(),
        );
        let edge = EdgeRecord {
            id: edge_id,
            from: payload.from,
            to: payload.to,
            kind: payload.kind,
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddEdge(edge))?;
        Ok(edge_id)
    }
    pub fn merge_node_metadata<I>(
        &mut self,
        node_id: NodeId,
        updates: I,
    ) -> Result<(), GraphDeltaError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let Some(node) = self.materializer.nodes.get_mut(&node_id) else {
            return Err(GraphDeltaError::NodeMissing(node_id));
        };
        for (k, v) in updates {
            node.metadata.insert(k, v);
        }
        Ok(())
    }
    pub fn snapshot(&self) -> super::graph::GraphSnapshot {
        self.materializer.snapshot()
    }
    pub fn finalize(&self) -> super::graph::GraphSnapshot {
        self.snapshot()
    }
}


impl KernelGraphBuilder {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_node(&mut self, payload: NodePayload) -> Result<NodeId, GraphDeltaError> {
        let node_id = NodeId::from_key(&payload.key);
        let node = NodeRecord {
            id: node_id,
            key: Arc::<str>::from(payload.key),
            label: Arc::<str>::from(payload.label),
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddNode(node))?;
        Ok(node_id)
    }
    pub fn add_edge(&mut self, payload: EdgePayload) -> Result<EdgeId, GraphDeltaError> {
        let edge_id = EdgeId::from_components(
            &payload.from,
            &payload.to,
            payload.kind.as_str(),
        );
        let edge = EdgeRecord {
            id: edge_id,
            from: payload.from,
            to: payload.to,
            kind: payload.kind,
            metadata: payload.metadata,
        };
        self.materializer.apply(GraphDelta::AddEdge(edge))?;
        Ok(edge_id)
    }
    pub fn merge_node_metadata<I>(
        &mut self,
        node_id: NodeId,
        updates: I,
    ) -> Result<(), GraphDeltaError>
    where
        I: IntoIterator<Item = (String, String)>,
    {
        let Some(node) = self.materializer.nodes.get_mut(&node_id) else {
            return Err(GraphDeltaError::NodeMissing(node_id));
        };
        for (k, v) in updates {
            node.metadata.insert(k, v);
        }
        Ok(())
    }
    pub fn snapshot(&self) -> super::graph::GraphSnapshot {
        self.materializer.snapshot()
    }
    pub fn finalize(&self) -> super::graph::GraphSnapshot {
        self.snapshot()
    }
}


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


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


#[derive(Debug, Clone)]
pub struct NodePayload {
    key: String,
    label: String,
    metadata: BTreeMap<String, String>,
}


#[derive(Debug, Clone)]
pub struct EdgePayload {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
    metadata: BTreeMap<String, String>,
}


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


#[derive(Debug, Default)]
pub struct KernelGraphBuilder {
    materializer: GraphMaterializer,
}
