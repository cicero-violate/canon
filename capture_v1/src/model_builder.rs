use crate::shared::internal_graph::GraphDelta;
use anyhow::Result;
use model::model_ir::Model;

/// Canonical Model₀ builder.
/// Consumes GraphDelta stream emitted by rustc frontend
/// and reconstructs canonical Model IR.
pub struct ModelBuilder;

impl ModelBuilder {
    pub fn build(deltas: Vec<GraphDelta>) -> Result<Model> {
        let mut model = Model::default();

        for delta in deltas {
            match delta {
                GraphDelta::AddNode(node) => {
                    model.insert_node(node.id.clone(), node.metadata.clone());
                }
                GraphDelta::AddEdge(edge) => {
                    model.insert_edge(
                        edge.from.clone(),
                        edge.to.clone(),
                        edge.kind.clone(),
                        edge.metadata.clone(),
                    );
                }
                GraphDelta::MergeNodeMetadata { id, metadata } => {
                    model.merge_node_metadata(id, metadata);
                }
                GraphDelta::RemoveNode { id } => {
                    model.remove_node(&id);
                }
                GraphDelta::RemoveEdge { from, to, kind } => {
                    model.remove_edge(&from, &to, &kind);
                }
            }
        }

        Ok(model)
    }
}

