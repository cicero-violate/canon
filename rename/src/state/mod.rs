//! State layer for structural editing and graph capture.

pub mod builder;
pub mod graph;
pub mod ids;
pub mod node;
pub mod workspace;

pub use builder::{EdgePayload, KernelGraphBuilder, NodePayload};
pub use graph::{
    EdgeKind, EdgeRecord, GraphDelta, GraphDeltaError, GraphMaterializer, GraphSnapshot, NodeRecord,
};
pub use ids::{EdgeId, NodeId};
pub use node::{NodeHandle, NodeKind, NodeRegistry};
pub use workspace::{GraphWorkspace, WorkspaceBuilder};
pub mod capability;
