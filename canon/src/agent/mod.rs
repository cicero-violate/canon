//! L3 Capability Graph — stateless LLM call orchestration layer.
//!
//! Each CapabilityNode represents one stateless agent call over a typed
//! slice of CanonicalIr. The CapabilityGraph defines which nodes exist,
//! which fields they may read/write, and how their outputs chain together.
//!
//! State lives exclusively in CanonicalIr and CapabilityGraph edges.
//! Nothing inside a node is stateful.

pub mod call;
pub mod capability;
pub mod dispatcher;
pub mod pipeline;
pub mod refactor;
pub mod slice;

pub use call::{
    AgentCallError, AgentCallId, AgentCallInput, AgentCallOutput, AgentCallResult,
};
pub use capability::{
    CapabilityEdge, CapabilityGraph, CapabilityKind, CapabilityNode, IrField,
};
pub use dispatcher::{AgentCallDispatcher, DEFAULT_TRUST_THRESHOLD};
pub use pipeline::{PipelineError, PipelineResult, RefactorStage, run_pipeline};
pub use refactor::{RefactorKind, RefactorProposal, RefactorTarget};
pub use slice::build_ir_slice;
