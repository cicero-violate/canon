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
pub mod llm_provider;
pub mod meta;
pub mod observe;
pub mod pipeline;
pub mod refactor;
pub mod reward;
pub mod runner;
pub mod slice;

pub use call::{
    AgentCallError, AgentCallId, AgentCallInput, AgentCallOutput, AgentCallResult,
};
pub use capability::{
    CapabilityEdge, CapabilityGraph, CapabilityKind, CapabilityNode, IrField,
};
pub use dispatcher::{AgentCallDispatcher, DEFAULT_TRUST_THRESHOLD};
pub use meta::{
    GraphMutation, MAX_ENTROPY_DELTA, MIN_NODES, MetaTickError, MetaTickResult,
    UNDERPERFORM_THRESHOLD, run_meta_tick,
};
pub use pipeline::{PipelineError, PipelineResult, RefactorStage, run_pipeline};
pub use pipeline::record_pipeline_outcome;
pub use refactor::{RefactorKind, RefactorProposal, RefactorTarget};
pub use reward::{NodeOutcome, NodeRewardEntry, RewardLedger};
pub use slice::build_ir_slice;
pub use observe::{IrObservation, IrTotals, observe_ir, observation_to_payload};
pub use llm_provider::{LlmProviderError, call_llm};
pub use runner::{RunnerConfig, RunnerError, TickStats, run_agent};
