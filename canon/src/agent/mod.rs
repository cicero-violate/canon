//! L3 Capability Graph — stateless LLM call orchestration layer.
//!
//! Each CapabilityNode represents one stateless agent call over a typed
//! slice of CanonicalIr. The CapabilityGraph defines which nodes exist,
//! which fields they may read/write, and how their outputs chain together.
//!
//! State lives exclusively in CanonicalIr and CapabilityGraph edges.
//! Nothing inside a node is stateful.

pub mod bootstrap;
pub mod call;
pub mod capability;
pub mod dispatcher;
pub mod io;
pub mod llm_provider;
pub mod meta;
pub mod observe;
pub mod pipeline;
pub mod refactor;
pub mod reward;
pub mod runner;
pub mod slice;
pub mod sse;
pub mod ws_server;

pub use bootstrap::{bootstrap_graph, bootstrap_proposal};
pub use call::{AgentCallError, AgentCallId, AgentCallInput, AgentCallOutput, AgentCallResult};
pub use capability::{CapabilityEdge, CapabilityGraph, CapabilityKind, CapabilityNode, IrField};
pub use dispatcher::{AgentCallDispatcher, DEFAULT_TRUST_THRESHOLD};
pub use io::{load_capability_graph, save_capability_graph};
pub use llm_provider::{LlmProviderError, call_llm};
pub use meta::{
    GraphMutation, MAX_ENTROPY_DELTA, MIN_NODES, MetaTickError, MetaTickResult,
    UNDERPERFORM_THRESHOLD, run_meta_tick,
};
pub use observe::{IrObservation, IrTotals, observation_to_payload, observe_ir};
pub use pipeline::record_pipeline_outcome;
pub use pipeline::{PipelineError, PipelineResult, RefactorStage, run_pipeline};
pub use refactor::{RefactorKind, RefactorProposal, RefactorTarget};
pub use reward::{NodeOutcome, NodeRewardEntry, RewardLedger};
pub use runner::{RunnerConfig, RunnerError, TickStats, run_agent};
pub use slice::build_ir_slice;
pub use ws_server::WsBridge;
