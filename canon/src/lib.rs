pub mod agent;
pub mod cli;
pub mod commands;
pub mod decision;
pub mod diagnose;
pub mod diff;
pub mod dot_export;
pub mod dot_import;
pub mod evolution;
pub mod gpu;
pub mod ingest;
pub mod io_utils;
pub mod ir;
pub mod layout;
pub mod materialize;
pub mod observe;
pub mod patch_protocol;
pub mod proof;
pub mod runtime;
pub mod schema;
pub mod semantic_builder;
pub mod storage;
pub mod validate;
pub mod version_gate;
pub use cli::Command;
pub use commands::execute_command;
pub use decision::{
    AcceptProposalError, AutoAcceptDslError, ProposalAcceptance, ProposalAcceptanceInput,
    accept_proposal, apply_dsl_proposal,
};
pub use decision::{AutoAcceptDotError, auto_accept_dot_proposal};
pub use decision::{AutoAcceptFnAstError, auto_accept_fn_ast};
pub use dot_export::verify_dot;
pub use evolution::{EvolutionError, apply_deltas};
pub use gpu::{
    codegen::{GpuProgram, flatten_ports, generate_shader},
    dispatch::{GpuExecutor, GpuExecutorError},
};
pub use ir::{CanonicalIr, PipelineStage};
pub use layout::{
    LayoutAssignment, LayoutGraph, LayoutMap, LayoutModule, LayoutNode, LayoutStrategy,
    OriginalLayoutStrategy, PerTypeLayoutStrategy, SemanticGraph,
    SingleFileLayoutStrategy,
};
pub use materialize::{
    FileEntry, FileTree, MaterializeResult, materialize, render_impl_function,
    write_file_tree,
};
pub use observe::execution_events_to_deltas;
pub use patch_protocol::{
    ApprovedPatchRegistry, PatchApplier, PatchDecision, PatchError, PatchGate,
    PatchMetadata, PatchProposal, PatchQueue, VerifiedPatch,
};
pub use proof::smt_bridge::{
    SmtCertificate, SmtError, attach_function_proofs, verify_function_postconditions,
};
pub use semantic_builder::SemanticIrBuilder;
pub use agent::slice_ir_fields;
pub use agent::record_refactor_reward;
pub use agent::{CapabilityNodeDispatcher, DEFAULT_TRUST_THRESHOLD};
pub use agent::{
    AgentCallError, AgentCallId, AgentCallInput, AgentCallOutput, AgentCallResult,
    CapabilityEdge, CapabilityGraph, CapabilityKind, CapabilityNode, IrField,
};
pub use agent::{
    GraphMutation, MAX_ENTROPY_DELTA, MIN_NODES, GraphEvolutionError,
    GraphEvolutionResult, UNDERPERFORM_THRESHOLD, evolve_capability_graph,
};
pub use agent::{PipelineNodeOutcome, NodeRewardEntry, NodeRewardLedger};
pub use agent::{RefactorError, RefactorResult, RefactorStage, run_refactor_pipeline};
pub use agent::{RefactorKind, RefactorProposal, RefactorTarget};
pub use agent::{load_capability_graph, save_capability_graph};
pub use evolution::{
    DEFAULT_TOPOLOGY_THETA, LyapunovError, TopologyFingerprint, enforce_lyapunov_bound,
};
pub use ir::proposal::{DslProposalArtifacts, DslProposalError, create_proposal_from_dsl};
pub use schema::generate_schema;
pub use validate::{CanonRule, ValidationErrors, Violation, validate_ir};
