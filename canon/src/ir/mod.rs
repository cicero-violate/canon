mod admission;
mod artifacts;
mod core;
mod delta;
mod errors;
mod functions;
pub mod goals;
mod gpu;
mod graphs;
mod ids;
mod judgment;
mod learning;
mod policy;
mod project;
mod proofs;
mod proposal;
mod reward;
mod timeline;
mod types;
mod word;
pub mod world_model;

pub use admission::{AppliedDeltaRecord, DeltaAdmission};
pub use artifacts::{
    AssociatedConst, AssociatedType, ConstItem, EnumNode, EnumVariant, EnumVariantFields, Field,
    ImplBlock, ImplFunctionBinding, Module, ModuleEdge, PubUseItem, StaticItem, Struct, Trait,
    TraitFunction, TypeAlias,
};
pub use core::{CanonicalIr, CanonicalMeta, Language, PipelineStage, VersionContract};
pub use delta::{Delta, DeltaKind, DeltaPayload};
pub use errors::ErrorArtifact;
pub use functions::{
    DeltaRef, Function, FunctionContract, FunctionMetadata, FunctionSignature, GenericParam,
    Postcondition, WhereClause,
};
pub use goals::{GoalDriftMetric, GoalMutation, GoalMutationStatus};
pub use gpu::{GpuFunction, GpuProperties, VectorPort};
pub use graphs::{
    CallEdge, SystemEdge, SystemEdgeKind, SystemGraph, SystemNode, SystemNodeKind, TickEdge,
    TickGraph,
};
pub use ids::*; // EnumId now included
pub use judgment::{Judgment, JudgmentDecision, JudgmentPredicate};
pub use learning::Learning;
pub use policy::PolicyParameters;
pub use project::{ExternalDependency, Project};
pub use proofs::{Proof, ProofArtifact, ProofScope};
pub use proposal::{
    Proposal, ProposalGoal, ProposalKind, ProposalStatus, ProposedApi, ProposedEdge, ProposedNode,
    ProposedNodeKind,
};
pub use reward::{RewardRecord, UtilityKind};
pub use timeline::{
    ExecutionError, ExecutionEvent, ExecutionRecord, LoopPolicy, Plan, Tick, TickEpoch,
};
pub use types::{
    Receiver, RefKind, ScalarType, StructKind, TypeKind, TypeRef, ValuePort, Visibility,
};
pub use word::{WORD_PATTERN, Word, WordError};
