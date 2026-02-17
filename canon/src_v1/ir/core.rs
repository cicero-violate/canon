use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::reward::RewardRecord;
use super::world_model::WorldModel;
use super::{
    admission::{AppliedDeltaRecord, DeltaAdmission},
    artifacts::{EnumNode, ImplBlock, Module, ModuleEdge, Struct, Trait},
    delta::Delta,
    errors::ErrorArtifact,
    functions::Function,
    goals::GoalMutation,
    gpu::GpuFunction,
    graphs::{CallEdge, SystemGraph, TickGraph},
    ids::{PolicyParameterId, ProofId},
    judgment::{Judgment, JudgmentPredicate},
    learning::Learning,
    policy::PolicyParameters,
    project::{ExternalDependency, Project},
    proofs::Proof,
    proposal::Proposal,
    timeline::{ExecutionRecord, LoopPolicy, Plan, Tick, TickEpoch},
    word::Word,
};

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct CanonicalIr {
    pub meta: CanonicalMeta,
    pub version_contract: VersionContract,
    pub project: Project,
    pub modules: Vec<Module>,
    pub module_edges: Vec<ModuleEdge>,
    pub structs: Vec<Struct>,
    #[serde(default)]
    pub enums: Vec<EnumNode>,
    pub traits: Vec<Trait>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    pub call_edges: Vec<CallEdge>,
    pub tick_graphs: Vec<TickGraph>,
    #[serde(default)]
    pub system_graphs: Vec<SystemGraph>,
    pub loop_policies: Vec<LoopPolicy>,
    pub ticks: Vec<Tick>,
    pub tick_epochs: Vec<TickEpoch>,
    #[serde(default)]
    pub policy_parameters: Vec<PolicyParameters>,
    pub plans: Vec<Plan>,
    pub executions: Vec<ExecutionRecord>,
    pub admissions: Vec<DeltaAdmission>,
    pub applied_deltas: Vec<AppliedDeltaRecord>,
    pub gpu_functions: Vec<GpuFunction>,
    pub proposals: Vec<Proposal>,
    pub judgments: Vec<Judgment>,
    pub judgment_predicates: Vec<JudgmentPredicate>,
    pub deltas: Vec<Delta>,
    pub proofs: Vec<Proof>,
    pub learning: Vec<Learning>,
    pub errors: Vec<ErrorArtifact>,
    pub dependencies: Vec<ExternalDependency>,
    #[serde(default)]
    pub file_hashes: HashMap<String, String>,
    /// Append-only reward log: one RewardRecord per tick execution.
    #[serde(default)]
    pub reward_deltas: Vec<RewardRecord>,
    /// Predictive world model (Layer 2).
    #[serde(default)]
    pub world_model: WorldModel,
    #[serde(default)]
    pub goal_mutations: Vec<GoalMutation>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct CanonicalMeta {
    pub version: String,
    pub law_revision: Word,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct VersionContract {
    pub current: String,
    pub compatible_with: Vec<String>,
    pub migration_proofs: Vec<ProofId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    Observe,
    Learn,
    Decide,
    Plan,
    Act,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Rust,
}
