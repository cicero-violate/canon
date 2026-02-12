use once_cell::sync::Lazy;
use regex::Regex;
use schemars::{
    JsonSchema,
    schema::{InstanceType, Schema, SchemaObject, StringValidation},
};
use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value as JsonValue;
use std::fmt;
use thiserror::Error;

pub const WORD_PATTERN: &str = "^[A-Za-z][A-Za-z0-9]*$";

static WORD_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(WORD_PATTERN).expect("Canonical word pattern must compile"));

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Word(String);

impl Word {
    pub fn new(value: impl Into<String>) -> Result<Self, WordError> {
        let value = value.into();
        if WORD_REGEX.is_match(&value) {
            Ok(Self(value))
        } else {
            Err(WordError::Invalid(value))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Error)]
pub enum WordError {
    #[error("value `{0}` is not a single canonical word")]
    Invalid(String),
}

impl Serialize for Word {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for Word {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Word::new(value).map_err(D::Error::custom)
    }
}

impl JsonSchema for Word {
    fn schema_name() -> String {
        "Word".to_owned()
    }

    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> Schema {
        Schema::Object(SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                pattern: Some(WORD_PATTERN.to_owned()),
                ..Default::default()
            })),
            ..Default::default()
        })
    }
}

pub type ModuleId = String;

pub type StructId = String;
pub type TraitId = String;
pub type TraitFunctionId = String;
pub type ImplId = String;
pub type FunctionId = String;
pub type CallEdgeId = String;
pub type TickGraphId = String;
pub type SystemGraphId = String;
pub type SystemNodeId = String;
pub type LoopPolicyId = String;
pub type TickId = String;
pub type TickEpochId = String;
pub type PlanId = String;
pub type ExecutionRecordId = String;
pub type AdmissionId = String;
pub type AppliedDeltaId = String;
pub type ProposalId = String;
pub type JudgmentId = String;
pub type JudgmentPredicateId = String;
pub type LearningId = String;
pub type DeltaId = String;
pub type ProofId = String;
pub type ErrorId = String;
pub type GpuFunctionId = String;

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct CanonicalIr {
    pub meta: CanonicalMeta,
    pub version_contract: VersionContract,
    pub project: Project,
    pub modules: Vec<Module>,
    pub module_edges: Vec<ModuleEdge>,
    pub structs: Vec<Struct>,
    pub traits: Vec<Trait>,
    #[serde(rename = "impls")]
    pub impl_blocks: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    pub call_edges: Vec<CallEdge>,
    pub tick_graphs: Vec<TickGraph>,
    #[serde(default)]
    pub system_graphs: Vec<SystemGraph>,
    pub loop_policies: Vec<LoopPolicy>,
    pub ticks: Vec<Tick>,
    pub tick_epochs: Vec<TickEpoch>,
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
pub struct Module {
    pub id: ModuleId,
    pub name: Word,
    pub visibility: Visibility,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ModuleEdge {
    #[serde(rename = "from")]
    pub source: ModuleId,
    #[serde(rename = "to")]
    pub target: ModuleId,
    pub rationale: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Struct {
    pub id: StructId,
    pub name: Word,
    pub module: ModuleId,
    pub visibility: Visibility,
    pub fields: Vec<Field>,
    pub history: Vec<DeltaRef>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub name: Word,
    pub ty: TypeRef,
    pub visibility: Visibility,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Trait {
    pub id: TraitId,
    pub name: Word,
    pub module: ModuleId,
    pub visibility: Visibility,
    pub functions: Vec<TraitFunction>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct TraitFunction {
    pub id: TraitFunctionId,
    pub name: Word,
    pub inputs: Vec<ValuePort>,
    pub outputs: Vec<ValuePort>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ImplBlock {
    pub id: ImplId,
    pub module: ModuleId,
    pub struct_id: StructId,
    pub trait_id: TraitId,
    pub functions: Vec<ImplFunctionBinding>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ImplFunctionBinding {
    pub trait_fn: TraitFunctionId,
    pub function: FunctionId,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Function {
    pub id: FunctionId,
    pub name: Word,
    pub module: ModuleId,
    pub impl_id: ImplId,
    pub trait_function: TraitFunctionId,
    pub visibility: Visibility,
    pub inputs: Vec<ValuePort>,
    pub outputs: Vec<ValuePort>,
    pub deltas: Vec<DeltaRef>,
    pub contract: FunctionContract,
    #[serde(default)]
    pub metadata: FunctionMetadata,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ValuePort {
    pub name: Word,
    pub ty: TypeRef,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct TypeRef {
    pub name: Word,
    pub kind: TypeKind,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct DeltaRef {
    pub delta: DeltaId,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct FunctionContract {
    pub total: bool,
    pub deterministic: bool,
    pub explicit_inputs: bool,
    pub explicit_outputs: bool,
    pub effects_are_deltas: bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct FunctionMetadata {
    #[serde(default)]
    pub bytecode_b64: Option<String>,
    #[serde(default)]
    pub ast: Option<JsonValue>,
    #[serde(default)]
    pub postconditions: Vec<Postcondition>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Postcondition {
    NonNegative { output: Word },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct CallEdge {
    pub id: CallEdgeId,
    pub caller: FunctionId,
    pub callee: FunctionId,
    pub rationale: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct TickGraph {
    pub id: TickGraphId,
    pub name: Word,
    pub nodes: Vec<FunctionId>,
    pub edges: Vec<TickEdge>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct TickEdge {
    pub from: FunctionId,
    pub to: FunctionId,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct SystemGraph {
    pub id: SystemGraphId,
    pub name: Word,
    pub nodes: Vec<SystemNode>,
    pub edges: Vec<SystemEdge>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct SystemNode {
    pub id: SystemNodeId,
    pub function: FunctionId,
    pub kind: SystemNodeKind,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "snake_case")]
pub enum SystemNodeKind {
    Function,
    Gate,
    Persist,
    Materialize,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct SystemEdge {
    pub from: SystemNodeId,
    pub to: SystemNodeId,
    pub kind: SystemEdgeKind,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "snake_case")]
pub enum SystemEdgeKind {
    Control,
    Data,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct LoopPolicy {
    pub id: LoopPolicyId,
    pub graph: TickGraphId,
    pub continuation: JudgmentPredicateId,
    pub max_ticks: Option<u64>,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Tick {
    pub id: TickId,
    pub graph: TickGraphId,
    pub input_state: Vec<DeltaId>,
    pub output_deltas: Vec<DeltaId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct TickEpoch {
    pub id: TickEpochId,
    pub ticks: Vec<TickId>,
    pub parent_epoch: Option<TickEpochId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub id: PlanId,
    pub judgment: JudgmentId,
    pub steps: Vec<FunctionId>,
    pub expected_deltas: Vec<DeltaId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRecord {
    pub id: ExecutionRecordId,
    pub tick: TickId,
    pub plan: PlanId,
    pub outcome_deltas: Vec<DeltaId>,
    #[serde(default)]
    pub errors: Vec<ExecutionError>,
    #[serde(default)]
    pub events: Vec<ExecutionEvent>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ExecutionError {
    pub code: Word,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionEvent {
    Stdout { text: String },
    Stderr { text: String },
    Artifact { path: String, hash: String },
    Error { code: Word, message: String },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct DeltaAdmission {
    pub id: AdmissionId,
    pub judgment: JudgmentId,
    pub tick: TickId,
    pub delta_ids: Vec<DeltaId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct AppliedDeltaRecord {
    pub id: AppliedDeltaId,
    pub admission: AdmissionId,
    pub delta: DeltaId,
    pub order: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct GpuFunction {
    pub id: GpuFunctionId,
    pub function: FunctionId,
    pub inputs: Vec<VectorPort>,
    pub outputs: Vec<VectorPort>,
    pub properties: GpuProperties,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct VectorPort {
    pub name: Word,
    pub scalar: ScalarType,
    pub lanes: u32,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct GpuProperties {
    pub pure: bool,
    pub no_io: bool,
    pub no_alloc: bool,
    pub no_branch: bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub id: ProposalId,
    pub goal: ProposalGoal,
    pub nodes: Vec<ProposedNode>,
    pub apis: Vec<ProposedApi>,
    pub edges: Vec<ProposedEdge>,
    pub status: ProposalStatus,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposalGoal {
    pub id: Word,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedNode {
    pub id: Option<String>,
    pub name: Word,
    pub module: Option<ModuleId>,
    pub kind: ProposedNodeKind,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedApi {
    pub trait_id: TraitId,
    pub functions: Vec<TraitFunctionId>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProposedEdge {
    pub from: ModuleId,
    pub to: ModuleId,
    pub rationale: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Judgment {
    pub id: JudgmentId,
    pub proposal: ProposalId,
    pub predicate: JudgmentPredicateId,
    pub decision: JudgmentDecision,
    pub rationale: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct JudgmentPredicate {
    pub id: JudgmentPredicateId,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Learning {
    pub id: LearningId,
    pub proposal: ProposalId,
    pub new_rules: Vec<String>,
    pub notes: String,
    pub proof_object_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Delta {
    pub id: DeltaId,
    pub kind: DeltaKind,
    pub stage: PipelineStage,
    pub append_only: bool,
    pub proof: ProofId,
    pub description: String,
    pub related_function: Option<FunctionId>,
    pub payload: Option<DeltaPayload>,
    #[serde(default)]
    pub proof_object_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Proof {
    pub id: ProofId,
    pub invariant: String,
    pub scope: ProofScope,
    pub evidence: ProofArtifact,
    pub proof_object_hash: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ProofArtifact {
    pub uri: String,
    pub hash: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ErrorArtifact {
    pub id: ErrorId,
    pub rule: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct Project {
    pub name: Word,
    pub version: String,
    pub language: Language,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct ExternalDependency {
    pub name: Word,
    pub source: Word,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Public,
    Private,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Scalar,
    Struct,
    Trait,
    Delta,
    External,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    F32,
    F64,
    I32,
    U32,
    Bool,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposedNodeKind {
    Module,
    Struct,
    Trait,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProposalStatus {
    Draft,
    Submitted,
    Accepted,
    Rejected,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JudgmentDecision {
    Accept,
    Reject,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeltaKind {
    State,
    Io,
    Structure,
    History,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DeltaPayload {
    AddModule {
        module_id: ModuleId,
        name: Word,
        visibility: Visibility,
        description: String,
    },
    AddStruct {
        module: ModuleId,
        struct_id: StructId,
        name: Word,
    },
    AddField {
        struct_id: StructId,
        field: Field,
    },
    AddTrait {
        module: ModuleId,
        trait_id: TraitId,
        name: Word,
    },
    AddTraitFunction {
        trait_id: TraitId,
        function: TraitFunction,
    },
    AddImpl {
        module: ModuleId,
        impl_id: ImplId,
        struct_id: StructId,
        trait_id: TraitId,
    },
    AddFunction {
        function_id: FunctionId,
        impl_id: ImplId,
        signature: FunctionSignature,
    },
    AddModuleEdge {
        from: ModuleId,
        to: ModuleId,
        rationale: String,
    },
    AddCallEdge {
        caller: FunctionId,
        callee: FunctionId,
    },
    AttachExecutionEvent {
        execution_id: ExecutionRecordId,
        event: ExecutionEvent,
    },
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
pub struct FunctionSignature {
    pub name: Word,
    pub inputs: Vec<ValuePort>,
    pub outputs: Vec<ValuePort>,
    pub visibility: Visibility,
    pub trait_function: TraitFunctionId,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProofScope {
    Structure,
    Execution,
    Law,
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
