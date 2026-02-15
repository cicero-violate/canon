use super::rules::CanonRule;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViolationDetail {
    Message(String),

    MissingModule { module: String },
    MissingStruct { struct_id: String },
    MissingTrait { trait_id: String },
    MissingFunction { function_id: String },
    MissingDelta { delta_id: String },

    ModuleCycle,
    CallCycle,
    TickCycle { graph: String },

    PermissionDenied {
        caller_module: String,
        callee_module: String,
    },

    Duplicate { name: String },

    InvalidContract,

    DeltaReferencesUnknownFunction { delta: String, function_id: String },
    DeltaReferencesUnknownStruct { delta: String, struct_id: String },
    DeltaReferencesUnknownArtifact { delta: String, kind: String, id: String },
    DeltaMissingField { delta: String, struct_id: String, field: String },
    DeltaAppendOnlyViolation { delta: String },
    DeltaPipelineViolation { delta: String },
    DeltaMissingProof { delta: String, proof: String },
    ProofScopeViolation { delta: String },
    StructHistoryMissingDelta { struct_id: String, delta: String },
    AdmissionMissingTick { admission: String, tick: String },
    AdmissionMissingJudgment { admission: String, judgment: String },
    AdmissionNotAccepted { admission: String },
    AdmissionMissingDelta { admission: String, delta: String },
    AppliedMissingAdmission { applied: String, admission: String },
    AppliedMissingDelta { applied: String, delta: String },
    AppliedOrderViolation { applied: String },

    ProposalIncomplete { proposal: String },
    ProposalMissingGoal { proposal: String },
    ProposalInvalid { proposal: String },
    ProposalUnknownModule { proposal: String, module: String },
    ProposalUnknownTrait { proposal: String, trait_id: String },
    ProposalUnknownEdgeModule { proposal: String, module: String },

    JudgmentMissingProposal { judgment: String, proposal: String },
    JudgmentMissingPredicate { judgment: String, predicate: String },

    LearningMissingProposal { learning: String, proposal: String },
    LearningMissingRules { learning: String },
    LearningMissingProofObject { learning: String },

    GoalMutationMissingJudgment { mutation: String },
    GoalMutationMissingProof { mutation: String, proof: String },

    StructMissingModule { struct_id: String, module: String },
    TupleStructEmpty { struct_id: String },
    StructEmptyDerive { struct_id: String },

    EnumMissingModule { enum_id: String, module: String },

    TraitMissingModule { trait_id: String, module: String },
    TraitMissingSupertrait { trait_id: String, supertrait: String },
    TraitDuplicateFunction { function_id: String },

    ImplMissingStruct { impl_id: String, struct_id: String },
    ImplMissingTrait { impl_id: String, trait_id: String },
    ImplMissingModule { impl_id: String, module: String },
    ImplWrongModuleForStruct { impl_id: String },
    ImplWrongModuleForTrait { impl_id: String },
    ImplDuplicateBinding { impl_id: String, function: String },
    ImplWrongTraitFunction { impl_id: String, trait_fn: String },

    FunctionMissingImpl { function_id: String, impl_id: String },
    FunctionWrongModule { function_id: String, module: String },
    FunctionWrongTraitBinding { function_id: String },
    FunctionUnknownTraitFunction { function_id: String, trait_fn: String },
    FunctionContractViolation { function_id: String },
    FunctionMissingOutputs { function_id: String },
    FunctionDuplicateGeneric { function_id: String, generic: String },
    FunctionDuplicateLifetime { function_id: String, lifetime: String },
    FunctionMissingDelta { function_id: String, delta: String },

    ProjectLanguageInvalid,
    ProjectVersionMissing,
    VersionMismatch { expected: String, found: String },
    VersionDuplicateCompatibility { version: String },
    VersionMissingMigrationProof,
    DependencyMissingVersion { dependency: String },
    DependencyDuplicate { dependency: String },

    TraitFunctionDuplicate { function_id: String },

    ImplFunctionDuplicate { function_id: String },
    ImplTraitMismatch { impl_id: String, trait_fn: String },

    VersionProofWrongScope { proof: String },
    VersionProofMissing { proof: String },

    ModuleEdgeEmptyImport { source: String, target: String },

    EpochMissingTick { epoch: String, tick: String },
    EpochSelfParent { epoch: String },
    EpochMissingParent { epoch: String, parent: String },
    EpochCycle { epoch: String },

    TickMissingGraph { tick: String, graph: String },
    TickMissingDelta { tick: String, delta: String },

    PlanMissingJudgment { plan: String, judgment: String },
    PlanNotAccepted { plan: String },
    PlanMissingFunction { plan: String, function: String },
    PlanMissingDelta { plan: String, delta: String },

    ExecutionMissingTick { execution: String, tick: String },
    ExecutionMissingPlan { execution: String, plan: String },
    ExecutionMissingDelta { execution: String, delta: String },

    RewardDrop { subject: String },

    GpuMissingFunction { gpu: String, function: String },
    GpuMissingPorts { gpu: String },
    GpuInvalidLanes { gpu: String, port: String },
    GpuContractViolation { gpu: String },
}

#[derive(Debug, Clone)]
pub struct Violation {
    rule: CanonRule,
    detail: ViolationDetail,
    subject: String,
}

impl Violation {

    pub fn structured(
        rule: CanonRule,
        subject: impl Into<String>,
        detail: ViolationDetail,
    ) -> Self {
        Self {
            rule,
            detail,
            subject: subject.into(),
        }
    }

    pub fn rule(&self) -> CanonRule {
        self.rule
    }

    pub fn detail(&self) -> &str {
        match &self.detail {
            ViolationDetail::Message(msg) => msg.as_str(),
            other => {
                // temporary stringification fallback
                // ensures Display keeps working during migration
                match other {
                    ViolationDetail::MissingModule { module } =>
                        module.as_str(),
                    ViolationDetail::MissingStruct { struct_id } =>
                        struct_id.as_str(),
                    ViolationDetail::MissingTrait { trait_id } =>
                        trait_id.as_str(),
                    ViolationDetail::MissingFunction { function_id } =>
                        function_id.as_str(),
                    ViolationDetail::MissingDelta { delta_id } =>
                        delta_id.as_str(),
                    ViolationDetail::TickCycle { graph } =>
                        graph.as_str(),
                    ViolationDetail::Duplicate { name } =>
                        name.as_str(),
                    _ => "structured violation",
                }
            }
        }
    }

    /// Returns the primary subject artifact id for this violation.
    pub fn subject_id(&self) -> Option<&str> {
        Some(self.subject.as_str())
    }
}

#[derive(Debug)]
pub struct ValidationErrors {
    violations: Vec<Violation>,
}

impl ValidationErrors {
    pub fn new(violations: Vec<Violation>) -> Self {
        Self { violations }
    }

    pub fn violations(&self) -> &[Violation] {
        &self.violations
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Canon validation failed with {} violation(s):",
            self.violations.len()
        )?;
        for v in &self.violations {
            writeln!(
                f,
                "- {} ({}) → {}",
                v.rule().code(),
                v.rule().text(),
                v.detail()
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}
