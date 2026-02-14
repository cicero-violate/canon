//! IR slice builder — extracts a typed CanonicalIr subset per IrField declaration.
//!
//! This is the enforcement mechanism for stateless call safety.
//! A node literally cannot see fields it did not declare in CapabilityNode::reads.

use serde_json::{Map, Value};

use crate::ir::CanonicalIr;

use super::capability::IrField;

/// Builds a JSON object containing only the IR fields listed in `fields`.
/// The returned Value is what gets handed to AgentCallInput::ir_slice.
pub fn build_ir_slice(ir: &CanonicalIr, fields: &[IrField]) -> Value {
    let mut map = Map::new();
    for field in fields {
        let (key, value) = extract_field(ir, field);
        map.insert(key, value);
    }
    Value::Object(map)
}

/// Extracts a single IrField from CanonicalIr as a (key, JSON value) pair.
fn extract_field(ir: &CanonicalIr, field: &IrField) -> (String, Value) {
    match field {
        IrField::Modules => (
            "modules".into(),
            serde_json::to_value(&ir.modules).unwrap_or(Value::Null),
        ),
        IrField::ModuleEdges => (
            "module_edges".into(),
            serde_json::to_value(&ir.module_edges).unwrap_or(Value::Null),
        ),
        IrField::Structs => (
            "structs".into(),
            serde_json::to_value(&ir.structs).unwrap_or(Value::Null),
        ),
        IrField::Enums => (
            "enums".into(),
            serde_json::to_value(&ir.enums).unwrap_or(Value::Null),
        ),
        IrField::Traits => (
            "traits".into(),
            serde_json::to_value(&ir.traits).unwrap_or(Value::Null),
        ),
        IrField::ImplBlocks => (
            "impl_blocks".into(),
            serde_json::to_value(&ir.impl_blocks).unwrap_or(Value::Null),
        ),
        IrField::Functions => (
            "functions".into(),
            serde_json::to_value(&ir.functions).unwrap_or(Value::Null),
        ),
        IrField::CallEdges => (
            "call_edges".into(),
            serde_json::to_value(&ir.call_edges).unwrap_or(Value::Null),
        ),
        IrField::TickGraphs => (
            "tick_graphs".into(),
            serde_json::to_value(&ir.tick_graphs).unwrap_or(Value::Null),
        ),
        IrField::SystemGraphs => (
            "system_graphs".into(),
            serde_json::to_value(&ir.system_graphs).unwrap_or(Value::Null),
        ),
        IrField::LoopPolicies => (
            "loop_policies".into(),
            serde_json::to_value(&ir.loop_policies).unwrap_or(Value::Null),
        ),
        IrField::Ticks => (
            "ticks".into(),
            serde_json::to_value(&ir.ticks).unwrap_or(Value::Null),
        ),
        IrField::TickEpochs => (
            "tick_epochs".into(),
            serde_json::to_value(&ir.tick_epochs).unwrap_or(Value::Null),
        ),
        IrField::PolicyParameters => (
            "policy_parameters".into(),
            serde_json::to_value(&ir.policy_parameters).unwrap_or(Value::Null),
        ),
        IrField::Plans => (
            "plans".into(),
            serde_json::to_value(&ir.plans).unwrap_or(Value::Null),
        ),
        IrField::Executions => (
            "executions".into(),
            serde_json::to_value(&ir.executions).unwrap_or(Value::Null),
        ),
        IrField::Admissions => (
            "admissions".into(),
            serde_json::to_value(&ir.admissions).unwrap_or(Value::Null),
        ),
        IrField::AppliedDeltas => (
            "applied_deltas".into(),
            serde_json::to_value(&ir.applied_deltas).unwrap_or(Value::Null),
        ),
        IrField::GpuFunctions => (
            "gpu_functions".into(),
            serde_json::to_value(&ir.gpu_functions).unwrap_or(Value::Null),
        ),
        IrField::Proposals => (
            "proposals".into(),
            serde_json::to_value(&ir.proposals).unwrap_or(Value::Null),
        ),
        IrField::Judgments => (
            "judgments".into(),
            serde_json::to_value(&ir.judgments).unwrap_or(Value::Null),
        ),
        IrField::JudgmentPredicates => (
            "judgment_predicates".into(),
            serde_json::to_value(&ir.judgment_predicates).unwrap_or(Value::Null),
        ),
        IrField::Deltas => (
            "deltas".into(),
            serde_json::to_value(&ir.deltas).unwrap_or(Value::Null),
        ),
        IrField::Proofs => (
            "proofs".into(),
            serde_json::to_value(&ir.proofs).unwrap_or(Value::Null),
        ),
        IrField::Learning => (
            "learning".into(),
            serde_json::to_value(&ir.learning).unwrap_or(Value::Null),
        ),
        IrField::Errors => (
            "errors".into(),
            serde_json::to_value(&ir.errors).unwrap_or(Value::Null),
        ),
        IrField::Dependencies => (
            "dependencies".into(),
            serde_json::to_value(&ir.dependencies).unwrap_or(Value::Null),
        ),
        IrField::FileHashes => (
            "file_hashes".into(),
            serde_json::to_value(&ir.file_hashes).unwrap_or(Value::Null),
        ),
        IrField::RewardDeltas => (
            "reward_deltas".into(),
            serde_json::to_value(&ir.reward_deltas).unwrap_or(Value::Null),
        ),
        IrField::WorldModel => (
            "world_model".into(),
            serde_json::to_value(&ir.world_model).unwrap_or(Value::Null),
        ),
        IrField::GoalMutations => (
            "goal_mutations".into(),
            serde_json::to_value(&ir.goal_mutations).unwrap_or(Value::Null),
        ),
    }
}
