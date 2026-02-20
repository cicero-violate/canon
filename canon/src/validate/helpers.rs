use super::error::{Violation, ViolationDetail};
use super::rules::CanonRule;
use crate::ir::*;
use std::collections::HashMap;
pub struct Indexes<'a> {
    pub modules: HashMap<&'a str, &'a Module>,
    pub structs: HashMap<&'a str, &'a Struct>,
    pub traits: HashMap<&'a str, &'a Trait>,
    pub impls: HashMap<&'a str, &'a ImplBlock>,
    pub functions: HashMap<&'a str, &'a Function>,
    pub deltas: HashMap<&'a str, &'a StateChange>,
    pub proofs: HashMap<&'a str, &'a Proof>,
    pub predicates: HashMap<&'a str, &'a Rule>,
    pub judgments: HashMap<&'a str, &'a Decision>,
    pub admissions: HashMap<&'a str, &'a ChangeAdmission>,
    pub tick_graphs: HashMap<&'a str, &'a ExecutionGraph>,
    pub ticks: HashMap<&'a str, &'a Tick>,
    pub epochs: HashMap<&'a str, &'a ExecutionEpoch>,
    pub plans: HashMap<&'a str, &'a Plan>,
    pub proposals: HashMap<&'a str, &'a Proposal>,
}
pub fn build_indexes<'a>(
    ir: &'a SystemState,
    violations: &mut Vec<Violation>,
) -> Indexes<'a> {
    Indexes {
        modules: index_by_id(
            &ir.modules,
            |m| m.id.as_str(),
            CanonRule::ExplicitArtifacts,
            "module",
            violations,
        ),
        structs: index_by_id(
            &ir.structs,
            |s| s.id.as_str(),
            CanonRule::ExplicitArtifacts,
            "struct",
            violations,
        ),
        traits: index_by_id(
            &ir.traits,
            |t| t.id.as_str(),
            CanonRule::ExplicitArtifacts,
            "trait",
            violations,
        ),
        impls: index_by_id(
            &ir.impls,
            |i| i.id.as_str(),
            CanonRule::ImplBinding,
            "impl",
            violations,
        ),
        functions: index_by_id(
            &ir.functions,
            |f| f.id.as_str(),
            CanonRule::ExecutionOnlyInImpl,
            "function",
            violations,
        ),
        deltas: index_by_id(
            &ir.deltas,
            |d| d.id.as_str(),
            CanonRule::EffectsAreDeltas,
            "delta",
            violations,
        ),
        proofs: index_by_id(
            &ir.proofs,
            |p| p.id.as_str(),
            CanonRule::DeltaProofs,
            "proof",
            violations,
        ),
        predicates: index_by_id(
            &ir.judgment_predicates,
            |p| p.id.as_str(),
            CanonRule::JudgmentDecisions,
            "judgment predicate",
            violations,
        ),
        judgments: index_by_id(
            &ir.judgments,
            |j| j.id.as_str(),
            CanonRule::JudgmentDecisions,
            "judgment",
            violations,
        ),
        admissions: index_by_id(
            &ir.admissions,
            |a| a.id.as_str(),
            CanonRule::AdmissionBridge,
            "admission",
            violations,
        ),
        tick_graphs: index_by_id(
            &ir.tick_graphs,
            |g| g.id.as_str(),
            CanonRule::TickGraphAcyclic,
            "tick graph",
            violations,
        ),
        ticks: index_by_id(
            &ir.ticks,
            |t| t.id.as_str(),
            CanonRule::TickRoot,
            "tick",
            violations,
        ),
        epochs: index_by_id(
            &ir.tick_epochs,
            |e| e.id.as_str(),
            CanonRule::TickEpochs,
            "tick epoch",
            violations,
        ),
        plans: index_by_id(
            &ir.plans,
            |p| p.id.as_str(),
            CanonRule::PlanArtifacts,
            "plan",
            violations,
        ),
        proposals: index_by_id(
            &ir.proposals,
            |p| p.id.as_str(),
            CanonRule::ProposalDeclarative,
            "proposal",
            violations,
        ),
    }
}
pub fn index_by_id<'a, T, F>(
    items: &'a [T],
    id_fn: F,
    rule: CanonRule,
    _kind: &str,
    violations: &mut Vec<Violation>,
) -> HashMap<&'a str, &'a T>
where
    F: Fn(&'a T) -> &'a str,
{
    let mut map = HashMap::new();
    for item in items {
        let id = id_fn(item);
        if map.insert(id, item).is_some() {
            violations
                .push(
                    Violation::structured(
                        rule,
                        id.to_string(),
                        ViolationDetail::Duplicate {
                            name: id.to_string(),
                        },
                    ),
                );
        }
    }
    map
}
pub fn pipeline_stage_allows(stage: PipelineStage, kind: DeltaKind) -> bool {
    match stage {
        PipelineStage::Observe => true,
        PipelineStage::Learn => {
            matches!(kind, DeltaKind::State | DeltaKind::Structure | DeltaKind::History)
        }
        PipelineStage::Decide => {
            matches!(kind, DeltaKind::Structure | DeltaKind::History)
        }
        PipelineStage::Plan => matches!(kind, DeltaKind::Structure | DeltaKind::History),
        PipelineStage::Act => {
            matches!(kind, DeltaKind::State | DeltaKind::Io | DeltaKind::History)
        }
    }
}
pub fn proof_scope_allows(kind: DeltaKind, scope: ProofScope) -> bool {
    match kind {
        DeltaKind::State => matches!(scope, ProofScope::Execution),
        DeltaKind::Io => matches!(scope, ProofScope::Execution),
        DeltaKind::Structure => matches!(scope, ProofScope::Structure | ProofScope::Law),
        DeltaKind::History => {
            matches!(scope, ProofScope::Structure | ProofScope::Execution)
        }
    }
}
pub fn module_has_permission<'a>(
    from: &'a str,
    to: &'a str,
    adjacency: &HashMap<&'a str, Vec<&'a str>>,
    cache: &mut HashMap<&'a str, std::collections::HashSet<&'a str>>,
) -> bool {
    if from == to {
        return true;
    }
    if cache.get(from).map(|s| s.contains(to)).unwrap_or(false) {
        return true;
    }
    let mut stack = vec![from];
    let mut seen = std::collections::HashSet::new();
    while let Some(node) = stack.pop() {
        if node == to {
            cache.entry(from).or_default().insert(to);
            return true;
        }
        if !seen.insert(node) {
            continue;
        }
        if let Some(neighbors) = adjacency.get(node) {
            for nb in neighbors {
                stack.push(nb);
            }
        }
    }
    false
}
