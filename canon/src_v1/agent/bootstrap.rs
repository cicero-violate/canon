//! Bootstrap helpers — build the standard 5-node capability graph and
//! a seed RefactorProposal so the runner loop has something to dispatch.
//!
//! Standard graph topology:
//!
//!   Observer(Observe) → Reasoner(Learn) → Prover(Decide) → Judge(Plan) → Mutator(Act)
//!
//! Each node is assigned the minimum IrField reads required for its role.
//! Edges carry proof_confidence = 0.7 (unverified but trusted by default).
//!
//! No LLM calls. No unsafe. Pure construction.

use crate::ir::PipelineStage;

use super::capability::{CapabilityEdge, CapabilityGraph, CapabilityKind, CapabilityNode, IrField};
use super::refactor::{RefactorKind, RefactorProposal, RefactorTarget};

/// Default proof_confidence on all bootstrapped edges.
/// Must satisfy c^(chain_length-1) >= base_trust_threshold.
/// For a 5-node linear chain with threshold 0.5: c >= 0.5^(1/4) ≈ 0.841.
const DEFAULT_EDGE_CONFIDENCE: f64 = 0.85;

/// Build the standard 5-node Observer→Reasoner→Prover→Judge→Mutator graph.
///
/// This is the minimum viable graph for one full pipeline run.
/// All edges carry DEFAULT_EDGE_CONFIDENCE.
pub fn bootstrap_graph() -> CapabilityGraph {
    let mut g = CapabilityGraph::new();

    // --- Nodes ---

    // Observer: reads the structural shape of the IR.
    // Needs modules, functions, call edges, structs, traits to produce
    // a hottest-modules / largest-structs observation.
    g.add_node(CapabilityNode {
        id: "observer".to_string(),
        kind: CapabilityKind::Observer,
        label: "IR Observer".to_string(),
        reads: vec![
            IrField::Modules,
            IrField::ModuleEdges,
            IrField::Functions,
            IrField::CallEdges,
            IrField::Structs,
            IrField::Traits,
            IrField::Deltas,
            IrField::Errors,
        ],
        writes: vec![],
        stage: PipelineStage::Observe,
    });

    // Reasoner: reads observation context + proposals + reward history
    // to produce a rationale and proposed refactor kind.
    g.add_node(CapabilityNode {
        id: "reasoner".to_string(),
        kind: CapabilityKind::Reasoner,
        label: "Refactor Reasoner".to_string(),
        reads: vec![
            IrField::Modules,
            IrField::Functions,
            IrField::Proposals,
            IrField::Judgments,
            IrField::RewardDeltas,
            IrField::PolicyParameters,
            IrField::Learning,
        ],
        writes: vec![IrField::Proposals],
        stage: PipelineStage::Learn,
    });

    // Prover: reads proofs + deltas + functions to generate or verify
    // a proof_id for the proposal.
    g.add_node(CapabilityNode {
        id: "prover".to_string(),
        kind: CapabilityKind::Prover,
        label: "SMT Prover".to_string(),
        reads: vec![
            IrField::Proofs,
            IrField::Deltas,
            IrField::Functions,
            IrField::Proposals,
        ],
        writes: vec![IrField::Proofs],
        stage: PipelineStage::Decide,
    });

    // Judge: reads judgments + admissions + predicates to accept or
    // reject the proposal and emit an admission_id.
    g.add_node(CapabilityNode {
        id: "judge".to_string(),
        kind: CapabilityKind::Judge,
        label: "Proposal Judge".to_string(),
        reads: vec![
            IrField::Judgments,
            IrField::JudgmentPredicates,
            IrField::Admissions,
            IrField::Proposals,
            IrField::Proofs,
        ],
        writes: vec![IrField::Judgments, IrField::Admissions],
        stage: PipelineStage::Plan,
    });

    // Mutator: reads applied deltas + admissions to confirm mutation
    // was applied correctly. Writes nothing (apply_deltas is internal).
    g.add_node(CapabilityNode {
        id: "mutator".to_string(),
        kind: CapabilityKind::Mutator,
        label: "Delta Mutator".to_string(),
        reads: vec![
            IrField::Admissions,
            IrField::AppliedDeltas,
            IrField::Deltas,
            IrField::Modules,
            IrField::Functions,
        ],
        writes: vec![IrField::AppliedDeltas],
        stage: PipelineStage::Act,
    });

    // --- Edges (linear chain) ---
    g.add_edge(CapabilityEdge {
        from: "observer".to_string(),
        to: "reasoner".to_string(),
        proof_confidence: DEFAULT_EDGE_CONFIDENCE,
    });
    g.add_edge(CapabilityEdge {
        from: "reasoner".to_string(),
        to: "prover".to_string(),
        proof_confidence: DEFAULT_EDGE_CONFIDENCE,
    });
    g.add_edge(CapabilityEdge {
        from: "prover".to_string(),
        to: "judge".to_string(),
        proof_confidence: DEFAULT_EDGE_CONFIDENCE,
    });
    g.add_edge(CapabilityEdge {
        from: "judge".to_string(),
        to: "mutator".to_string(),
        proof_confidence: DEFAULT_EDGE_CONFIDENCE,
    });

    g
}

/// Build a seed RefactorProposal for the first pipeline run.
///
/// Targets the first module in the IR by convention.
/// The runner increments the id each tick so proposals stay unique.
pub fn bootstrap_proposal(target_module_id: &str) -> RefactorProposal {
    RefactorProposal::new(
        "seed",
        RefactorKind::SplitModule,
        RefactorTarget {
            artifact_id: target_module_id.to_string(),
            artifact_kind: "module".to_string(),
        },
        "Bootstrap proposal — Observer will refine this rationale on first run.",
        PipelineStage::Observe,
    )
}
