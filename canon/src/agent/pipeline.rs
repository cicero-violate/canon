//! RefactorProposal pipeline: Observe → Reason → Prove → Judge → Mutate.
//!
//! Pure data pipeline — no LLM calls, no async.
//! Each stage validates the previous stage's output before proceeding.
//! If any stage fails the pipeline halts with a typed PipelineError.
//!
//! The caller drives each stage by supplying AgentCallOutputs from their
//! LLM client. The pipeline assembles them into IR mutations via the
//! existing accept_proposal + apply_deltas + Lyapunov gate chain.

use serde_json::Value;

use crate::{
    evolution::{DEFAULT_TOPOLOGY_THETA, EvolutionError, apply_deltas, check_topology_drift},
    ir::CanonicalIr,
    layout::LayoutGraph,
};

use super::call::AgentCallOutput;
use super::refactor::RefactorProposal;

/// Which stage the pipeline is currently at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefactorStage {
    Observe,
    Reason,
    Prove,
    Judge,
    Mutate,
    Complete,
}

impl std::fmt::Display for RefactorStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RefactorStage::Observe => write!(f, "Observe"),
            RefactorStage::Reason => write!(f, "Reason"),
            RefactorStage::Prove => write!(f, "Prove"),
            RefactorStage::Judge => write!(f, "Judge"),
            RefactorStage::Mutate => write!(f, "Mutate"),
            RefactorStage::Complete => write!(f, "Complete"),
        }
    }
}

/// Typed error for each pipeline stage.
#[derive(Debug)]
pub enum PipelineError {
    /// A required field was missing from an AgentCallOutput payload.
    MissingPayloadField { stage: RefactorStage, field: String },
    /// The Prover stage did not populate proof_id on the proposal.
    MissingProof,
    /// The Judge stage rejected the proposal.
    Rejected { rationale: String },
    /// The Lyapunov gate blocked the mutation.
    TopologyDrift(crate::evolution::LyapunovError),
    /// apply_deltas failed.
    Evolution(EvolutionError),
    /// No admission_id was found in the Judge output payload.
    MissingAdmission,
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::MissingPayloadField { stage, field } => {
                write!(f, "stage {stage}: missing payload field `{field}`")
            }
            PipelineError::MissingProof => write!(f, "Prove stage: proof_id not populated"),
            PipelineError::Rejected { rationale } => {
                write!(f, "Judge stage: proposal rejected — {rationale}")
            }
            PipelineError::TopologyDrift(e) => write!(f, "Mutate stage: {e}"),
            PipelineError::Evolution(e) => write!(f, "Mutate stage: {e}"),
            PipelineError::MissingAdmission => {
                write!(f, "Judge stage: admission_id not found in payload")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

/// Result of a completed pipeline run.
#[derive(Debug)]
pub struct PipelineResult {
    /// The mutated IR after all stages completed successfully.
    pub ir: CanonicalIr,
    /// The layout after mutation.
    pub layout: LayoutGraph,
    /// The refactor proposal that was applied.
    pub proposal: RefactorProposal,
    /// Admission id recorded in the IR.
    pub admission_id: String,
}

/// Drives a RefactorProposal through the Observe→Reason→Prove→Judge→Mutate pipeline.
///
/// Each `stage_output` corresponds to one LLM call result in order:
/// [0] = Observer output, [1] = Reasoner output, [2] = Prover output,
/// [3] = Judge output. The Mutate stage is driven internally.
///
/// Returns PipelineResult on success, PipelineError on first failure.
pub fn run_pipeline(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    mut proposal: RefactorProposal,
    stage_outputs: &[AgentCallOutput],
) -> Result<PipelineResult, PipelineError> {
    // --- Observe stage ---
    // Observer output must confirm it found relevant IR artifacts.
    let observer_out = stage_output(stage_outputs, 0, RefactorStage::Observe)?;
    extract_str_field(&observer_out.payload, "observation", RefactorStage::Observe)?;

    // --- Reason stage ---
    // Reasoner output must include a rationale string.
    let reasoner_out = stage_output(stage_outputs, 1, RefactorStage::Reason)?;
    let rationale =
        extract_str_field(&reasoner_out.payload, "rationale", RefactorStage::Reason)?;
    proposal.rationale = rationale;

    // --- Prove stage ---
    // Prover output must include a proof_id.
    let prover_out = stage_output(stage_outputs, 2, RefactorStage::Prove)?;
    let proof_id = prover_out
        .proof_id
        .clone()
        .or_else(|| {
            prover_out
                .payload
                .get("proof_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .ok_or(PipelineError::MissingProof)?;
    proposal.proof_id = Some(proof_id);

    // --- Judge stage ---
    // Judge output must include decision = "accept" | "reject" and admission_id.
    let judge_out = stage_output(stage_outputs, 3, RefactorStage::Judge)?;
    let decision =
        extract_str_field(&judge_out.payload, "decision", RefactorStage::Judge)?;
    if decision.to_lowercase() != "accept" {
        let rationale = judge_out
            .payload
            .get("rationale")
            .and_then(|v| v.as_str())
            .unwrap_or("no rationale provided")
            .to_string();
        return Err(PipelineError::Rejected { rationale });
    }
    let admission_id = judge_out
        .payload
        .get("admission_id")
        .and_then(|v| v.as_str())
        .ok_or(PipelineError::MissingAdmission)?
        .to_string();

    // --- Mutate stage ---
    // Run Lyapunov gate then apply_deltas.
    let proof_ids: Vec<String> = ir.proofs.iter().map(|p| p.id.clone()).collect();
    let candidate = apply_deltas(ir, &[admission_id.clone()])
        .map_err(PipelineError::Evolution)?;
    check_topology_drift(ir, &candidate, &proof_ids, DEFAULT_TOPOLOGY_THETA)
        .map_err(PipelineError::TopologyDrift)?;

    // Sync layout — carry forward existing layout for now.
    let next_layout = layout.clone();

    Ok(PipelineResult {
        ir: candidate,
        layout: next_layout,
        proposal,
        admission_id,
    })
}

// --- helpers ---

fn stage_output(
    outputs: &[AgentCallOutput],
    idx: usize,
    stage: RefactorStage,
) -> Result<&AgentCallOutput, PipelineError> {
    outputs.get(idx).ok_or_else(|| PipelineError::MissingPayloadField {
        stage,
        field: format!("stage_outputs[{idx}]"),
    })
}

fn extract_str_field<'a>(
    payload: &'a Value,
    field: &str,
    stage: RefactorStage,
) -> Result<String, PipelineError> {
    payload
        .get(field)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| PipelineError::MissingPayloadField {
            stage,
            field: field.to_string(),
        })
}
