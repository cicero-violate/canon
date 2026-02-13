use std::collections::HashSet;

use thiserror::Error;

use crate::{
    evolution::{EvolutionError, apply_deltas},
    ir::{
        CanonicalIr, DeltaAdmission, DeltaId, Judgment, JudgmentDecision, ProposalStatus, WordError,
    },
    proof::smt_bridge::{SmtError, attach_function_proofs},
    proposal::{ProposalResolutionError, resolve_proposal_nodes},
};

use self::{
    delta_emitter::{build_trait_function_map, emit_deltas},
    proposal_checks::{
        enforce_proposal_ready, enforce_references, ensure_predicate_exists, ensure_proof_exists,
        ensure_tick_exists, ensure_unique_admission, ensure_unique_judgment,
    },
};

mod delta_emitter;
mod proposal_checks;

#[derive(Debug, Clone)]
pub struct ProposalAcceptanceInput {
    pub proposal_id: String,
    pub proof_id: String,
    pub predicate_id: String,
    pub judgment_id: String,
    pub admission_id: String,
    pub tick_id: String,
    pub rationale: String,
}

#[derive(Debug, Clone)]
pub struct ProposalAcceptance {
    pub ir: CanonicalIr,
    pub delta_ids: Vec<DeltaId>,
    pub judgment_id: String,
    pub admission_id: String,
}

#[derive(Debug, Error)]
pub enum AcceptProposalError {
    #[error("proposal `{0}` does not exist")]
    UnknownProposal(String),
    #[error("proposal `{proposal}` must be submitted; found status `{status:?}`")]
    InvalidProposalStatus {
        proposal: String,
        status: ProposalStatus,
    },
    #[error("proposal `{0}` must enumerate nodes, APIs, and edges")]
    IncompleteProposal(String),
    #[error("proof `{0}` is not registered")]
    UnknownProof(String),
    #[error("judgment predicate `{0}` is not registered")]
    UnknownPredicate(String),
    #[error("tick `{0}` is not registered")]
    UnknownTick(String),
    #[error("judgment `{0}` already exists")]
    DuplicateJudgment(String),
    #[error("admission `{0}` already exists")]
    DuplicateAdmission(String),
    #[error("delta `{0}` already exists")]
    DuplicateDelta(String),
    #[error("proposal `{0}` did not emit any structural deltas")]
    NoDeltas(String),
    #[error("module `{0}` referenced by proposal is unknown")]
    UnknownModule(String),
    #[error("trait `{0}` referenced by proposal is unknown")]
    UnknownTrait(String),
    #[error("proposal referenced trait `{trait_id}` without declaring any functions")]
    EmptyApi { trait_id: String },
    #[error("artifact `{kind}` with id `{id}` already exists in Canon")]
    ArtifactExists { kind: &'static str, id: String },
    #[error(transparent)]
    Resolution(#[from] ProposalResolutionError),
    #[error(transparent)]
    Evolution(#[from] EvolutionError),
    #[error("word error: {0}")]
    Word(#[from] WordError),
    #[error(transparent)]
    Proof(#[from] SmtError),
}

pub fn accept_proposal(
    ir: &CanonicalIr,
    input: ProposalAcceptanceInput,
) -> Result<ProposalAcceptance, AcceptProposalError> {
    let mut working = ir.clone();
    let proposal_index = working
        .proposals
        .iter()
        .position(|proposal| proposal.id == input.proposal_id)
        .ok_or_else(|| AcceptProposalError::UnknownProposal(input.proposal_id.clone()))?;

    let proposal = working.proposals.get(proposal_index).expect("index");
    enforce_proposal_ready(proposal)?;
    let resolved = resolve_proposal_nodes(proposal)?;
    let trait_function_map = build_trait_function_map(proposal)?;
    enforce_references(ir, proposal, &resolved, &trait_function_map)?;
    ensure_proof_exists(&working, &input.proof_id)?;
    ensure_predicate_exists(&working, &input.predicate_id)?;
    ensure_tick_exists(&working, &input.tick_id)?;
    ensure_unique_judgment(&working, &input.judgment_id)?;
    ensure_unique_admission(&working, &input.admission_id)?;

    let mut known_delta_ids: HashSet<String> =
        working.deltas.iter().map(|d| d.id.clone()).collect();
    let (mut deltas, delta_ids) = emit_deltas(
        &input,
        proposal,
        &resolved,
        &trait_function_map,
        &mut known_delta_ids,
    )?;
    if deltas.is_empty() {
        return Err(AcceptProposalError::NoDeltas(proposal.id.clone()));
    }

    working.deltas.append(&mut deltas);
    working.proposals[proposal_index].status = ProposalStatus::Accepted;
    working.judgments.push(Judgment {
        id: input.judgment_id.clone(),
        proposal: input.proposal_id.clone(),
        predicate: input.predicate_id.clone(),
        decision: JudgmentDecision::Accept,
        rationale: input.rationale.clone(),
    });
    working.admissions.push(DeltaAdmission {
        id: input.admission_id.clone(),
        judgment: input.judgment_id.clone(),
        tick: input.tick_id.clone(),
        delta_ids: delta_ids.clone(),
    });

    let mut evolved = apply_deltas(&working, &[input.admission_id.clone()])?;
    attach_function_proofs(&mut evolved)?;
    Ok(ProposalAcceptance {
        ir: evolved,
        delta_ids,
        judgment_id: input.judgment_id,
        admission_id: input.admission_id,
    })
}
