use thiserror::Error;

use crate::{
    ir::CanonicalIr,
    layout::LayoutGraph,
    ir::proposal::{DslProposalArtifacts, DslProposalError, create_proposal_from_dsl},
};

use super::{
    DSL_PREDICATE_ID, DSL_PROOF_ID, DSL_TICK_ID,
    accept::{AcceptProposalError, ProposalAcceptance, ProposalAcceptanceInput, accept_proposal},
    bootstrap::{ensure_dsl_predicate, ensure_dsl_proof, ensure_dsl_tick},
};

#[derive(Debug, Error)]
pub enum AutoAcceptDslError {
    #[error(transparent)]
    Proposal(#[from] DslProposalError),
    #[error(transparent)]
    Accept(#[from] AcceptProposalError),
    #[error("unable to infer tick graph for bootstrap tick")]
    MissingTickGraph,
}

pub fn auto_accept_dsl_proposal(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    dsl_source: &str,
) -> Result<ProposalAcceptance, AutoAcceptDslError> {
    let DslProposalArtifacts {
        proposal,
        goal_slug,
    } = create_proposal_from_dsl(dsl_source)?;
    let mut working = ir.clone();
    ensure_dsl_proof(&mut working);
    ensure_dsl_predicate(&mut working);
    ensure_dsl_tick(&mut working).map_err(|_| AutoAcceptDslError::MissingTickGraph)?;
    working.proposals.push(proposal);

    let proposal_id = format!("proposal.dsl.{goal_slug}");
    let judgment = format!("judgment.dsl.{goal_slug}");
    let admission = format!("admission.dsl.{goal_slug}");
    let acceptance = accept_proposal(
        &working,
        layout,
        ProposalAcceptanceInput {
            proposal_id,
            proof_id: DSL_PROOF_ID.to_string(),
            predicate_id: DSL_PREDICATE_ID.to_string(),
            judgment_id: judgment,
            admission_id: admission,
            tick_id: DSL_TICK_ID.to_string(),
            rationale: "Auto-accepted DSL proposal.".to_string(),
        },
    )?;
    Ok(acceptance)
}
