use thiserror::Error;

use crate::{
    dot_import::{
        DotImportError, dot_graph_to_file_topology, dot_graph_to_imported_types,
        dot_graph_to_proposal, parse_dot,
    },
    ir::CanonicalIr,
    proposal::sanitize_identifier,
};

use super::{
    DSL_PREDICATE_ID, DSL_PROOF_ID, DSL_TICK_ID,
    accept::{AcceptProposalError, ProposalAcceptance, ProposalAcceptanceInput, accept_proposal},
    bootstrap::{ensure_dsl_predicate, ensure_dsl_proof, ensure_dsl_tick},
};

#[derive(Debug, Error)]
pub enum AutoAcceptDotError {
    #[error(transparent)]
    Import(#[from] DotImportError),
    #[error(transparent)]
    Accept(#[from] AcceptProposalError),
    #[error("dot graph contains no clusters")]
    Empty,
    #[error("unable to infer tick graph for bootstrap tick")]
    MissingTickGraph,
}

/// Parse a DOT source, bootstrap proof/predicate/tick if absent,
/// accept the generated proposal, then patch file topology and
/// imported_types onto the resulting IR — all in one call.
pub fn auto_accept_dot_proposal(
    ir: &CanonicalIr,
    dot_source: &str,
    goal: &str,
) -> Result<ProposalAcceptance, AutoAcceptDotError> {
    let graph = parse_dot(dot_source)?;
    if graph.clusters.is_empty() {
        return Err(AutoAcceptDotError::Empty);
    }

    let proposal = dot_graph_to_proposal(&graph, goal)?;
    let proposal_id = proposal.id.clone();
    let goal_slug = sanitize_identifier(goal);
    let judgment_id = format!("judgment.dot.{goal_slug}");
    let admission_id = format!("admission.dot.{goal_slug}");

    let mut working = ir.clone();
    ensure_dsl_proof(&mut working);
    ensure_dsl_predicate(&mut working);
    ensure_dsl_tick(&mut working).map_err(|_| AutoAcceptDotError::MissingTickGraph)?;
    working.proposals.push(proposal);

    let mut acceptance = accept_proposal(
        &working,
        ProposalAcceptanceInput {
            proposal_id,
            proof_id: DSL_PROOF_ID.to_string(),
            predicate_id: DSL_PREDICATE_ID.to_string(),
            judgment_id,
            admission_id,
            tick_id: DSL_TICK_ID.to_string(),
            rationale: format!("Auto-accepted DOT proposal for goal `{goal}`."),
        },
    )?;

    // patch file topology
    let topology = dot_graph_to_file_topology(&graph);
    for module in acceptance.ir.modules.iter_mut() {
        if let Some((files, file_edges)) = topology.get(&module.id) {
            module.files = files.clone();
            module.file_edges = file_edges.clone();
        }
    }

    // patch imported_types
    let type_map = dot_graph_to_imported_types(&graph);
    for edge in acceptance.ir.module_edges.iter_mut() {
        if let Some(types) = type_map.get(&(edge.source.clone(), edge.target.clone())) {
            edge.imported_types = types.clone();
        }
    }

    Ok(acceptance)
}
