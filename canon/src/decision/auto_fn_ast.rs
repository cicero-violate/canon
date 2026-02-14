use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::{
    ir::{
        CanonicalIr, Delta, DeltaKind, DeltaPayload, PipelineStage, Proposal, ProposalGoal,
        ProposalKind, ProposalStatus, ProposedApi, ProposedNode, ProposedNodeKind,
    },
    layout::LayoutGraph,
    ir::proposal::{derive_word_from_identifier, sanitize_identifier},
};

use super::{
    DSL_PREDICATE_ID, DSL_PROOF_ID, DSL_TICK_ID,
    accept::{AcceptProposalError, ProposalAcceptance, ProposalAcceptanceInput, accept_proposal},
    bootstrap::{ensure_dsl_predicate, ensure_dsl_proof, ensure_dsl_tick},
};

#[derive(Debug, Error)]
pub enum AutoAcceptFnAstError {
    #[error(transparent)]
    Accept(#[from] AcceptProposalError),
    #[error("function `{0}` does not exist in IR")]
    UnknownFunction(String),
    #[error("unable to infer tick graph for bootstrap tick")]
    MissingTickGraph,
}

/// Wrap an AST update for `function_id` in a full proposal+delta pipeline
/// and auto-accept it, returning the evolved IR with the new function body.
pub fn auto_accept_fn_ast(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    function_id: &str,
    ast: JsonValue,
) -> Result<ProposalAcceptance, AutoAcceptFnAstError> {
    // ensure the function exists before building the proposal
    if !ir.functions.iter().any(|f| f.id == function_id) {
        return Err(AutoAcceptFnAstError::UnknownFunction(
            function_id.to_string(),
        ));
    }

    let fn_slug = sanitize_identifier(function_id);
    let proposal_id = format!("proposal.fn_ast.{fn_slug}");
    let judgment_id = format!("judgment.fn_ast.{fn_slug}");
    let admission_id = format!("admission.fn_ast.{fn_slug}");
    let delta_id = format!("delta.fn_ast.{fn_slug}");

    // build the delta directly — no proposal nodes/edges needed for a body update
    let delta = Delta {
        id: delta_id.clone(),
        kind: DeltaKind::Structure,
        stage: PipelineStage::Act,
        append_only: false,
        proof: DSL_PROOF_ID.to_string(),
        description: format!("Update AST body for function `{function_id}`."),
        related_function: Some(function_id.to_string()),
        payload: Some(DeltaPayload::UpdateFunctionAst {
            function_id: function_id.to_string(),
            ast,
        }),
        proof_object_hash: None,
    };

    let mut working = ir.clone();
    ensure_dsl_proof(&mut working);
    ensure_dsl_predicate(&mut working);
    ensure_dsl_tick(&mut working).map_err(|_| AutoAcceptFnAstError::MissingTickGraph)?;

    // build a minimal structural proposal so accept_proposal is satisfied
    let fn_word = derive_word_from_identifier(function_id)
        .map_err(|e| AutoAcceptFnAstError::Accept(AcceptProposalError::Resolution(e)))?;
    let module_id = working
        .functions
        .iter()
        .find(|f| f.id == function_id)
        .map(|f| f.module.clone())
        .unwrap_or_default();
    let trait_id = format!("trait.fn_ast.{fn_slug}");
    let trait_fn_id = format!("trait_fn.fn_ast.{fn_slug}");

    working.proposals.push(Proposal {
        id: proposal_id.clone(),
        kind: ProposalKind::FunctionBody,
        goal: ProposalGoal {
            id: fn_word.clone(),
            description: format!("AST update for function `{function_id}`."),
        },
        nodes: vec![ProposedNode {
            id: Some(trait_id.clone()),
            name: fn_word,
            module: Some(module_id),
            kind: ProposedNodeKind::Trait,
        }],
        apis: vec![ProposedApi {
            trait_id: trait_id.clone(),
            functions: vec![trait_fn_id],
        }],
        edges: vec![],
        status: ProposalStatus::Submitted,
    });

    // inject the delta before acceptance so apply_deltas can find it
    working.deltas.push(delta);

    let mut acceptance = accept_proposal(
        &working,
        layout,
        ProposalAcceptanceInput {
            proposal_id,
            proof_id: DSL_PROOF_ID.to_string(),
            predicate_id: DSL_PREDICATE_ID.to_string(),
            judgment_id,
            admission_id,
            tick_id: DSL_TICK_ID.to_string(),
            rationale: format!("Auto-accepted AST update for `{function_id}`."),
        },
    )?;

    // patch the ast directly — accept_proposal only runs its own emitted deltas,
    // not our pre-injected one, so we apply it manually here
    if let Some(function) = acceptance
        .ir
        .functions
        .iter_mut()
        .find(|f| f.id == function_id)
    {
        if let Some(DeltaPayload::UpdateFunctionAst { ast, .. }) = working
            .deltas
            .iter()
            .find(|d| d.id == delta_id)
            .and_then(|d| d.payload.as_ref())
        {
            function.metadata.ast = Some(ast.clone());
        }
    }

    Ok(acceptance)
}
