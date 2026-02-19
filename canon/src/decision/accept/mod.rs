use std::collections::HashSet;
use thiserror::Error;
use crate::{
    evolution::{EvolutionError, apply_admitted_deltas},
    ir::proposal::{ProposalResolutionError, resolve_proposal_nodes},
    ir::{
        CanonicalIr, Delta, DeltaAdmission, DeltaId, Judgment, JudgmentDecision,
        ProposalStatus, Word, WordError,
    },
    layout::{LayoutAssignment, LayoutFile, LayoutGraph, LayoutModule, LayoutNode},
    proof::smt_bridge::{SmtError, attach_function_proofs},
};
use self::{
    delta_emitter::{build_trait_function_map, emit_deltas},
    proposal_checks::{
        enforce_proposal_ready, enforce_references, ensure_predicate_exists,
        ensure_proof_exists, ensure_tick_exists, ensure_unique_admission,
        ensure_unique_judgment,
    },
};
mod delta_emitter;
pub mod proposal_checks;
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
    pub layout: LayoutGraph,
    pub delta_ids: Vec<DeltaId>,
    pub judgment_id: String,
    pub admission_id: String,
}
#[derive(Debug, Error)]
pub enum AcceptProposalError {
    #[error("proposal `{0}` does not exist")]
    UnknownProposal(String),
    #[error("proposal `{proposal}` must be submitted; found status `{status:?}`")]
    InvalidProposalStatus { proposal: String, status: ProposalStatus },
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
    layout: &LayoutGraph,
    input: ProposalAcceptanceInput,
) -> Result<ProposalAcceptance, AcceptProposalError> {
    let mut working = ir.clone();
    let mut working_layout = layout.clone();
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
    let mut known_delta_ids: HashSet<String> = working
        .deltas
        .iter()
        .map(|d| d.id.clone())
        .collect();
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
    apply_layout_deltas(&mut working_layout, &working, &working.deltas);
    working.proposals[proposal_index].status = ProposalStatus::Accepted;
    working
        .judgments
        .push(Judgment {
            id: input.judgment_id.clone(),
            proposal: input.proposal_id.clone(),
            predicate: input.predicate_id.clone(),
            decision: JudgmentDecision::Accept,
            rationale: input.rationale.clone(),
        });
    working
        .admissions
        .push(DeltaAdmission {
            id: input.admission_id.clone(),
            judgment: input.judgment_id.clone(),
            tick: input.tick_id.clone(),
            delta_ids: delta_ids.clone(),
        });
    let mut evolved = apply_admitted_deltas(&working, &[input.admission_id.clone()])?;
    sync_layout_modules(&mut working_layout, &evolved);
    attach_function_proofs(&mut evolved)?;
    Ok(ProposalAcceptance {
        ir: evolved,
        layout: working_layout,
        delta_ids,
        judgment_id: input.judgment_id,
        admission_id: input.admission_id,
    })
}
fn sync_layout_modules(layout: &mut LayoutGraph, ir: &CanonicalIr) {
    for module in &ir.modules {
        if layout.modules.iter().any(|m| m.id == module.id) {
            continue;
        }
        layout
            .modules
            .push(LayoutModule {
                id: module.id.clone(),
                name: module.name.clone(),
                files: vec![
                    LayoutFile { id : default_file_id(module.id.as_str()), path :
                    "mod.rs".to_owned(), use_block : Vec::new(), }
                ],
                imports: Vec::new(),
            });
    }
}
fn apply_layout_deltas(layout: &mut LayoutGraph, ir: &CanonicalIr, deltas: &[Delta]) {
    for delta in deltas {
        let Some(payload) = &delta.payload else {
            continue;
        };
        match payload {
            crate::ir::DeltaPayload::AddModule { module_id, name, .. } => {
                ensure_layout_module(layout, module_id, name);
            }
            crate::ir::DeltaPayload::AddStruct { module, struct_id, .. } => {
                ensure_assignment(
                    layout,
                    module,
                    LayoutNode::Struct(struct_id.clone()),
                    ir,
                );
            }
            crate::ir::DeltaPayload::AddTrait { module, trait_id, .. } => {
                ensure_assignment(
                    layout,
                    module,
                    LayoutNode::Trait(trait_id.clone()),
                    ir,
                );
            }
            crate::ir::DeltaPayload::AddFunction { function_id, impl_id, .. } => {
                if let Some(module_id) = ir
                    .impls
                    .iter()
                    .find(|block| block.id == *impl_id)
                    .map(|block| block.module.clone())
                {
                    ensure_assignment(
                        layout,
                        module_id.as_str(),
                        LayoutNode::Function(function_id.clone()),
                        ir,
                    );
                }
            }
            crate::ir::DeltaPayload::AddEnum { module, enum_id, name, .. } => {
                ensure_layout_module(layout, module, name);
                ensure_assignment(layout, module, LayoutNode::Enum(enum_id.clone()), ir);
            }
            _ => {}
        }
    }
}
fn ensure_layout_module(layout: &mut LayoutGraph, module_id: &str, module_name: &Word) {
    if layout.modules.iter().any(|m| m.id == module_id) {
        return;
    }
    layout
        .modules
        .push(LayoutModule {
            id: module_id.to_owned(),
            name: module_name.clone(),
            files: vec![
                LayoutFile { id : default_file_id(module_id), path : "mod.rs".to_owned(),
                use_block : Vec::new(), }
            ],
            imports: Vec::new(),
        });
}
fn ensure_assignment(
    layout: &mut LayoutGraph,
    module_id: &str,
    node: LayoutNode,
    ir: &CanonicalIr,
) {
    if layout.routing.iter().any(|assignment| assignment.node == node) {
        return;
    }
    let module_name = ir
        .modules
        .iter()
        .find(|m| m.id == module_id)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| {
            Word::new(module_id).unwrap_or_else(|_| Word::new("Module").unwrap())
        });
    ensure_layout_module(layout, module_id, &module_name);
    let file_id = layout
        .modules
        .iter()
        .find(|m| m.id == module_id)
        .and_then(|module| module.files.first())
        .map(|file| file.id.clone())
        .unwrap_or_else(|| default_file_id(module_id));
    layout
        .routing
        .push(LayoutAssignment {
            node,
            file_id,
            rationale: "LAY-003: default routing".to_owned(),
        });
}
fn default_file_id(module_id: &str) -> String {
    format!("file.{}.mod", slugify(module_id))
}
fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }
        })
        .collect()
}
