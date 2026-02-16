use std::collections::{BTreeMap, HashSet};

use crate::{
    ir::proposal::ResolvedProposalNodes,
    ir::{CanonicalIr, Proposal, ProposalStatus, TraitFunction},
};

use super::AcceptProposalError;

pub(super) fn enforce_proposal_ready(proposal: &Proposal) -> Result<(), AcceptProposalError> {
    if proposal.status != ProposalStatus::Submitted {
        return Err(AcceptProposalError::InvalidProposalStatus {
            proposal: proposal.id.clone(),
            status: proposal.status,
        });
    }
    if proposal.nodes.is_empty() || proposal.apis.is_empty() || proposal.edges.is_empty() {
        return Err(AcceptProposalError::IncompleteProposal(proposal.id.clone()));
    }
    Ok(())
}

pub fn ensure_proof_exists(ir: &CanonicalIr, proof_id: &str) -> Result<(), AcceptProposalError> {
    if ir.proofs.iter().any(|proof| proof.id == proof_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownProof(proof_id.to_owned()))
    }
}

pub(super) fn ensure_predicate_exists(
    ir: &CanonicalIr,
    predicate_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.judgment_predicates.iter().any(|p| p.id == predicate_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownPredicate(
            predicate_id.to_owned(),
        ))
    }
}

pub(super) fn ensure_tick_exists(
    ir: &CanonicalIr,
    tick_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.ticks.iter().any(|tick| tick.id == tick_id) {
        Ok(())
    } else {
        Err(AcceptProposalError::UnknownTick(tick_id.to_owned()))
    }
}

pub(super) fn ensure_unique_judgment(
    ir: &CanonicalIr,
    judgment_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.judgments.iter().any(|j| j.id == judgment_id) {
        Err(AcceptProposalError::DuplicateJudgment(
            judgment_id.to_owned(),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn ensure_unique_admission(
    ir: &CanonicalIr,
    admission_id: &str,
) -> Result<(), AcceptProposalError> {
    if ir.admissions.iter().any(|a| a.id == admission_id) {
        Err(AcceptProposalError::DuplicateAdmission(
            admission_id.to_owned(),
        ))
    } else {
        Ok(())
    }
}

pub(super) fn enforce_references(
    ir: &CanonicalIr,
    proposal: &Proposal,
    resolved: &ResolvedProposalNodes,
    trait_functions: &BTreeMap<String, Vec<TraitFunction>>,
) -> Result<(), AcceptProposalError> {
    let existing_modules: HashSet<&str> = ir.modules.iter().map(|m| m.id.as_str()).collect();
    for module in &resolved.modules {
        if existing_modules.contains(module.id.as_str()) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "module",
                id: module.id.clone(),
            });
        }
    }
    let known_modules: HashSet<&str> = existing_modules
        .into_iter()
        .chain(resolved.modules.iter().map(|m| m.id.as_str()))
        .collect();

    let existing_traits: HashSet<&str> = ir.traits.iter().map(|t| t.id.as_str()).collect();
    for trait_spec in &resolved.traits {
        if existing_traits.contains(trait_spec.id.as_str()) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "trait",
                id: trait_spec.id.clone(),
            });
        }
    }
    for trait_spec in &resolved.traits {
        if !known_modules.contains(trait_spec.module.as_str()) {
            return Err(AcceptProposalError::UnknownModule(
                trait_spec.module.clone(),
            ));
        }
    }
    let known_traits: HashSet<&str> = existing_traits
        .into_iter()
        .chain(resolved.traits.iter().map(|t| t.id.as_str()))
        .collect();

    for structure in &resolved.structs {
        if !known_modules.contains(structure.module.as_str()) {
            return Err(AcceptProposalError::UnknownModule(structure.module.clone()));
        }
        if ir.structs.iter().any(|s| s.id == structure.id) {
            return Err(AcceptProposalError::ArtifactExists {
                kind: "struct",
                id: structure.id.clone(),
            });
        }
    }

    for edge in &proposal.edges {
        if !known_modules.contains(edge.from.as_str()) {
            return Err(AcceptProposalError::UnknownModule(edge.from.clone()));
        }
        if !known_modules.contains(edge.to.as_str()) {
            return Err(AcceptProposalError::UnknownModule(edge.to.clone()));
        }
    }

    for trait_id in trait_functions.keys() {
        if !known_traits.contains(trait_id.as_str()) {
            return Err(AcceptProposalError::UnknownTrait(trait_id.clone()));
        }
    }

    let existing_impls: HashSet<(String, String)> = ir
        .impls
        .iter()
        .map(|block| (block.struct_id.clone(), block.trait_id.clone()))
        .collect();
    for structure in &resolved.structs {
        for trait_id in trait_functions.keys() {
            if existing_impls.contains(&(structure.id.clone(), trait_id.clone())) {
                return Err(AcceptProposalError::ArtifactExists {
                    kind: "impl",
                    id: format!("{}->{}", structure.id, trait_id),
                });
            }
        }
    }

    Ok(())
}
