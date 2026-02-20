use super::error::{Violation, ViolationDetail};
use super::helpers::Indexes;
use super::rules::CanonRule;
use crate::ir::proposal::resolve_proposal_nodes;
use crate::ir::*;
use std::collections::HashSet;

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    check_proposals(ir, idx, violations);
    check_judgments(ir, idx, violations);
    check_learning(ir, idx, violations);
    check_goal_mutations(ir, idx, violations);
}

fn check_proposals(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for proposal in &ir.proposals {
        if proposal.nodes.is_empty() || proposal.apis.is_empty() || proposal.edges.is_empty() {
            violations.push(Violation::structured(CanonRule::ProposalDeclarative, proposal.id.clone(), ViolationDetail::ProposalIncomplete { proposal: proposal.id.clone() }));
        }
        if proposal.goal.description.trim().is_empty() {
            violations.push(Violation::structured(CanonRule::ProposalDeclarative, proposal.id.clone(), ViolationDetail::ProposalMissingGoal { proposal: proposal.id.clone() }));
        }
        let resolved = match resolve_proposal_nodes(proposal) {
            Ok(r) => r,
            Err(e) => {
                violations.push(Violation::structured(CanonRule::ProposalDeclarative, proposal.id.clone(), ViolationDetail::ProposalInvalid { proposal: proposal.id.clone() }));
                continue;
            }
        };
        let proposed_mods: HashSet<&str> = resolved.modules.iter().map(|m| m.id.as_str()).collect();
        let proposed_traits: HashSet<&str> = resolved.traits.iter().map(|t| t.id.as_str()).collect();
        for s in &resolved.structs {
            if idx.modules.get(s.module.as_str()).is_none() && !proposed_mods.contains(s.module.as_str()) {
                violations.push(Violation::structured(
                    CanonRule::ProposalDeclarative,
                    proposal.id.clone(),
                    ViolationDetail::ProposalUnknownModule { proposal: proposal.id.clone(), module: s.module.clone() },
                ));
            }
        }
        for t in &resolved.traits {
            if idx.modules.get(t.module.as_str()).is_none() && !proposed_mods.contains(t.module.as_str()) {
                violations.push(Violation::structured(
                    CanonRule::ProposalDeclarative,
                    proposal.id.clone(),
                    ViolationDetail::ProposalUnknownModule { proposal: proposal.id.clone(), module: t.module.clone() },
                ));
            }
        }
        for edge in &proposal.edges {
            if idx.modules.get(edge.from.as_str()).is_none() && !proposed_mods.contains(edge.from.as_str()) {
                violations.push(Violation::structured(
                    CanonRule::ProposalDeclarative,
                    proposal.id.clone(),
                    ViolationDetail::ProposalUnknownEdgeModule { proposal: proposal.id.clone(), module: edge.from.clone() },
                ));
            }
            if idx.modules.get(edge.to.as_str()).is_none() && !proposed_mods.contains(edge.to.as_str()) {
                violations.push(Violation::structured(
                    CanonRule::ProposalDeclarative,
                    proposal.id.clone(),
                    ViolationDetail::ProposalUnknownEdgeModule { proposal: proposal.id.clone(), module: edge.to.clone() },
                ));
            }
        }
        for api in &proposal.apis {
            if idx.traits.get(api.trait_id.as_str()).is_none() && !proposed_traits.contains(api.trait_id.as_str()) {
                violations.push(Violation::structured(
                    CanonRule::ProposalDeclarative,
                    proposal.id.clone(),
                    ViolationDetail::ProposalUnknownTrait { proposal: proposal.id.clone(), trait_id: api.trait_id.clone() },
                ));
            }
        }
    }
}

fn check_judgments(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for j in &ir.judgments {
        if idx.proposals.get(j.proposal.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::JudgmentDecisions, j.id.clone(), ViolationDetail::JudgmentMissingProposal { judgment: j.id.clone(), proposal: j.proposal.clone() }));
        }
        if idx.predicates.get(j.predicate.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::JudgmentDecisions, j.id.clone(), ViolationDetail::JudgmentMissingPredicate { judgment: j.id.clone(), predicate: j.predicate.clone() }));
        }
    }
}

fn check_learning(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for item in &ir.learning {
        if idx.proposals.get(item.proposal.as_str()).is_none() {
            violations.push(Violation::structured(
                CanonRule::LearningDeclarations,
                item.id.clone(),
                ViolationDetail::LearningMissingProposal { learning: item.id.clone(), proposal: item.proposal.clone() },
            ));
        }
        if item.new_rules.is_empty() {
            violations.push(Violation::structured(CanonRule::LearningDeclarations, item.id.clone(), ViolationDetail::LearningMissingRules { learning: item.id.clone() }));
        }
        if item.proof_object_hash.is_none() {
            violations.push(Violation::structured(CanonRule::LearningDeclarations, item.id.clone(), ViolationDetail::LearningMissingProofObject { learning: item.id.clone() }));
        }
    }
}

fn check_goal_mutations(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for mutation in &ir.goal_mutations {
        if mutation.status == GoalMutationStatus::Accepted && mutation.judgment_id.is_none() {
            violations.push(Violation::structured(CanonRule::GoalMutationRequiresJudgment, mutation.id.clone(), ViolationDetail::GoalMutationMissingJudgment { mutation: mutation.id.clone() }));
        }
        for proof_id in &mutation.invariant_proof_ids {
            if idx.proofs.get(proof_id.as_str()).is_none() {
                violations.push(Violation::structured(
                    CanonRule::GoalMutationInvariantMissing,
                    mutation.id.clone(),
                    ViolationDetail::GoalMutationMissingProof { mutation: mutation.id.clone(), proof: proof_id.clone() },
                ));
            }
        }
    }
}
