use super::error::Violation;
use super::helpers::Indexes;
use super::rules::CanonRule;
use crate::ir::*;
use crate::proposal::resolve_proposal_nodes;
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
            violations.push(Violation::new(
                CanonRule::ProposalDeclarative,
                format!(
                    "proposal `{}` must enumerate nodes, APIs, and edges",
                    proposal.id
                ),
            ));
        }
        if proposal.goal.description.trim().is_empty() {
            violations.push(Violation::new(
                CanonRule::ProposalDeclarative,
                format!(
                    "proposal `{}` must include a textual goal description",
                    proposal.id
                ),
            ));
        }
        let resolved = match resolve_proposal_nodes(proposal) {
            Ok(r) => r,
            Err(e) => {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!("proposal `{}` is invalid: {e}", proposal.id),
                ));
                continue;
            }
        };
        let proposed_mods: HashSet<&str> = resolved.modules.iter().map(|m| m.id.as_str()).collect();
        let proposed_traits: HashSet<&str> =
            resolved.traits.iter().map(|t| t.id.as_str()).collect();
        for s in &resolved.structs {
            if idx.modules.get(s.module.as_str()).is_none()
                && !proposed_mods.contains(s.module.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown module `{}` for struct `{}`",
                        proposal.id, s.module, s.id
                    ),
                ));
            }
        }
        for t in &resolved.traits {
            if idx.modules.get(t.module.as_str()).is_none()
                && !proposed_mods.contains(t.module.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown module `{}` for trait `{}`",
                        proposal.id, t.module, t.id
                    ),
                ));
            }
        }
        for edge in &proposal.edges {
            if idx.modules.get(edge.from.as_str()).is_none()
                && !proposed_mods.contains(edge.from.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` edge `{}` -> `{}` references unknown from-module",
                        proposal.id, edge.from, edge.to
                    ),
                ));
            }
            if idx.modules.get(edge.to.as_str()).is_none()
                && !proposed_mods.contains(edge.to.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` edge `{}` -> `{}` references unknown to-module",
                        proposal.id, edge.from, edge.to
                    ),
                ));
            }
        }
        for api in &proposal.apis {
            if idx.traits.get(api.trait_id.as_str()).is_none()
                && !proposed_traits.contains(api.trait_id.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ProposalDeclarative,
                    format!(
                        "proposal `{}` references unknown trait `{}`",
                        proposal.id, api.trait_id
                    ),
                ));
            }
        }
    }
}

fn check_judgments(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for j in &ir.judgments {
        if idx.proposals.get(j.proposal.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::JudgmentDecisions,
                format!(
                    "judgment `{}` references missing proposal `{}`",
                    j.id, j.proposal
                ),
            ));
        }
        if idx.predicates.get(j.predicate.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::JudgmentDecisions,
                format!(
                    "judgment `{}` references missing predicate `{}`",
                    j.id, j.predicate
                ),
            ));
        }
    }
}

fn check_learning(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for item in &ir.learning {
        if idx.proposals.get(item.proposal.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!(
                    "learning `{}` references missing proposal `{}`",
                    item.id, item.proposal
                ),
            ));
        }
        if item.new_rules.is_empty() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!("learning `{}` must enumerate proposed rules", item.id),
            ));
        }
        if item.proof_object_hash.is_none() {
            violations.push(Violation::new(
                CanonRule::LearningDeclarations,
                format!("learning `{}` must include proof_object_hash", item.id),
            ));
        }
    }
}

fn check_goal_mutations(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for mutation in &ir.goal_mutations {
        if mutation.status == GoalMutationStatus::Accepted && mutation.judgment_id.is_none() {
            violations.push(Violation::new(
                CanonRule::GoalMutationRequiresJudgment,
                format!("goal mutation `{}` lacks a judgment reference", mutation.id),
            ));
        }
        for proof_id in &mutation.invariant_proof_ids {
            if idx.proofs.get(proof_id.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::GoalMutationInvariantMissing,
                    format!(
                        "goal mutation `{}` references unknown proof `{}`",
                        mutation.id, proof_id
                    ),
                ));
            }
        }
    }
}
