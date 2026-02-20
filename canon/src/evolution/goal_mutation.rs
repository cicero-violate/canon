use crate::{
    decision::accept::proposal_checks::ensure_proof_exists,
    ir::{
        goals::{compute_goal_drift, GoalMutation, GoalMutationStatus},
        SystemState,
    },
};
pub fn mutate_goal(
    ir: &SystemState,
    mutation: &GoalMutation,
) -> Result<SystemState, GoalMutationError> {
    for proof_id in &mutation.invariant_proof_ids {
        ensure_proof_exists(ir, proof_id).map_err(|_| GoalMutationError::MissingProof)?;
    }
    let mut metric = compute_goal_drift(
        &mutation.original_goal,
        &mutation.proposed_goal,
        0.05,
    );
    metric.mutation_id = mutation.id.clone();
    if !metric.within_bound {
        return Err(GoalMutationError::DriftExceeded);
    }
    if mutation.status != GoalMutationStatus::Accepted || mutation.judgment_id.is_none()
    {
        return Err(GoalMutationError::NotAccepted);
    }
    let mut next = ir.clone();
    next.goal_mutations.push(mutation.clone());
    Ok(next)
}
#[derive(Debug)]
pub enum GoalMutationError {
    MissingProof,
    InvariantViolated,
    DriftExceeded,
    NotAccepted,
    UnknownMutation,
}
