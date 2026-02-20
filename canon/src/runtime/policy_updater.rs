use crate::ir::{CanonicalIr, PolicyParameters};

/// Mutable handle that can update policy parameters over epochs.
pub struct PolicyUpdater {
    ir: *mut CanonicalIr,
}

impl PolicyUpdater {
    pub fn new(ir: *mut CanonicalIr) -> Self {
        Self { ir }
    }
}

/// Rule-based proxy for a gradient step.
pub fn update_policy(current: &PolicyParameters, reward: f64) -> Result<PolicyParameters, PolicyUpdateError> {
    if current.learning_rate <= 0.0 {
        return Err(PolicyUpdateError::InvalidLearningRate);
    }
    let adjustment = current.learning_rate * (reward - current.reward_baseline);
    let nudge = |value: f64| value + adjustment;
    Ok(PolicyParameters {
        id: current.id.clone(),
        version: current.version.saturating_add(1),
        epoch: current.epoch.clone(),
        learning_rate: nudge(current.learning_rate),
        discount_factor: nudge(current.discount_factor),
        entropy_weight: nudge(current.entropy_weight),
        reward_baseline: nudge(current.reward_baseline),
        proof_id: current.proof_id.clone(),
    })
}

#[derive(Debug)]
pub enum PolicyUpdateError {
    EpochNotFound,
    NoPriorPolicy,
    InvalidLearningRate,
}
