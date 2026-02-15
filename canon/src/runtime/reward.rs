use crate::ir::{CanonicalIr, ExecutionRecord};

/// Deterministic scalar utility computation.
/// Minimal foundation implementation:
/// U = (#outcome_deltas) - (#errors)
pub fn compute_reward(_ir: &CanonicalIr, record: &ExecutionRecord) -> f64 {
    let positive = record.outcome_deltas.len() as f64;
    let penalties = record.errors.len() as f64;
    positive - penalties
}

/// Reward weights for pipeline reward computation.
const W_CAPS: f64 = 0.5;   // weight per new capability (function/trait/module)
const W_DELTA: f64 = 0.1;  // penalty per delta above threshold
const DELTA_THRESHOLD: usize = 10; // deltas above this count are penalised
const W_ENTROPY: f64 = 0.2; // bonus for entropy reduction in capability graph

/// Computes a real reward signal for a completed pipeline run.
///
/// r = Δcaps * W_CAPS - max(0, Δdeltas - threshold) * W_DELTA + ΔH⁻¹ * W_ENTROPY
///
/// Where:
///   Δcaps    = new functions + new traits + new modules in candidate vs before
///   Δdeltas  = total applied deltas in candidate
///   ΔH⁻¹    = entropy reduction (before_H - after_H), positive = more ordered
/// Computes a real reward signal for a completed pipeline run.
///
/// r = Δcaps * W_CAPS - max(0, Δdeltas - threshold) * W_DELTA + ΔH⁻¹ * W_ENTROPY
///
/// Where:
///   Δcaps    = new functions + new traits + new modules in candidate vs before
///   Δdeltas  = total applied deltas in candidate
///   entropy_before / entropy_after = pre-computed graph entropy scalars (pass 0.0 if unavailable)
pub fn compute_pipeline_reward(
    before: &CanonicalIr,
    after: &CanonicalIr,
    entropy_before: f64,
    entropy_after: f64,
) -> f64 {
    // --- capability delta ---
    let new_functions = after.functions.len().saturating_sub(before.functions.len()) as f64;
    let new_traits = after.traits.len().saturating_sub(before.traits.len()) as f64;
    let new_modules = after.modules.len().saturating_sub(before.modules.len()) as f64;
    let delta_caps = new_functions + new_traits + new_modules;

    // --- delta count penalty ---
    let delta_count = after.applied_deltas.len();
    let delta_penalty = (delta_count.saturating_sub(DELTA_THRESHOLD)) as f64;

    // --- entropy reduction bonus ---
    // Positive when entropy decreased (graph became more ordered).
    let entropy_reduction = entropy_before - entropy_after;

    delta_caps * W_CAPS - delta_penalty * W_DELTA + entropy_reduction * W_ENTROPY
}
