use crate::ir::{CanonicalIr, ExecutionRecord};
use crate::agent::CapabilityGraph;

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
pub fn compute_pipeline_reward(
    before: &CanonicalIr,
    after: &CanonicalIr,
    graph_before: Option<&CapabilityGraph>,
    graph_after: Option<&CapabilityGraph>,
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
    let entropy_reduction = match (graph_before, graph_after) {
        (Some(gb), Some(ga)) => {
            let h_before: f64 = gb.entropy();
            let h_after: f64 = ga.entropy();
            // Positive when entropy decreased (graph became more ordered)
            h_before - h_after
        }
        _ => 0.0,
    };

    delta_caps * W_CAPS - delta_penalty * W_DELTA + entropy_reduction * W_ENTROPY
}
