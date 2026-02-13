use crate::ir::{CanonicalIr, ExecutionRecord};

/// Deterministic scalar utility computation.
/// Minimal foundation implementation:
/// U = (#outcome_deltas) - (#errors)
pub fn compute_reward(_ir: &CanonicalIr, record: &ExecutionRecord) -> f64 {
    let positive = record.outcome_deltas.len() as f64;
    let penalties = record.errors.len() as f64;
    positive - penalties
}

