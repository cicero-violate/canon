//! Prediction reconciliation and world-model update.

use crate::ir::world_model::{PredictionRecord, StateSnapshot};

use super::planning::update_epoch_entropy;
use super::types::{PredictionContext, TickExecutionResult};
use crate::ir::CanonicalIr;

pub(super) fn reconcile_prediction(
    ir: &mut CanonicalIr,
    result: &TickExecutionResult,
    prediction: PredictionContext,
) {
    let actual_delta_ids: Vec<_> = result
        .emitted_deltas
        .iter()
        .map(|d| d.delta_id.clone())
        .collect();

    let record = PredictionRecord::new(
        result.tick_id.clone(),
        prediction.predicted_deltas,
        actual_delta_ids.clone(),
    );

    let actual_snapshot = StateSnapshot {
        tick: result.tick_id.clone(),
        delta_ids: actual_delta_ids,
        description: prediction
            .predicted_snapshot
            .as_ref()
            .map(|snap| snap.description.clone())
            .unwrap_or_else(|| "observed".to_owned()),
    };

    ir.world_model.record_prediction(record.clone(), actual_snapshot);

    let entropy = (record.error.max(1e-6_f64)).ln();
    update_epoch_entropy(ir, &result.tick_id, entropy);
}
