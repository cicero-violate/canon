use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{DeltaId, TickId, Word};

/// Predictive world model attached to CanonicalIr.
/// Layer 2 — minimal deterministic scaffold.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct WorldModel {
    /// Monotonic version for schema evolution.
    pub version: String,

    /// Last observed tick.
    pub last_tick: Option<TickId>,

    /// Hash or identifier of the state snapshot used for prediction.
    pub state_root: Option<Word>,

    /// Rolling prediction records.
    pub predictions: Vec<PredictionRecord>,
}

impl WorldModel {
    pub fn new() -> Self {
        Self {
            version: "0.1.0-world-model".to_string(),
            last_tick: None,
            state_root: None,
            predictions: Vec::new(),
        }
    }

    pub fn record_prediction(&mut self, record: PredictionRecord) {
        self.last_tick = Some(record.tick.clone());
        self.predictions.push(record);
    }
}

/// Stores predicted vs actual outcome for a tick.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PredictionRecord {
    /// Tick being predicted.
    pub tick: TickId,

    /// Predicted deltas (IDs only — no duplication of payload).
    pub predicted_deltas: Vec<DeltaId>,

    /// Actual realized deltas.
    pub actual_deltas: Vec<DeltaId>,

    /// Absolute prediction error (|predicted - actual|).
    pub error: f64,
}

impl PredictionRecord {
    pub fn new(
        tick: TickId,
        predicted_deltas: Vec<DeltaId>,
        actual_deltas: Vec<DeltaId>,
    ) -> Self {
        let error = (predicted_deltas.len() as i64 - actual_deltas.len() as i64)
            .unsigned_abs() as f64;

        Self {
            tick,
            predicted_deltas,
            actual_deltas,
            error,
        }
    }
}
