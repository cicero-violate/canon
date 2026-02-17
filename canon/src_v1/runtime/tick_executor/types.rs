//! Core types for tick graph execution.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;
use thiserror::Error;

use crate::ir::world_model::StateSnapshot;
use crate::ir::{DeltaId, FunctionId};
use crate::runtime::executor::ExecutorError;
use crate::runtime::value::{DeltaValue, Value};

/// Execution mode for tick graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickExecutionMode {
    Sequential,
    ParallelVerified,
}

#[derive(Debug, Clone)]
pub struct TickExecutionResult {
    pub tick_id: String,
    pub function_results: HashMap<FunctionId, BTreeMap<String, Value>>,
    pub execution_order: Vec<FunctionId>,
    pub emitted_deltas: Vec<DeltaValue>,
    pub reward: f64,
    pub sequential_duration: Duration,
    pub parallel_duration: Option<Duration>,
}

#[derive(Clone)]
pub(super) struct PredictionContext {
    pub predicted_deltas: Vec<DeltaId>,
    pub predicted_snapshot: Option<StateSnapshot>,
}

impl Default for PredictionContext {
    fn default() -> Self {
        Self {
            predicted_deltas: Vec::new(),
            predicted_snapshot: None,
        }
    }
}

#[derive(Clone)]
pub(super) struct PlanContext {
    pub planned_utility: f64,
    pub planning_depth: u32,
    pub predicted_deltas: Vec<DeltaId>,
    pub prediction_context: PredictionContext,
}

impl Default for PlanContext {
    fn default() -> Self {
        Self {
            planned_utility: 0.0,
            planning_depth: 0,
            predicted_deltas: Vec::new(),
            prediction_context: PredictionContext::default(),
        }
    }
}

pub(super) fn default_predicted_snapshot(tick_id: &str, horizon: u32) -> StateSnapshot {
    StateSnapshot {
        tick: tick_id.to_string(),
        delta_ids: Vec::new(),
        description: format!("baseline horizon {horizon}"),
    }
}
