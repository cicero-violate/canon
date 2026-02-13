use std::collections::BTreeMap;

use crate::ir::CanonicalIr;
use crate::runtime::tick_executor::{TickExecutionMode, TickExecutor};
use crate::runtime::value::Value;

#[derive(Debug)]
pub struct RolloutResult {
    pub total_reward: f64,
    pub predicted_deltas: Vec<String>, // DeltaId
    pub depth_executed: u32,
}

#[derive(Debug)]
pub enum RolloutError {
    UnknownTick,
    Execution(String),
}

pub struct RolloutEngine<'a> {
    ir: &'a CanonicalIr,
}

impl<'a> RolloutEngine<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self { ir }
    }

    pub fn rollout(
        &self,
        tick_id: &str,
        depth: u32,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<RolloutResult, RolloutError> {
        if depth == 0 {
            return Ok(RolloutResult {
                total_reward: 0.0,
                predicted_deltas: vec![],
                depth_executed: 0,
            });
        }

        let executor = TickExecutor::new(self.ir);

        let mut total_reward = 0.0;
        let mut predicted_deltas = Vec::new();
        let mut current_inputs = initial_inputs;
        let mut depth_executed = 0;

        for _ in 0..depth {
            let result = executor
                .execute_tick_with_mode_and_inputs(
                    tick_id,
                    TickExecutionMode::Sequential,
                    current_inputs.clone(),
                )
                .map_err(|e| RolloutError::Execution(format!("{:?}", e)))?;

            total_reward += result.reward;

            for delta in &result.emitted_deltas {
                predicted_deltas.push(delta.delta_id.clone());
            }

            // Baseline behavior: reuse same inputs for speculative depth.
            current_inputs = current_inputs.clone();
            depth_executed += 1;
        }

        Ok(RolloutResult {
            total_reward,
            predicted_deltas,
            depth_executed,
        })
    }
}
