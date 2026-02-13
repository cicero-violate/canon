use crate::ir::{CanonicalIr, TickId};
use crate::runtime::tick_executor::{TickExecutionMode, TickExecutor};

/// Depth-limited speculative rollout.
/// Layer 2 — deterministic scaffold.
pub struct RolloutEngine<'a> {
    ir: &'a CanonicalIr,
    executor: TickExecutor<'a>,
}

impl<'a> RolloutEngine<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self {
            ir,
            executor: TickExecutor::new(ir),
        }
    }

    /// Execute a speculative rollout starting from `tick_id`
    /// up to `depth` sequential future ticks.
    ///
    /// NOTE:
    /// This is a pure simulation layer.
    /// It does NOT mutate CanonicalIr.
    pub fn rollout(
        &self,
        tick_id: &str,
        depth: u32,
    ) -> Result<Vec<RolloutStep>, RolloutError> {
        let mut steps = Vec::new();

        let mut current_tick: TickId = tick_id.to_string();

        for _ in 0..depth {
            let result = self
                .executor
                .execute_tick_with_mode(&current_tick, TickExecutionMode::Sequential)
                .map_err(RolloutError::Execution)?;

            steps.push(RolloutStep {
                tick: current_tick.clone(),
                reward: result.reward,
                emitted_deltas: result.emitted_deltas.len() as u64,
            });

            // Minimal scaffold:
            // stop after first execution unless caller wires future-tick graph.
            break;
        }

        Ok(steps)
    }
}

#[derive(Debug, Clone)]
pub struct RolloutStep {
    pub tick: TickId,
    pub reward: f64,
    pub emitted_deltas: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum RolloutError {
    #[error(transparent)]
    Execution(#[from] crate::runtime::tick_executor::TickExecutorError),
}

