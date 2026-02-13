use std::collections::BTreeMap;

use crate::ir::CanonicalIr;
use crate::runtime::rollout::{RolloutEngine, RolloutResult};
use crate::runtime::value::Value;

pub struct Planner<'a> {
    ir: &'a CanonicalIr,
}

impl<'a> Planner<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self { ir }
    }

    pub fn score_tick(
        &self,
        tick_id: &str,
        depth: u32,
        inputs: BTreeMap<String, Value>,
    ) -> Option<RolloutResult> {
        let engine = RolloutEngine::new(self.ir);
        engine.rollout(tick_id, depth, inputs).ok()
    }

    fn world_model_bonus(&self, tick_id: &str, reward: f64) -> f64 {
        self.ir
            .world_model
            .prediction_head
            .iter()
            .rev()
            .find(|head| head.tick == tick_id)
            .map(|head| (head.estimated_reward - reward) * 0.1)
            .unwrap_or(0.0)
    }

    /// Evaluate multiple depths and return best (rollout, utility)
    pub fn search_best_depth(
        &self,
        tick_id: &str,
        max_depth: u32,
        inputs: BTreeMap<String, Value>,
    ) -> Option<(RolloutResult, f64)> {
        let mut best = None;

        for depth in 1..=max_depth {
            if let Some(result) = self.score_tick(tick_id, depth, inputs.clone()) {
                let utility =
                    result.total_reward + self.world_model_bonus(tick_id, result.total_reward);
                if best
                    .as_ref()
                    .map_or(true, |(_, best_util)| utility > *best_util)
                {
                    best = Some((result, utility));
                }
            }
        }

        best
    }
}
