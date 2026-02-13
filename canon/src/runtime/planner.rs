use std::collections::BTreeMap;

use crate::ir::CanonicalIr;
use crate::runtime::rollout::RolloutEngine;
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
    ) -> f64 {
        let engine = RolloutEngine::new(self.ir);
        match engine.rollout(tick_id, depth, inputs) {
            Ok(result) => result.total_reward,
            Err(_) => f64::NEG_INFINITY,
        }
    }

    /// Evaluate multiple depths and return best (utility, depth)
    pub fn search_best_depth(
        &self,
        tick_id: &str,
        max_depth: u32,
        inputs: BTreeMap<String, Value>,
    ) -> (f64, u32) {
        let mut best = f64::NEG_INFINITY;
        let mut best_depth = 0;

        for depth in 1..=max_depth {
            let score = self.score_tick(tick_id, depth, inputs.clone());
            if score > best {
                best = score;
                best_depth = depth;
            }
        }

        (best, best_depth)
    }
}
