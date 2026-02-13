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
}

