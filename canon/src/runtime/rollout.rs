use std::collections::BTreeMap;
use crate::ir::world_model::StateSnapshot;
use crate::ir::SystemState;
use crate::ir::DeltaId;
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::FunctionExecutor;
use crate::runtime::graph::{build_dependency_map, gather_inputs, topological_sort};
use crate::runtime::reward::compute_reward_from_deltas;
use crate::runtime::value::Value;
#[derive(Debug)]
pub struct RolloutResult {
    pub total_reward: f64,
    pub predicted_deltas: Vec<DeltaId>,
    pub depth_executed: u32,
    pub predicted_state: Option<StateSnapshot>,
}
#[derive(Debug)]
pub enum RolloutError {
    UnknownTick,
    Execution(String),
}
pub struct RolloutEngine<'a> {
    ir: &'a SystemState,
}
impl<'a> RolloutEngine<'a> {
    pub fn new(ir: &'a SystemState) -> Self {
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
                predicted_state: None,
            });
        }
        let mut ir_clone = self.ir.clone();
        let mut total_reward = 0.0;
        let mut predicted_deltas = Vec::new();
        let mut current_inputs = initial_inputs;
        let mut depth_executed = 0;
        let mut predicted_state = None;
        for _ in 0..depth {
            let tick = ir_clone
                .ticks
                .iter()
                .find(|t| t.id == tick_id)
                .ok_or(RolloutError::UnknownTick)?;
            let graph = ir_clone
                .tick_graphs
                .iter()
                .find(|g| g.id == tick.graph)
                .ok_or(RolloutError::UnknownTick)?;
            let dependencies = build_dependency_map(graph);
            let execution_order = topological_sort(graph, &dependencies)
                .map_err(|e| RolloutError::Execution(format!("{:?}", e)))?;
            let mut context = ExecutionContext::new(current_inputs.clone());
            let mut results = std::collections::HashMap::new();
            let mut function_executor = FunctionExecutor::new(&mut ir_clone);
            for function_id in &execution_order {
                let inputs = gather_inputs(
                        function_id,
                        &dependencies,
                        &results,
                        &current_inputs,
                    )
                    .map_err(|e| RolloutError::Execution(format!("{:?}", e)))?;
                let outputs = function_executor
                    .execute_by_id(function_id, inputs, &mut context)
                    .map_err(|e| RolloutError::Execution(format!("{:?}", e)))?;
                results.insert(function_id.clone(), outputs);
            }
            let reward = compute_reward_from_deltas(context.deltas());
            total_reward += reward;
            let actual_delta_ids: Vec<_> = context
                .deltas()
                .iter()
                .map(|delta| delta.delta_id.clone())
                .collect();
            predicted_deltas.extend(actual_delta_ids.clone());
            predicted_state = Some(StateSnapshot {
                tick: tick_id.to_string(),
                delta_ids: actual_delta_ids,
                description: format!("rollout depth {}", depth_executed + 1),
            });
            current_inputs = current_inputs.clone();
            depth_executed += 1;
        }
        Ok(RolloutResult {
            total_reward,
            predicted_deltas,
            depth_executed,
            predicted_state,
        })
    }
}
