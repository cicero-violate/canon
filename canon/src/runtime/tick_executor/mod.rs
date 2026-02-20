//! Tick graph executor.
//!
//! Executes an entire tick graph, respecting data dependencies (Canon Lines 46-50).

pub(crate) mod graph;
mod parallel;
mod planning;
mod reconcile;
mod types;

pub use types::{TickExecutionMode, TickExecutionResult};

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use crate::ir::{CanonicalIr, TickGraph};
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::FunctionExecutor;
use crate::runtime::value::Value;

use crate::runtime::error::RuntimeError;
use crate::runtime::reward::compute_reward_from_deltas;
use graph::{build_dependency_map, gather_inputs, topological_sort};
use parallel::{execute_parallel, verify_parallel_deltas, verify_parallel_outputs};
use planning::{finalize_execution, plan_tick};
use reconcile::reconcile_prediction;
use types::{PlanContext, PredictionContext};

/// Executes a tick graph in topological order.
pub struct TickExecutor<'a> {
    ir: &'a mut CanonicalIr,
    skip_planning: bool,
}

impl<'a> TickExecutor<'a> {
    pub fn new(ir: &'a mut CanonicalIr) -> Self {
        Self::new_with_skip_planning(ir, false)
    }

    pub fn new_with_skip_planning(ir: &'a mut CanonicalIr, skip_planning: bool) -> Self {
        Self { ir, skip_planning }
    }

    /// Execute a tick by its ID.
    pub fn execute_tick(&mut self, tick_id: &str) -> Result<TickExecutionResult, RuntimeError> {
        self.execute_tick_with_mode_and_inputs(tick_id, TickExecutionMode::Sequential, BTreeMap::new())
    }

    /// Execute a tick with explicit initial inputs provided to the root nodes.
    pub fn execute_tick_with_inputs(&mut self, tick_id: &str, initial_inputs: BTreeMap<String, Value>) -> Result<TickExecutionResult, RuntimeError> {
        self.execute_tick_with_mode_and_inputs(tick_id, TickExecutionMode::Sequential, initial_inputs)
    }

    /// Execute a tick with an explicit mode (sequential or parallel-verified).
    pub fn execute_tick_with_mode(&mut self, tick_id: &str, mode: TickExecutionMode) -> Result<TickExecutionResult, RuntimeError> {
        self.execute_tick_with_mode_and_inputs(tick_id, mode, BTreeMap::new())
    }

    /// Execute a tick with an explicit mode and initial inputs.
    pub fn execute_tick_with_mode_and_inputs(&mut self, tick_id: &str, mode: TickExecutionMode, initial_inputs: BTreeMap<String, Value>) -> Result<TickExecutionResult, RuntimeError> {
        let plan_ctx = plan_tick(self.ir, tick_id, self.skip_planning);
        self.execute_internal(tick_id, mode, initial_inputs, plan_ctx)
    }

    fn execute_internal(&mut self, tick_id: &str, mode: TickExecutionMode, initial_inputs: BTreeMap<String, Value>, plan_ctx: PlanContext) -> Result<TickExecutionResult, RuntimeError> {
        let tick = self.ir.ticks.iter().find(|t| t.id == tick_id).ok_or_else(|| RuntimeError::UnknownTick(tick_id.to_string()))?;

        let graph = self.ir.tick_graphs.iter().find(|g| g.id == tick.graph).ok_or_else(|| RuntimeError::UnknownGraph(tick.graph.clone()))?;

        let graph_cloned = graph.clone();
        let tick_id_owned = tick.id.clone();
        let PlanContext { planned_utility, planning_depth, predicted_deltas, prediction_context } = plan_ctx;

        let result = self.execute_graph_with_prediction(&graph_cloned, &tick_id_owned, mode, &initial_inputs, prediction_context)?;

        finalize_execution(self.ir, tick_id, &result, planned_utility, planning_depth, predicted_deltas);

        Ok(result)
    }

    /// Execute a tick graph in topological order.
    /// Graph must be acyclic (Canon Line 48).
    fn execute_graph(&mut self, graph: &TickGraph, tick_id: &str, mode: TickExecutionMode, initial_inputs: &BTreeMap<String, Value>) -> Result<TickExecutionResult, RuntimeError> {
        let dependencies = build_dependency_map(graph);
        let execution_order = topological_sort(graph, &dependencies)?;
        let mut context = ExecutionContext::new(initial_inputs.clone());
        let mut results = HashMap::new();
        let sequential_start = Instant::now();
        let mut function_executor = FunctionExecutor::new(self.ir);

        for function_id in &execution_order {
            let inputs = gather_inputs(function_id, &dependencies, &results, initial_inputs)?;
            let outputs = function_executor.execute_by_id(function_id, inputs, &mut context).map_err(RuntimeError::Executor)?;
            results.insert(function_id.clone(), outputs);
        }
        let sequential_duration = sequential_start.elapsed();
        let mut parallel_duration = None;

        if matches!(mode, TickExecutionMode::ParallelVerified) {
            let parallel_start = Instant::now();
            let (parallel_results, parallel_deltas) = execute_parallel(self.ir, &execution_order, &dependencies, initial_inputs)?;
            parallel_duration = Some(parallel_start.elapsed());
            verify_parallel_outputs(&results, &parallel_results)?;
            verify_parallel_deltas(context.deltas(), &parallel_deltas)?;
        }

        Ok(TickExecutionResult {
            tick_id: tick_id.to_string(),
            function_results: results,
            execution_order,
            emitted_deltas: context.deltas().to_vec(),
            reward: compute_reward_from_deltas(context.deltas()),
            sequential_duration,
            parallel_duration,
        })
    }

    fn execute_graph_with_prediction(
        &mut self, graph: &TickGraph, tick_id: &str, mode: TickExecutionMode, initial_inputs: &BTreeMap<String, Value>, prediction: PredictionContext,
    ) -> Result<TickExecutionResult, RuntimeError> {
        let result = self.execute_graph(graph, tick_id, mode, initial_inputs)?;
        reconcile_prediction(self.ir, &result, prediction);
        Ok(result)
    }
}
