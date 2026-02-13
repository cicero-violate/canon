//! Tick graph executor.
//!
//! Executes an entire tick graph, respecting data dependencies (Canon Lines 46-50).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ir::{CanonicalIr, FunctionId, TickGraph};
use crate::ir::RewardRecord;
use crate::ir::world_model::PredictionRecord;

// Layer 2: PredictionRecord used for post-execution reconciliation.
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::{ExecutorError, FunctionExecutor};
use crate::runtime::parallel::{
    ParallelJob, ParallelJobResult, execute_jobs, partition_independent_batches,
};
use crate::runtime::value::{DeltaValue, Value};
use crate::runtime::planner::Planner;

fn compute_reward_from_deltas(emitted: &[DeltaValue]) -> f64 {
    // Layer 1 (Foundation): deterministic scalar utility.
    // Minimal rule: U = (# emitted deltas)
    emitted.len() as f64
}

// W4: World-model update is triggered after execution result construction.

/// Executes a tick graph in topological order.
pub struct TickExecutor<'a> {
    ir: &'a mut CanonicalIr,
}

// Layer 2 integration point: executor is world-model aware.

/// Execution mode for tick graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickExecutionMode {
    Sequential,
    ParallelVerified,
}

impl<'a> TickExecutor<'a> {
    pub fn new(ir: &'a mut CanonicalIr) -> Self {
        Self { ir }
    }

    /// Execute a tick by its ID.
    pub fn execute_tick(&mut self, tick_id: &str) -> Result<TickExecutionResult, TickExecutorError> {
        // Layer 3 — multi-depth planning pre-pass
        let planner = Planner::new(&*self.ir);
        let (planned_utility, planning_depth) =
            planner.search_best_depth(tick_id, 3, BTreeMap::new());

        let result =
            self.execute_tick_with_mode(tick_id, TickExecutionMode::Sequential)?;

        // Compare planned vs actual
        let actual = result.reward;
        let delta = actual - planned_utility;

        // Log comparison
        println!(
            "[planner] tick={} planned={} actual={} delta={}",
            tick_id, planned_utility, actual, delta
        );

        Ok(result)
    }

    // W4: Post-execution hook handled inside execute_graph.

    /// Execute a tick with explicit initial inputs provided to the root nodes.
    pub fn execute_tick_with_inputs(
        &mut self,
        tick_id: &str,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<TickExecutionResult, TickExecutorError> {
        self.execute_tick_with_mode_and_inputs(
            tick_id,
            TickExecutionMode::Sequential,
            initial_inputs,
        )
    }

    /// Execute a tick with an explicit mode (sequential or parallel-verified).
    pub fn execute_tick_with_mode(
        &mut self,
        tick_id: &str,
        mode: TickExecutionMode,
    ) -> Result<TickExecutionResult, TickExecutorError> {
        self.execute_tick_with_mode_and_inputs(tick_id, mode, BTreeMap::new())
    }

    /// Execute a tick with an explicit mode and initial inputs.
    pub fn execute_tick_with_mode_and_inputs(
        &mut self,
        tick_id: &str,
        mode: TickExecutionMode,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<TickExecutionResult, TickExecutorError> {
        let tick = self
            .ir
            .ticks
            .iter()
            .find(|t| t.id == tick_id)
            .ok_or_else(|| TickExecutorError::UnknownTick(tick_id.to_string()))?;

        let graph = self
            .ir
            .tick_graphs
            .iter()
            .find(|g| g.id == tick.graph)
            .ok_or_else(|| TickExecutorError::UnknownGraph(tick.graph.clone()))?;

        let graph_cloned = graph.clone();
        let tick_id_owned = tick.id.clone();

        self.execute_graph(&graph_cloned, &tick_id_owned, mode, &initial_inputs)
    }

    /// Execute a tick graph in topological order.
    /// Graph must be acyclic (Canon Line 48).
    fn execute_graph(
        &mut self,
        graph: &TickGraph,
        tick_id: &str,
        mode: TickExecutionMode,
        initial_inputs: &BTreeMap<String, Value>,
    ) -> Result<TickExecutionResult, TickExecutorError> {
        // Build dependency map
        let dependencies = self.build_dependency_map(graph);

        // Compute topological order
        let execution_order = self.topological_sort(graph, &dependencies)?;

        // Initialize context
        let mut context = ExecutionContext::new(initial_inputs.clone());

        // Execute functions in order
        let mut results = HashMap::new();
        let sequential_start = Instant::now();
        let mut function_executor = FunctionExecutor::new(self.ir);

        for function_id in &execution_order {
            // Gather inputs from previous function outputs
            let inputs =
                self.gather_inputs(function_id, &dependencies, &results, initial_inputs)?;

            // Execute function
            let outputs = function_executor
                .execute_by_id(function_id, inputs, &mut context)
                .map_err(TickExecutorError::Executor)?;

            // Store outputs
            results.insert(function_id.clone(), outputs);
        }
        let sequential_duration = sequential_start.elapsed();

        let mut parallel_duration = None;

        // W4: World-model reconciliation occurs before returning execution result.
        if matches!(mode, TickExecutionMode::ParallelVerified) {
            let parallel_start = Instant::now();
            let (parallel_results, parallel_deltas) =
                self.execute_parallel(&execution_order, &dependencies, initial_inputs)?;
            parallel_duration = Some(parallel_start.elapsed());
            self.verify_parallel_outputs(&results, &parallel_results)?;
            self.verify_parallel_deltas(context.deltas(), &parallel_deltas)?;
        }

        // W4 — World-model update step post-execution
        // NOTE: CanonicalIr is currently held immutably by TickExecutor.
        // WorldModel mutation requires an explicit mutable integration point.
        // This hook marks the post-execution update boundary.

        let result = TickExecutionResult {
            tick_id: tick_id.to_string(),
            function_results: results,
            execution_order,
            emitted_deltas: context.deltas().to_vec(),
            reward: compute_reward_from_deltas(context.deltas()),
            sequential_duration,
            parallel_duration,
        };

        // --- World Model Reconciliation (W4) ---
        let actual_delta_ids: Vec<_> = result
            .emitted_deltas
            .iter()
            .map(|d| d.delta_id.clone())
            .collect();

        let record = PredictionRecord::new(
            result.tick_id.clone(),
            Vec::new(),
            actual_delta_ids,
        );

        self.ir.world_model.record_prediction(record);

        // --- Reward Logging ---
        self.ir.reward_deltas.push(RewardRecord {
            id: format!("reward-{}", result.tick_id),
            tick: result.tick_id.clone(),
            execution: String::new(),
            delta: Some(result.reward),
            reward: result.reward,
        });

        Ok(result)
    }

    // W4: PredictionRecord creation logic resides in world_model module.

    fn build_dependency_map(&self, graph: &TickGraph) -> HashMap<FunctionId, Vec<FunctionId>> {
        let mut dependencies: HashMap<FunctionId, Vec<FunctionId>> = HashMap::new();

        // Initialize all nodes
        for node in &graph.nodes {
            dependencies.entry(node.clone()).or_default();
        }

        // Add edges (from -> to means to depends on from)
        for edge in &graph.edges {
            dependencies
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }

        dependencies
    }

    fn topological_sort(
        &self,
        graph: &TickGraph,
        dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
    ) -> Result<Vec<FunctionId>, TickExecutorError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut in_progress = HashSet::new();

        for node in &graph.nodes {
            if !visited.contains(node) {
                self.visit_node(
                    node,
                    dependencies,
                    &mut visited,
                    &mut in_progress,
                    &mut sorted,
                )?;
            }
        }

        Ok(sorted)
    }

    fn visit_node(
        &self,
        node: &FunctionId,
        dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
        visited: &mut HashSet<FunctionId>,
        in_progress: &mut HashSet<FunctionId>,
        sorted: &mut Vec<FunctionId>,
    ) -> Result<(), TickExecutorError> {
        // Cycle detection (Canon Line 48: graphs must be acyclic)
        if in_progress.contains(node) {
            return Err(TickExecutorError::CycleDetected(node.clone()));
        }

        if visited.contains(node) {
            return Ok(());
        }

        in_progress.insert(node.clone());

        // Visit dependencies first
        if let Some(deps) = dependencies.get(node) {
            for dep in deps {
                self.visit_node(dep, dependencies, visited, in_progress, sorted)?;
            }
        }

        in_progress.remove(node);
        visited.insert(node.clone());
        sorted.push(node.clone());

        Ok(())
    }

    fn gather_inputs(
        &self,
        function_id: &FunctionId,
        dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
        results: &HashMap<FunctionId, BTreeMap<String, Value>>,
        initial_inputs: &BTreeMap<String, Value>,
    ) -> Result<BTreeMap<String, Value>, TickExecutorError> {
        let mut inputs = initial_inputs.clone();

        // Gather outputs from all dependencies
        if let Some(deps) = dependencies.get(function_id) {
            for dep in deps {
                if let Some(outputs) = results.get(dep) {
                    // Merge outputs (Canon Line 30: composition)
                    inputs.extend(outputs.clone());
                }
            }
        }

        Ok(inputs)
    }

    fn execute_parallel(
        &self,
        execution_order: &[FunctionId],
        dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
        initial_inputs: &BTreeMap<String, Value>,
    ) -> Result<
        (
            HashMap<FunctionId, BTreeMap<String, Value>>,
            Vec<DeltaValue>,
        ),
        TickExecutorError,
    > {
        let batches = partition_independent_batches(execution_order, dependencies);
        let mut results = HashMap::new();
        let mut deltas = Vec::new();

        for batch in batches {
            if batch.is_empty() {
                continue;
            }

            let mut jobs = Vec::new();
            for function_id in &batch {
                let inputs =
                    self.gather_inputs(function_id, dependencies, &results, initial_inputs)?;
                jobs.push(ParallelJob {
                    function: function_id.clone(),
                    inputs,
                });
            }

            let worker = |function_id: &FunctionId,
                          inputs: BTreeMap<String, Value>|
             -> Result<ParallelJobResult, ExecutorError> {
                let mut local_context = ExecutionContext::new(initial_inputs.clone());
                let function_executor = FunctionExecutor::new(self.ir);
                let outputs = function_executor
                    .execute_by_id(function_id, inputs, &mut local_context)?;
                Ok(ParallelJobResult {
                    function: function_id.clone(),
                    outputs,
                    deltas: local_context.deltas().to_vec(),
                })
            };

            let batch_results = execute_jobs(jobs, &worker).map_err(TickExecutorError::Executor)?;

            let mut batch_delta_map: HashMap<FunctionId, Vec<DeltaValue>> = HashMap::new();
            for result in batch_results {
                batch_delta_map.insert(result.function.clone(), result.deltas);
                results.insert(result.function, result.outputs);
            }

            for function_id in &batch {
                if let Some(delta_list) = batch_delta_map.remove(function_id) {
                    deltas.extend(delta_list);
                }
            }
        }

        Ok((results, deltas))
    }

    fn verify_parallel_outputs(
        &self,
        sequential: &HashMap<FunctionId, BTreeMap<String, Value>>,
        parallel: &HashMap<FunctionId, BTreeMap<String, Value>>,
    ) -> Result<(), TickExecutorError> {
        if sequential.len() != parallel.len() {
            return Err(TickExecutorError::ParallelMismatch {
                function: "<count mismatch>".into(),
            });
        }

        for (function, seq_outputs) in sequential {
            match parallel.get(function) {
                Some(p_outputs) if p_outputs == seq_outputs => continue,
                Some(_) => {
                    return Err(TickExecutorError::ParallelMismatch {
                        function: function.clone(),
                    });
                }
                None => {
                    return Err(TickExecutorError::ParallelMismatch {
                        function: function.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    fn verify_parallel_deltas(
        &self,
        sequential: &[DeltaValue],
        parallel: &[DeltaValue],
    ) -> Result<(), TickExecutorError> {
        if sequential.len() != parallel.len() {
            return Err(TickExecutorError::ParallelDeltaMismatch {
                index: sequential.len().min(parallel.len()),
            });
        }

        for (idx, (seq, par)) in sequential.iter().zip(parallel.iter()).enumerate() {
            if seq != par {
                return Err(TickExecutorError::ParallelDeltaMismatch { index: idx });
            }
        }

        Ok(())
    }
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

// W4 complete: execution result contains sufficient data for model update.

#[derive(Debug, Error)]
pub enum TickExecutorError {
    #[error("unknown tick `{0}`")]
    UnknownTick(String),
    #[error("unknown graph `{0}`")]
    UnknownGraph(String),
    #[error("cycle detected in tick graph at function `{0}` (Canon Line 48 violation)")]
    CycleDetected(FunctionId),
    #[error("parallel execution mismatch for function `{function}`")]
    ParallelMismatch { function: FunctionId },
    #[error("parallel delta mismatch at index {index}")]
    ParallelDeltaMismatch { index: usize },
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}

// End of TickExecutor — world-model update integrated (Layer 2).
