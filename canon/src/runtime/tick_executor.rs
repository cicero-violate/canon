//! Tick graph executor.
//!
//! Executes an entire tick graph, respecting data dependencies (Canon Lines 46-50).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;

use crate::ir::{CanonicalIr, FunctionId, Tick, TickGraph};
use crate::ir::world_model::PredictionRecord;
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::{ExecutorError, FunctionExecutor};
use crate::runtime::parallel::{
    ParallelJob, ParallelJobResult, execute_jobs, partition_independent_batches,
};
use crate::runtime::value::{DeltaValue, Value};

fn compute_reward_from_deltas(emitted: &[DeltaValue]) -> f64 {
    // Layer 1 (Foundation): deterministic scalar utility.
    // Minimal rule: U = (# emitted deltas)
    emitted.len() as f64
}

/// Executes a tick graph in topological order.
pub struct TickExecutor<'a> {
    ir: &'a CanonicalIr,
    function_executor: FunctionExecutor<'a>,
}

/// Execution mode for tick graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickExecutionMode {
    Sequential,
    ParallelVerified,
}

impl<'a> TickExecutor<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self {
            ir,
            function_executor: FunctionExecutor::new(ir),
        }
    }

    /// Execute a tick by its ID.
    pub fn execute_tick(&self, tick_id: &str) -> Result<TickExecutionResult, TickExecutorError> {
        self.execute_tick_with_mode(tick_id, TickExecutionMode::Sequential)
    }

    /// Execute a tick with explicit initial inputs provided to the root nodes.
    pub fn execute_tick_with_inputs(
        &self,
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
        &self,
        tick_id: &str,
        mode: TickExecutionMode,
    ) -> Result<TickExecutionResult, TickExecutorError> {
        self.execute_tick_with_mode_and_inputs(tick_id, mode, BTreeMap::new())
    }

    /// Execute a tick with an explicit mode and initial inputs.
    pub fn execute_tick_with_mode_and_inputs(
        &self,
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

        self.execute_graph(graph, tick, mode, &initial_inputs)
    }

    /// Execute a tick graph in topological order.
    /// Graph must be acyclic (Canon Line 48).
    fn execute_graph(
        &self,
        graph: &TickGraph,
        tick: &Tick,
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
        for function_id in &execution_order {
            // Gather inputs from previous function outputs
            let inputs =
                self.gather_inputs(function_id, &dependencies, &results, initial_inputs)?;

            // Execute function
            let outputs = self
                .function_executor
                .execute_by_id(function_id, inputs, &mut context)
                .map_err(TickExecutorError::Executor)?;

            // Store outputs
            results.insert(function_id.clone(), outputs);
        }
        let sequential_duration = sequential_start.elapsed();

        let mut parallel_duration = None;
        if matches!(mode, TickExecutionMode::ParallelVerified) {
            let parallel_start = Instant::now();
            let (parallel_results, parallel_deltas) =
                self.execute_parallel(&execution_order, &dependencies, initial_inputs)?;
            parallel_duration = Some(parallel_start.elapsed());
            self.verify_parallel_outputs(&results, &parallel_results)?;
            self.verify_parallel_deltas(context.deltas(), &parallel_deltas)?;
        }

        Ok(TickExecutionResult {
            tick_id: tick.id.clone(),
            function_results: results,
            execution_order,
            emitted_deltas: context.deltas().to_vec(),
            reward: compute_reward_from_deltas(context.deltas()),
            sequential_duration,
            parallel_duration,
        })
    }

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
                let outputs = self.function_executor.execute_by_id(
                    function_id,
                    inputs,
                    &mut local_context,
                )?;
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
