//! System graph interpreter.
//!
//! Executes [`SystemGraph`](crate::ir::SystemGraph) nodes strictly in topological order.

use std::collections::{BTreeMap, HashMap};

use thiserror::Error;

use crate::ir::{CanonicalIr, FunctionId, SystemGraph, SystemNode, SystemNodeId, SystemNodeKind};
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::{ExecutorError, FunctionExecutor};
use crate::runtime::value::Value;
use memory_engine::Delta;

mod delta;
mod effects;
mod planner;

/// Executes Canon system graphs.
pub struct SystemInterpreter<'a> {
    ir: &'a CanonicalIr,
    executor: FunctionExecutor<'a>,
}

impl<'a> SystemInterpreter<'a> {
    pub fn new(ir: &'a CanonicalIr) -> Self {
        Self {
            ir,
            executor: FunctionExecutor::new(ir),
        }
    }

    /// Execute a system graph by ID.
    pub fn execute_graph(
        &self,
        graph_id: &str,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<SystemExecutionResult, SystemInterpreterError> {
        let graph = self
            .ir
            .system_graphs
            .iter()
            .find(|g| g.id == graph_id)
            .ok_or_else(|| SystemInterpreterError::UnknownGraph(graph_id.to_string()))?;

        self.run_graph(graph, initial_inputs)
    }

    /// Execute an inline system graph that isn't stored in the IR.
    pub fn execute_inline(
        &self,
        graph: &SystemGraph,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<SystemExecutionResult, SystemInterpreterError> {
        self.run_graph(graph, initial_inputs)
    }

    fn run_graph(
        &self,
        graph: &SystemGraph,
        initial_inputs: BTreeMap<String, Value>,
    ) -> Result<SystemExecutionResult, SystemInterpreterError> {
        let mut events = Vec::new();
        let node_index = Self::index_nodes(graph)?;
        let function_index = self.index_functions();
        self.validate_nodes(&node_index, &function_index, &mut events)?;
        let dependencies = self.build_dependency_map(graph, &node_index)?;
        events.push(SystemExecutionEvent::Validation {
            check: "edges_declared",
            detail: format!("validated {} edges", graph.edges.len()),
        });
        let order = self.topological_sort(graph, &dependencies)?;
        events.push(SystemExecutionEvent::Validation {
            check: "topology",
            detail: format!("computed execution order of {} nodes", order.len()),
        });
        let mut context = ExecutionContext::new(initial_inputs.clone());
        let mut results: HashMap<SystemNodeId, BTreeMap<String, Value>> = HashMap::new();
        let mut proofs = Vec::new();
        let mut delta_provenance = Vec::new();
        let mut emitted_deltas = Vec::new();
        let mut delta_counter: u64 = 0;

        for node_id in &order {
            let node = node_index
                .get(node_id)
                .copied()
                .ok_or_else(|| SystemInterpreterError::UnknownNode(node_id.clone()))?;
            let inputs = self.gather_inputs(node_id, &dependencies, &results, &initial_inputs);
            let delta_start = context.deltas().len();
            let outputs = self.execute_node(node, inputs, &mut context)?;
            let emitted_slice = context.deltas()[delta_start..].to_vec();
            let mut concrete_deltas = Vec::new();
            for value in emitted_slice {
                let delta = self.materialize_delta(&value, delta_counter)?;
                delta_counter = delta_counter.saturating_add(1);
                emitted_deltas.push(delta.clone());
                concrete_deltas.push(delta);
            }
            self.apply_node_effects(
                node,
                &outputs,
                &concrete_deltas,
                &mut proofs,
                &mut delta_provenance,
                &mut events,
            )?;
            results.insert(node.id.clone(), outputs);
            events.push(SystemExecutionEvent::NodeExecuted {
                node_id: node.id.clone(),
                kind: node.kind.clone(),
            });
        }

        Ok(SystemExecutionResult {
            graph_id: graph.id.clone(),
            node_results: results,
            execution_order: order,
            emitted_deltas,
            proof_artifacts: proofs,
            delta_provenance,
            events,
        })
    }

    fn execute_node(
        &self,
        node: &SystemNode,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, SystemInterpreterError> {
        match node.kind {
            SystemNodeKind::Function => self.execute_function(&node.function, inputs, context),
            SystemNodeKind::Gate => self.execute_gate(&node.function, inputs, context),
            SystemNodeKind::Persist => self.execute_persist(&node.function, inputs, context),
            SystemNodeKind::Materialize => {
                self.execute_materialize(&node.function, inputs, context)
            }
        }
    }

    fn execute_function(
        &self,
        function_id: &FunctionId,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, SystemInterpreterError> {
        self.executor
            .execute_by_id(function_id, inputs, context)
            .map_err(SystemInterpreterError::Executor)
    }

    fn execute_gate(
        &self,
        function_id: &FunctionId,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, SystemInterpreterError> {
        // Gate nodes currently behave like pure functions; future revisions will
        // verify proofs before permitting downstream execution.
        self.execute_function(function_id, inputs, context)
    }

    fn execute_persist(
        &self,
        function_id: &FunctionId,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, SystemInterpreterError> {
        // Persist nodes model durable writes; currently they run the bound function.
        self.execute_function(function_id, inputs, context)
    }

    fn execute_materialize(
        &self,
        function_id: &FunctionId,
        inputs: BTreeMap<String, Value>,
        context: &mut ExecutionContext,
    ) -> Result<BTreeMap<String, Value>, SystemInterpreterError> {
        // Materialize nodes convert deltas into artifacts; placeholder implementation.
        self.execute_function(function_id, inputs, context)
    }
}

#[derive(Debug, Clone)]
pub struct SystemExecutionResult {
    pub graph_id: String,
    pub node_results: HashMap<SystemNodeId, BTreeMap<String, Value>>,
    pub execution_order: Vec<SystemNodeId>,
    pub emitted_deltas: Vec<Delta>,
    pub proof_artifacts: Vec<ProofArtifact>,
    pub delta_provenance: Vec<DeltaEmission>,
    pub events: Vec<SystemExecutionEvent>,
}

#[derive(Debug, Clone)]
pub struct ProofArtifact {
    pub node_id: SystemNodeId,
    pub proof_id: i32,
    pub accepted: bool,
}

#[derive(Debug, Clone)]
pub struct DeltaEmission {
    pub node_id: SystemNodeId,
    pub deltas: Vec<Delta>,
}

#[derive(Debug, Clone)]
pub enum SystemExecutionEvent {
    Validation {
        check: &'static str,
        detail: String,
    },
    NodeExecuted {
        node_id: SystemNodeId,
        kind: SystemNodeKind,
    },
    ProofRecorded {
        node_id: SystemNodeId,
        proof_id: i32,
    },
    DeltaRecorded {
        node_id: SystemNodeId,
        count: usize,
    },
}

#[derive(Debug, Error)]
pub enum SystemInterpreterError {
    #[error("unknown system graph `{0}`")]
    UnknownGraph(String),
    #[error("unknown system node `{0}`")]
    UnknownNode(SystemNodeId),
    #[error("duplicate system node `{0}`")]
    DuplicateNode(SystemNodeId),
    #[error("system node `{node}` references unknown function `{function}`")]
    UnknownFunction {
        node: SystemNodeId,
        function: FunctionId,
    },
    #[error("function `{0}` is not total and cannot execute inside the system graph")]
    NonTotalFunction(FunctionId),
    #[error("node `{node}` missing required output `{output}`")]
    MissingOutput { node: SystemNodeId, output: String },
    #[error("node `{node}` struct output `{output}` missing field `{field}`")]
    MissingField {
        node: SystemNodeId,
        output: String,
        field: String,
    },
    #[error("node `{node}` output `{output}` invalid: {message}")]
    OutputTypeMismatch {
        node: SystemNodeId,
        output: String,
        message: String,
    },
    #[error("edge `{from}` -> `{to}` references unknown node")]
    InvalidEdge {
        from: SystemNodeId,
        to: SystemNodeId,
    },
    #[error("edge from `{0}` to itself is not allowed")]
    SelfLoop(SystemNodeId),
    #[error("gate node `{0}` rejected execution")]
    GateRejected(SystemNodeId),
    #[error("persist node `{0}` emitted no deltas")]
    PersistWithoutDelta(SystemNodeId),
    #[error("failed to materialize delta: {0}")]
    DeltaMaterialization(String),
    #[error("cycle detected in system graph at `{0}`")]
    CycleDetected(SystemNodeId),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}
