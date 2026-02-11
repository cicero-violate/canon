//! System graph interpreter.
//!
//! Executes [`SystemGraph`](crate::ir::SystemGraph) nodes strictly in topological order.

use std::collections::{BTreeMap, HashMap, HashSet};

use thiserror::Error;

use crate::ir::{
    CanonicalIr, Function, FunctionId, SystemEdge, SystemGraph, SystemNode, SystemNodeId,
    SystemNodeKind,
};
use crate::memory::delta::{Delta, DeltaError, Source};
use crate::memory::epoch::Epoch;
use crate::memory::primitives::{DeltaID, PageID};
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::{ExecutorError, FunctionExecutor};
use crate::runtime::value::{DeltaValue, ScalarValue, StructValue, Value};

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
            let inputs =
                self.gather_inputs(node_id, &dependencies, &results, &initial_inputs);
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

    fn index_nodes(
        graph: &SystemGraph,
    ) -> Result<HashMap<SystemNodeId, &SystemNode>, SystemInterpreterError> {
        let mut map = HashMap::new();
        for node in &graph.nodes {
            if map.insert(node.id.clone(), node).is_some() {
                return Err(SystemInterpreterError::DuplicateNode(node.id.clone()));
            }
        }
        Ok(map)
    }

    fn index_functions(&self) -> HashMap<FunctionId, &Function> {
        self.ir
            .functions
            .iter()
            .map(|function| (function.id.clone(), function))
            .collect()
    }

    fn validate_nodes(
        &self,
        node_index: &HashMap<SystemNodeId, &SystemNode>,
        function_index: &HashMap<FunctionId, &Function>,
        events: &mut Vec<SystemExecutionEvent>,
    ) -> Result<(), SystemInterpreterError> {
        for node in node_index.values() {
            let function = function_index
                .get(&node.function)
                .ok_or_else(|| SystemInterpreterError::UnknownFunction {
                    node: node.id.clone(),
                    function: node.function.clone(),
                })?;
            if !function.contract.total {
                return Err(SystemInterpreterError::NonTotalFunction(function.id.clone()));
            }
        }
        events.push(SystemExecutionEvent::Validation {
            check: "node_contracts",
            detail: format!("validated {} node contracts", node_index.len()),
        });
        Ok(())
    }

    fn build_dependency_map(
        &self,
        graph: &SystemGraph,
        node_index: &HashMap<SystemNodeId, &SystemNode>,
    ) -> Result<HashMap<SystemNodeId, Vec<SystemNodeId>>, SystemInterpreterError> {
        let mut dependencies: HashMap<SystemNodeId, Vec<SystemNodeId>> = HashMap::new();
        for node in node_index.keys() {
            dependencies.entry(node.clone()).or_default();
        }

        for edge in &graph.edges {
            if edge.from == edge.to {
                return Err(SystemInterpreterError::SelfLoop(edge.from.clone()));
            }
            if !node_index.contains_key(&edge.from) || !node_index.contains_key(&edge.to) {
                return Err(SystemInterpreterError::InvalidEdge {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                });
            }

            dependencies
                .entry(edge.to.clone())
                .or_default()
                .push(edge.from.clone());
        }

        Ok(dependencies)
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

    fn apply_node_effects(
        &self,
        node: &SystemNode,
        outputs: &BTreeMap<String, Value>,
        emitted_deltas: &[Delta],
        proofs: &mut Vec<ProofArtifact>,
        delta_records: &mut Vec<DeltaEmission>,
        events: &mut Vec<SystemExecutionEvent>,
    ) -> Result<(), SystemInterpreterError> {
        match node.kind {
            SystemNodeKind::Function => Ok(()),
            SystemNodeKind::Gate => {
                let proof = self.extract_gate_proof(node, outputs)?;
                if !proof.accepted {
                    return Err(SystemInterpreterError::GateRejected(node.id.clone()));
                }
                events.push(SystemExecutionEvent::ProofRecorded {
                    node_id: proof.node_id.clone(),
                    proof_id: proof.proof_id,
                });
                proofs.push(proof);
                Ok(())
            }
            SystemNodeKind::Persist => {
                self.ensure_struct_output(node, outputs, "Record")?;
                if emitted_deltas.is_empty() {
                    return Err(SystemInterpreterError::PersistWithoutDelta(
                        node.id.clone(),
                    ));
                }
                delta_records.push(DeltaEmission {
                    node_id: node.id.clone(),
                    deltas: emitted_deltas.to_vec(),
                });
                events.push(SystemExecutionEvent::DeltaRecorded {
                    node_id: node.id.clone(),
                    count: emitted_deltas.len(),
                });
                Ok(())
            }
            SystemNodeKind::Materialize => {
                self.ensure_struct_output(node, outputs, "Artifact")?;
                Ok(())
            }
        }
    }

    fn extract_gate_proof(
        &self,
        node: &SystemNode,
        outputs: &BTreeMap<String, Value>,
    ) -> Result<ProofArtifact, SystemInterpreterError> {
        let decision = self.ensure_struct_output(node, outputs, "Decision")?;
        let accepted_value =
            self.expect_struct_field(node, "Decision", decision, "Accepted")?;
        let proof_id_value =
            self.expect_struct_field(node, "Decision", decision, "ProofId")?;

        let accepted = match accepted_value {
            Value::Scalar(ScalarValue::Bool(value)) => *value,
            other => {
                return Err(SystemInterpreterError::OutputTypeMismatch {
                    node: node.id.clone(),
                    output: "Decision.Accepted".to_string(),
                    message: format!("expected bool, found {:?}", other.kind()),
                })
            }
        };

        let proof_id = match proof_id_value {
            Value::Scalar(ScalarValue::I32(value)) => *value,
            Value::Scalar(ScalarValue::U32(value)) => *value as i32,
            other => {
                return Err(SystemInterpreterError::OutputTypeMismatch {
                    node: node.id.clone(),
                    output: "Decision.ProofId".to_string(),
                    message: format!("expected integer, found {:?}", other.kind()),
                })
            }
        };

        Ok(ProofArtifact {
            node_id: node.id.clone(),
            proof_id,
            accepted,
        })
    }

    fn ensure_struct_output<'value>(
        &self,
        node: &SystemNode,
        outputs: &'value BTreeMap<String, Value>,
        name: &str,
    ) -> Result<&'value StructValue, SystemInterpreterError> {
        let value = self.expect_output(node, outputs, name)?;
        match value {
            Value::Struct(struct_value) => Ok(struct_value),
            other => Err(SystemInterpreterError::OutputTypeMismatch {
                node: node.id.clone(),
                output: name.to_string(),
                message: format!("expected struct, found {:?}", other.kind()),
            }),
        }
    }

    fn expect_output<'value>(
        &self,
        node: &SystemNode,
        outputs: &'value BTreeMap<String, Value>,
        name: &str,
    ) -> Result<&'value Value, SystemInterpreterError> {
        outputs.get(name).ok_or_else(|| SystemInterpreterError::MissingOutput {
            node: node.id.clone(),
            output: name.to_string(),
        })
    }

    fn expect_struct_field<'value>(
        &self,
        node: &SystemNode,
        output_name: &str,
        value: &'value StructValue,
        field: &str,
    ) -> Result<&'value Value, SystemInterpreterError> {
        value.fields.get(field).ok_or_else(|| {
            SystemInterpreterError::MissingField {
                node: node.id.clone(),
                output: output_name.to_string(),
                field: field.to_string(),
            }
        })
    }

    fn gather_inputs(
        &self,
        node_id: &SystemNodeId,
        dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>,
        results: &HashMap<SystemNodeId, BTreeMap<String, Value>>,
        initial_inputs: &BTreeMap<String, Value>,
    ) -> BTreeMap<String, Value> {
        let mut inputs = initial_inputs.clone();
        if let Some(deps) = dependencies.get(node_id) {
            for dep in deps {
                if let Some(outputs) = results.get(dep) {
                    inputs.extend(outputs.clone());
                }
            }
        }
        inputs
    }

    fn topological_sort(
        &self,
        graph: &SystemGraph,
        dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>,
    ) -> Result<Vec<SystemNodeId>, SystemInterpreterError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut in_progress = HashSet::new();

        for node in &graph.nodes {
            if visited.contains(&node.id) {
                continue;
            }
            self.visit_node(
                &node.id,
                dependencies,
                &mut visited,
                &mut in_progress,
                &mut sorted,
            )?;
        }

        Ok(sorted)
    }

    fn visit_node(
        &self,
        node_id: &SystemNodeId,
        dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>,
        visited: &mut HashSet<SystemNodeId>,
        in_progress: &mut HashSet<SystemNodeId>,
        sorted: &mut Vec<SystemNodeId>,
    ) -> Result<(), SystemInterpreterError> {
        if in_progress.contains(node_id) {
            return Err(SystemInterpreterError::CycleDetected(node_id.clone()));
        }

        if visited.contains(node_id) {
            return Ok(());
        }

        in_progress.insert(node_id.clone());

        if let Some(deps) = dependencies.get(node_id) {
            for dep in deps {
                self.visit_node(dep, dependencies, visited, in_progress, sorted)?;
            }
        }

        in_progress.remove(node_id);
        visited.insert(node_id.clone());
        sorted.push(node_id.clone());
        Ok(())
    }

    fn materialize_delta(
        &self,
        value: &DeltaValue,
        sequence: u64,
    ) -> Result<Delta, SystemInterpreterError> {
        let payload = value.payload_hash.as_bytes().to_vec();
        let mask = vec![true; payload.len()];
        let page_id = PageID(sequence.saturating_add(1));
        let parsed_id = value.delta_id.parse::<u64>().unwrap_or(sequence.saturating_add(1));
        Delta::new_dense(
            DeltaID(parsed_id),
            page_id,
            Epoch(0),
            payload,
            mask,
            Source(format!("system_interpreter:{}", value.delta_id)),
        )
        .map_err(|err| SystemInterpreterError::DeltaMaterialization(err.to_string()))
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
    Validation { check: &'static str, detail: String },
    NodeExecuted { node_id: SystemNodeId, kind: SystemNodeKind },
    ProofRecorded { node_id: SystemNodeId, proof_id: i32 },
    DeltaRecorded { node_id: SystemNodeId, count: usize },
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
    UnknownFunction { node: SystemNodeId, function: FunctionId },
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
    InvalidEdge { from: SystemNodeId, to: SystemNodeId },
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
