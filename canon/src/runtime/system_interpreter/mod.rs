//! System graph interpreter.
//!
//! Executes [`SystemGraph`](crate::ir::SystemGraph) nodes strictly in topological order.

use std::collections::{BTreeMap, HashMap};
use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

use crate::ir::{CanonicalIr, FunctionId, SystemGraph, SystemNode, SystemNodeId, SystemNodeKind};
use crate::runtime::context::ExecutionContext;
use crate::runtime::executor::{ExecutorError, FunctionExecutor};
use crate::runtime::value::Value;
use blake3::Hasher;
use database::delta::Delta as EngineDelta;
use database::delta::delta_types::{DeltaError, Source};
use database::epoch::Epoch;
use database::primitives::{DeltaID, Hash as EngineHash, PageID};
use database::{
    AdmissionProof, CommitProof, JudgmentProof, MemoryEngine, MemoryEngineError, OutcomeProof,
};

mod effects;
mod planner;

/// Executes Canon system graphs.
pub struct SystemInterpreter<'a> {
    ir: &'a CanonicalIr,
    executor: FunctionExecutor<'a>,
    engine: &'a MemoryEngine,
}

impl<'a> SystemInterpreter<'a> {
    pub fn new(ir: &'a CanonicalIr, engine: &'a MemoryEngine) -> Self {
        Self {
            ir,
            executor: FunctionExecutor::new(ir),
            engine,
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

        for node_id in &order {
            let node = node_index
                .get(node_id)
                .copied()
                .ok_or_else(|| SystemInterpreterError::UnknownNode(node_id.clone()))?;
            let inputs = self.gather_inputs(node_id, &dependencies, &results, &initial_inputs);
            let delta_start = context.deltas().len();
            let outputs = self.execute_node(node, inputs, &mut context)?;
            let emitted_slice = context.deltas()[delta_start..].to_vec();
            for value in emitted_slice {
                emitted_deltas.push(value.clone());
            }
            self.apply_node_effects(
                node,
                &outputs,
                &emitted_deltas,
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

        let engine_artifacts = self.commit_emitted_deltas(&graph.id, &emitted_deltas)?;

        Ok(SystemExecutionResult {
            graph_id: graph.id.clone(),
            node_results: results,
            execution_order: order,
            emitted_deltas,
            proof_artifacts: proofs,
            delta_provenance,
            events,
            judgment_proof: engine_artifacts
                .as_ref()
                .map(|artifacts| artifacts.judgment_proof.clone()),
            admission_proof: engine_artifacts
                .as_ref()
                .map(|artifacts| artifacts.admission.clone()),
            commit_proofs: engine_artifacts
                .as_ref()
                .map(|artifacts| artifacts.commits.clone())
                .unwrap_or_default(),
            outcome_proofs: engine_artifacts
                .as_ref()
                .map(|artifacts| artifacts.outcomes.clone())
                .unwrap_or_default(),
            event_hashes: engine_artifacts
                .as_ref()
                .map(|artifacts| artifacts.event_hashes.clone())
                .unwrap_or_default(),
            state_root: engine_artifacts
                .as_ref()
                .and_then(|artifacts| artifacts.commits.last().map(|commit| commit.state_hash)),
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
    pub emitted_deltas: Vec<crate::runtime::value::DeltaValue>,
    pub proof_artifacts: Vec<ProofArtifact>,
    pub delta_provenance: Vec<DeltaEmission>,
    pub events: Vec<SystemExecutionEvent>,
    pub judgment_proof: Option<JudgmentProof>,
    pub admission_proof: Option<AdmissionProof>,
    pub commit_proofs: Vec<CommitProof>,
    pub outcome_proofs: Vec<OutcomeProof>,
    pub event_hashes: Vec<EngineHash>,
    pub state_root: Option<EngineHash>,
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
    pub deltas: Vec<crate::runtime::value::DeltaValue>,
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
    #[error("cycle detected in system graph at `{0}`")]
    CycleDetected(SystemNodeId),
    #[error("unable to encode delta `{delta_id}`: {source}")]
    DeltaEncoding {
        delta_id: String,
        #[source]
        source: DeltaError,
    },
    #[error("memory engine error: {0}")]
    MemoryEngine(#[from] MemoryEngineError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
}

impl<'a> SystemInterpreter<'a> {
    fn commit_emitted_deltas(
        &self,
        graph_id: &str,
        deltas: &[crate::runtime::value::DeltaValue],
    ) -> Result<Option<EngineArtifacts>, SystemInterpreterError> {
        if deltas.is_empty() {
            return Ok(None);
        }

        let encoded = deltas
            .iter()
            .map(Self::encode_delta)
            .collect::<Result<Vec<_>, _>>()?;

        let delta_hashes: Vec<EngineHash> = encoded
            .into_iter()
            .map(|delta| self.engine.register_delta(delta))
            .collect();

        let judgment_proof = Self::build_judgment_proof(graph_id, deltas);
        let admission = self
            .engine
            .admit_execution(&judgment_proof)
            .map_err(|err| SystemInterpreterError::MemoryEngine(err.into()))?;
        let commits = self.engine.commit_batch(&admission, &delta_hashes)?;

        let mut outcomes = Vec::with_capacity(commits.len());
        let mut event_hashes = Vec::with_capacity(commits.len());

        for commit in &commits {
            let outcome = self.engine.record_outcome(commit);
            event_hashes.push(self.engine.compute_event_hash(&admission, commit, &outcome));
            outcomes.push(outcome);
        }

        Ok(Some(EngineArtifacts {
            judgment_proof,
            admission,
            commits,
            outcomes,
            event_hashes,
        }))
    }

    fn encode_delta(
        delta: &crate::runtime::value::DeltaValue,
    ) -> Result<EngineDelta, SystemInterpreterError> {
        let payload = Self::decode_payload(&delta.payload_hash);
        let mask = vec![true; payload.len()];
        EngineDelta::new_dense(
            DeltaID(Self::deterministic_id(&delta.delta_id)),
            PageID(Self::deterministic_id(&format!("page::{}", delta.delta_id))),
            Epoch(0),
            payload,
            mask,
            Source(format!("canon.delta::{}", delta.delta_id)),
        )
        .map_err(|source| SystemInterpreterError::DeltaEncoding {
            delta_id: delta.delta_id.clone(),
            source,
        })
    }

    fn build_judgment_proof(
        graph_id: &str,
        deltas: &[crate::runtime::value::DeltaValue],
    ) -> JudgmentProof {
        let mut hasher = Hasher::new();
        hasher.update(graph_id.as_bytes());
        for delta in deltas {
            hasher.update(delta.delta_id.as_bytes());
            hasher.update(delta.payload_hash.as_bytes());
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&hasher.finalize().as_bytes()[..32]);
        JudgmentProof {
            approved: true,
            timestamp: Self::current_timestamp(),
            hash,
        }
    }

    fn deterministic_id(value: &str) -> u64 {
        let hash = blake3::hash(value.as_bytes());
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&hash.as_bytes()[..8]);
        u64::from_le_bytes(bytes)
    }

    fn decode_payload(payload_hash: &str) -> Vec<u8> {
        if payload_hash.is_empty() {
            return Vec::new();
        }
        if payload_hash.len() % 2 == 0 {
            if let Ok(bytes) = hex::decode(payload_hash) {
                return bytes;
            }
        }
        payload_hash.as_bytes().to_vec()
    }

    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

#[derive(Debug, Clone)]
struct EngineArtifacts {
    judgment_proof: JudgmentProof,
    admission: AdmissionProof,
    commits: Vec<CommitProof>,
    outcomes: Vec<OutcomeProof>,
    event_hashes: Vec<EngineHash>,
}
