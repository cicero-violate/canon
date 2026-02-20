use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{Function, FunctionId, SystemGraph, SystemNode, SystemNodeId};
use crate::runtime::value::Value;

use super::{SystemInterpreter, SystemInterpreterError};

impl<'a> SystemInterpreter<'a> {
    pub(super) fn index_nodes(graph: &SystemGraph) -> Result<HashMap<SystemNodeId, &SystemNode>, SystemInterpreterError> {
        let mut map = HashMap::new();
        for node in &graph.nodes {
            if map.insert(node.id.clone(), node).is_some() {
                return Err(SystemInterpreterError::DuplicateNode(node.id.clone()));
            }
        }
        Ok(map)
    }

    pub(super) fn index_functions(&self) -> HashMap<FunctionId, &Function> {
        self.ir.functions.iter().map(|function| (function.id.clone(), function)).collect()
    }

    pub(super) fn validate_nodes(
        &self, node_index: &HashMap<SystemNodeId, &SystemNode>, function_index: &HashMap<FunctionId, &Function>, events: &mut Vec<super::SystemExecutionEvent>,
    ) -> Result<(), SystemInterpreterError> {
        for node in node_index.values() {
            let function = function_index.get(&node.function).ok_or_else(|| SystemInterpreterError::UnknownFunction { node: node.id.clone(), function: node.function.clone() })?;
            if !function.contract.total {
                return Err(SystemInterpreterError::NonTotalFunction(function.id.clone()));
            }
        }
        events.push(super::SystemExecutionEvent::Validation { check: "node_contracts", detail: format!("validated {} node contracts", node_index.len()) });
        Ok(())
    }

    pub(super) fn build_dependency_map(&self, graph: &SystemGraph, node_index: &HashMap<SystemNodeId, &SystemNode>) -> Result<HashMap<SystemNodeId, Vec<SystemNodeId>>, SystemInterpreterError> {
        let mut dependencies: HashMap<SystemNodeId, Vec<SystemNodeId>> = HashMap::new();
        for node in node_index.keys() {
            dependencies.entry(node.clone()).or_default();
        }

        for edge in &graph.edges {
            if edge.from == edge.to {
                return Err(SystemInterpreterError::SelfLoop(edge.from.clone()));
            }
            if !node_index.contains_key(&edge.from) || !node_index.contains_key(&edge.to) {
                return Err(SystemInterpreterError::InvalidEdge { from: edge.from.clone(), to: edge.to.clone() });
            }

            dependencies.entry(edge.to.clone()).or_default().push(edge.from.clone());
        }

        Ok(dependencies)
    }

    pub(super) fn topological_sort(&self, graph: &SystemGraph, dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>) -> Result<Vec<SystemNodeId>, SystemInterpreterError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        let mut in_progress = HashSet::new();

        for node in &graph.nodes {
            if visited.contains(&node.id) {
                continue;
            }
            self.visit_node(&node.id, dependencies, &mut visited, &mut in_progress, &mut sorted)?;
        }

        Ok(sorted)
    }

    fn visit_node(
        &self, node_id: &SystemNodeId, dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>, visited: &mut HashSet<SystemNodeId>, in_progress: &mut HashSet<SystemNodeId>,
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

    pub(super) fn gather_inputs(
        &self, node_id: &SystemNodeId, dependencies: &HashMap<SystemNodeId, Vec<SystemNodeId>>, results: &HashMap<SystemNodeId, BTreeMap<String, Value>>, initial_inputs: &BTreeMap<String, Value>,
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
}
