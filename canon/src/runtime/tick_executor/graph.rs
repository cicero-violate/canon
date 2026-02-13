//! Graph traversal: dependency map, topological sort, input gathering.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ir::{FunctionId, TickGraph};
use crate::runtime::value::Value;

use super::types::TickExecutorError;

pub(super) fn build_dependency_map(graph: &TickGraph) -> HashMap<FunctionId, Vec<FunctionId>> {
    let mut dependencies: HashMap<FunctionId, Vec<FunctionId>> = HashMap::new();
    for node in &graph.nodes {
        dependencies.entry(node.clone()).or_default();
    }
    for edge in &graph.edges {
        dependencies
            .entry(edge.to.clone())
            .or_default()
            .push(edge.from.clone());
    }
    dependencies
}

pub(super) fn topological_sort(
    graph: &TickGraph,
    dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
) -> Result<Vec<FunctionId>, TickExecutorError> {
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    let mut in_progress = HashSet::new();

    for node in &graph.nodes {
        if !visited.contains(node) {
            visit_node(
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
    if let Some(deps) = dependencies.get(node) {
        for dep in deps {
            visit_node(dep, dependencies, visited, in_progress, sorted)?;
        }
    }
    in_progress.remove(node);
    visited.insert(node.clone());
    sorted.push(node.clone());
    Ok(())
}

pub(super) fn gather_inputs(
    function_id: &FunctionId,
    dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
    results: &HashMap<FunctionId, BTreeMap<String, Value>>,
    initial_inputs: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, Value>, TickExecutorError> {
    let mut inputs = initial_inputs.clone();
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
