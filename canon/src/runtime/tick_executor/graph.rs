//! Graph traversal: dependency map, topological sort, input gathering.
use std::collections::{BTreeMap, HashMap, HashSet};
use crate::ir::{FunctionId, ExecutionGraph};
use crate::runtime::value::Value;
use crate::runtime::error::RuntimeError;
pub(super) fn build_dependency_map(
    graph: &ExecutionGraph,
) -> HashMap<FunctionId, Vec<FunctionId>> {
    let mut dependencies: HashMap<FunctionId, Vec<FunctionId>> = HashMap::new();
    for node in &graph.nodes {
        dependencies.entry(node.clone()).or_default();
    }
    for edge in &graph.edges {
        dependencies.entry(edge.to.clone()).or_default().push(edge.from.clone());
    }
    dependencies
}
pub(super) fn topological_sort(
    graph: &ExecutionGraph,
    dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
) -> Result<Vec<FunctionId>, RuntimeError> {
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    let mut in_progress = HashSet::new();
    for node in &graph.nodes {
        if !visited.contains(node) {
            visit_node(node, dependencies, &mut visited, &mut in_progress, &mut sorted)?;
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
) -> Result<(), RuntimeError> {
    if in_progress.contains(node) {
        return Err(RuntimeError::CycleDetected(node.clone()));
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
) -> Result<BTreeMap<String, Value>, RuntimeError> {
    let mut inputs = initial_inputs.clone();
    if let Some(deps) = dependencies.get(function_id) {
        for dep in deps {
            if let Some(outputs) = results.get(dep) {
                inputs.extend(outputs.clone());
            }
        }
    }
    Ok(inputs)
}
