use crate::ir::{FunctionId, TickGraph};
use crate::runtime::error::RuntimeError;
use crate::runtime::value::Value;
use std::collections::{BTreeMap, HashMap};

pub fn build_dependency_map(graph: &TickGraph) -> HashMap<FunctionId, Vec<FunctionId>> {
    let mut map = HashMap::new();
    for edge in &graph.edges {
        map.entry(edge.to.clone())
            .or_insert_with(Vec::new)
            .push(edge.from.clone());
    }
    map
}

pub fn topological_sort(
    graph: &TickGraph,
    dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
) -> Result<Vec<FunctionId>, RuntimeError> {
    let mut sorted = Vec::new();
    let mut visited = HashMap::new();

    fn visit(
        node: &FunctionId,
        dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
        visited: &mut HashMap<FunctionId, bool>,
        sorted: &mut Vec<FunctionId>,
    ) -> Result<(), RuntimeError> {
        if let Some(true) = visited.get(node) {
            return Ok(());
        }
        visited.insert(node.clone(), true);
        if let Some(deps) = dependencies.get(node) {
            for dep in deps {
                visit(dep, dependencies, visited, sorted)?;
            }
        }
        sorted.push(node.clone());
        Ok(())
    }

    for node in &graph.nodes {
        visit(node, dependencies, &mut visited, &mut sorted)?;
    }

    Ok(sorted)
}

pub fn gather_inputs(
    function_id: &FunctionId,
    dependencies: &HashMap<FunctionId, Vec<FunctionId>>,
    results: &HashMap<FunctionId, BTreeMap<String, Value>>,
    initial_inputs: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, Value>, RuntimeError> {
    let mut inputs = initial_inputs.clone();
    if let Some(deps) = dependencies.get(function_id) {
        for dep in deps {
            if let Some(outputs) = results.get(dep) {
                for (k, v) in outputs {
                    inputs.insert(k.clone(), v.clone());
                }
            }
        }
    }
    Ok(inputs)
}
