use super::error::Violation;
use super::helpers::{Indexes, module_has_permission};
use super::rules::CanonRule;
use crate::ir::*;
use petgraph::algo::is_cyclic_directed;
use petgraph::graphmap::DiGraphMap;
use std::collections::{HashMap, HashSet};

pub fn check<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    let module_adj = check_module_graph(ir, idx, violations);
    check_call_graph(ir, idx, &module_adj, violations);
    check_tick_graphs(ir, idx, violations);
    check_loop_policies(ir, idx, violations);
}

fn check_module_graph<'a>(
    ir: &'a CanonicalIr,
    idx: &Indexes<'a>,
    violations: &mut Vec<Violation>,
) -> HashMap<&'a str, Vec<&'a str>> {
    let mut graph: DiGraphMap<&str, ()> = DiGraphMap::new();
    for id in idx.modules.keys() {
        graph.add_node(*id);
    }
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in &ir.module_edges {
        let from = edge.source.as_str();
        let to = edge.target.as_str();
        if idx.modules.get(from).is_none() || idx.modules.get(to).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("module edge `{from}` -> `{to}` references unknown modules"),
            ));
            continue;
        }
        if from == to {
            violations.push(Violation::new(
                CanonRule::ModuleSelfImport,
                format!("module `{from}` may not import itself"),
            ));
            continue;
        }
        graph.add_edge(from, to, ());
        adj.entry(from).or_default().push(to);
    }
    if is_cyclic_directed(&graph) {
        violations.push(Violation::new(
            CanonRule::ModuleDag,
            "module import permissions must form a strict DAG",
        ));
    }
    adj
}

fn check_call_graph<'a>(
    ir: &'a CanonicalIr,
    idx: &Indexes<'a>,
    module_adj: &HashMap<&'a str, Vec<&'a str>>,
    violations: &mut Vec<Violation>,
) {
    let mut call_graph: DiGraphMap<&str, ()> = DiGraphMap::new();
    for id in idx.functions.keys() {
        call_graph.add_node(*id);
    }
    let mut call_pairs: HashSet<(FunctionId, FunctionId)> = HashSet::new();
    let mut perm_cache: HashMap<&str, HashSet<&str>> = HashMap::new();

    for edge in &ir.call_edges {
        let Some(caller) = idx.functions.get(edge.caller.as_str()) else {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` references missing caller `{}`",
                    edge.id, edge.caller
                ),
            ));
            continue;
        };
        let Some(callee) = idx.functions.get(edge.callee.as_str()) else {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` references missing callee `{}`",
                    edge.id, edge.callee
                ),
            ));
            continue;
        };
        call_pairs.insert((edge.caller.clone(), edge.callee.clone()));
        call_graph.add_edge(edge.caller.as_str(), edge.callee.as_str(), ());
        if callee.visibility != Visibility::Public {
            violations.push(Violation::new(
                CanonRule::CallGraphPublicApis,
                format!(
                    "call edge `{}` targets private function `{}`",
                    edge.id, callee.name
                ),
            ));
        }
        if !module_has_permission(
            caller.module.as_str(),
            callee.module.as_str(),
            module_adj,
            &mut perm_cache,
        ) {
            violations.push(Violation::new(
                CanonRule::CallGraphRespectsDag,
                format!(
                    "module `{}` lacks permission to call `{}` in `{}`",
                    caller.module, callee.name, callee.module
                ),
            ));
        }
    }
    if is_cyclic_directed(&call_graph) {
        violations.push(Violation::new(
            CanonRule::CallGraphAcyclic,
            "call graph contains recursion which violates Canon rules 29 and 75",
        ));
    }
}

fn check_tick_graphs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let call_pairs: HashSet<(FunctionId, FunctionId)> = ir
        .call_edges
        .iter()
        .map(|e| (e.caller.clone(), e.callee.clone()))
        .collect();

    for graph in &ir.tick_graphs {
        let mut tg: DiGraphMap<&str, ()> = DiGraphMap::new();
        let mut node_set: HashSet<&str> = HashSet::new();
        for node in &graph.nodes {
            if idx.functions.get(node.as_str()).is_none() {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` references unknown function `{node}`",
                        graph.name
                    ),
                ));
                continue;
            }
            node_set.insert(node.as_str());
            tg.add_node(node.as_str());
        }
        for edge in &graph.edges {
            if !node_set.contains(edge.from.as_str()) || !node_set.contains(edge.to.as_str()) {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` edge {} -> {} references undeclared nodes",
                        graph.name, edge.from, edge.to
                    ),
                ));
                continue;
            }
            if !call_pairs.contains(&(edge.from.clone(), edge.to.clone())) {
                violations.push(Violation::new(
                    CanonRule::TickGraphEdgesDeclared,
                    format!(
                        "tick graph `{}` edge {} -> {} must exist in call graph",
                        graph.name, edge.from, edge.to
                    ),
                ));
            }
            tg.add_edge(edge.from.as_str(), edge.to.as_str(), ());
        }
        if is_cyclic_directed(&tg) {
            violations.push(Violation::new(
                CanonRule::TickGraphAcyclic,
                format!("tick graph `{}` must be acyclic", graph.name),
            ));
        }
    }
}

fn check_loop_policies(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for p in &ir.loop_policies {
        if idx.tick_graphs.get(p.graph.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::LoopContinuationJudgment,
                format!(
                    "loop policy `{}` references missing tick graph `{}`",
                    p.id, p.graph
                ),
            ));
        }
        if idx.predicates.get(p.continuation.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::LoopContinuationJudgment,
                format!(
                    "loop policy `{}` references missing predicate `{}`",
                    p.id, p.continuation
                ),
            ));
        }
    }
}
