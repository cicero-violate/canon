use std::collections::{HashMap, HashSet};
use crate::ir::SystemState;
use crate::validate::{CanonRule, Violation};
use super::DefectClass;
use petgraph::algo::kosaraju_scc;
use petgraph::graphmap::DiGraphMap;
/// Layer 3 — Heuristic Classifiers.
///
/// Each classifier receives the raw violation cluster and the live IR.
/// It queries IR struct fields directly; violation detail strings are only
/// used as a last-resort fallback when no IR signal is available.
pub fn classify(
    rule: CanonRule,
    violations: &[&Violation],
    ir: &SystemState,
) -> (DefectClass, Option<Vec<String>>) {
    match rule {
        CanonRule::ExecutionOnlyInImpl => (classify_rule27(violations, ir), None),
        CanonRule::ImplBinding => (classify_rule26(violations, ir), None),
        CanonRule::CallGraphRespectsDag | CanonRule::ModuleDag => {
            classify_rule43_13(violations, ir)
        }
        CanonRule::VersionEvolution => (DefectClass::ValidatorOverConstraint, None),
        _ => (DefectClass::WrongValue, None),
    }
}
fn classify_rule27(violations: &[&Violation], ir: &SystemState) -> DefectClass {
    let impl_ids_in_ir: HashSet<&str> = ir.impls.iter().map(|b| b.id.as_str()).collect();
    let failing_function_ids: HashSet<&str> = violations
        .iter()
        .filter_map(|v| v.subject_id())
        .collect();
    let failing_functions: Vec<&crate::ir::Function> = ir
        .functions
        .iter()
        .filter(|f| failing_function_ids.contains(f.id.as_str()))
        .collect();
    let has_missing_standalone = failing_functions
        .iter()
        .any(|f| {
            f.impl_id.ends_with(".standalone")
                && !impl_ids_in_ir.contains(f.impl_id.as_str())
        });
    if has_missing_standalone {
        DefectClass::MissingEmit
    } else {
        DefectClass::WrongValue
    }
}
fn classify_rule26(violations: &[&Violation], ir: &SystemState) -> DefectClass {
    let failing_impl_ids: HashSet<&str> = violations
        .iter()
        .filter_map(|v| v.subject_id())
        .collect();
    let failing_impls: Vec<&crate::ir::ImplBlock> = ir
        .impls
        .iter()
        .filter(|b| failing_impl_ids.contains(b.id.as_str()))
        .collect();
    let has_empty_trait_id = failing_impls.iter().any(|b| b.trait_id.is_empty());
    if has_empty_trait_id {
        let trait_related = violations
            .iter()
            .any(|v| {
                let d = v.detail();
                d.contains("ImplMissingTrait") || d.contains("ImplTraitMismatch")
            });
        if trait_related {
            return DefectClass::ValidatorOverConstraint;
        } else {
            return DefectClass::WrongValue;
        }
    }
    let trait_names_in_ir: HashSet<String> = ir
        .traits
        .iter()
        .map(|t| t.name.as_str().to_ascii_lowercase())
        .collect();
    let trait_id_in_ir: HashSet<&str> = ir
        .traits
        .iter()
        .map(|t| t.id.as_str())
        .collect();
    let wrong_module = failing_impls
        .iter()
        .any(|b| {
            if trait_id_in_ir.contains(b.trait_id.as_str()) {
                return false;
            }
            let name = b.trait_id.split('.').last().unwrap_or("").to_ascii_lowercase();
            trait_names_in_ir.contains(&name)
        });
    if wrong_module { DefectClass::WrongValue } else { DefectClass::WrongValue }
}
fn classify_rule43_13(
    violations: &[&Violation],
    ir: &SystemState,
) -> (DefectClass, Option<Vec<String>>) {
    if ir.module_edges.is_empty() {
        return (DefectClass::MissingEmit, None);
    }
    let mut graph: DiGraphMap<&str, ()> = DiGraphMap::new();
    for m in &ir.modules {
        graph.add_node(m.id.as_str());
    }
    for e in &ir.module_edges {
        graph.add_edge(e.source.as_str(), e.target.as_str(), ());
    }
    let sccs = kosaraju_scc(&graph);
    if let Some(comp) = sccs.into_iter().find(|c| c.len() > 1) {
        let cycle: Vec<String> = comp.into_iter().map(|s| s.to_string()).collect();
        return (DefectClass::CycleDetected, Some(cycle));
    }
    let failing_function_ids: HashSet<&str> = violations
        .iter()
        .filter_map(|v| v.subject_id())
        .collect();
    let caller_module_ids: HashSet<&str> = ir
        .functions
        .iter()
        .filter(|f| failing_function_ids.contains(f.id.as_str()))
        .map(|f| f.module.as_str())
        .collect();
    let edge_sources: HashSet<&str> = ir
        .module_edges
        .iter()
        .map(|e| e.source.as_str())
        .collect();
    let edge_targets: HashSet<&str> = ir
        .module_edges
        .iter()
        .map(|e| e.target.as_str())
        .collect();
    let inverted = caller_module_ids
        .iter()
        .any(|id| edge_targets.contains(*id) && !edge_sources.contains(*id));
    if inverted {
        (DefectClass::WrongDirection, None)
    } else {
        (DefectClass::MissingContext, None)
    }
}
