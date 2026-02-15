use std::collections::{HashMap, HashSet};

use crate::ir::CanonicalIr;
use crate::validate::{CanonRule, Violation};

use super::DefectClass;

/// Layer 3 — Heuristic Classifiers.
///
/// Each classifier receives the raw violation cluster and the live IR.
/// It queries IR struct fields directly; violation detail strings are only
/// used as a last-resort fallback when no IR signal is available.
pub fn classify(
    rule:       CanonRule,
    violations: &[&Violation],
    ir:         &CanonicalIr,
) -> DefectClass {
    match rule {
        CanonRule::ExecutionOnlyInImpl          => classify_rule27(violations, ir),
        CanonRule::ImplBinding                  => classify_rule26(violations, ir),
        CanonRule::CallGraphRespectsDag
        | CanonRule::ModuleDag                  => classify_rule43_13(violations, ir),
        CanonRule::VersionEvolution             => DefectClass::RequiresDomainInput,
        _                                       => DefectClass::WrongValue,
    }
}

// ── Rule 27 ───────────────────────────────────────────────────────────────────
//
// The validator checks that function.impl_id resolves in idx.impls.
//
// Heuristic (IR-based):
//   1. Collect all impl_ids referenced by failing functions.
//   2. If any of those ids end with ".standalone" AND are absent from
//      ir.impl_blocks → the ImplBlock node was never emitted → MissingEmit.
//   3. Otherwise the impl_id value itself is wrong → WrongValue.

fn classify_rule27(violations: &[&Violation], ir: &CanonicalIr) -> DefectClass {
    let impl_ids_in_ir: HashSet<&str> =
        ir.impl_blocks.iter().map(|b| b.id.as_str()).collect();

    // Pull the impl_id directly from the IR functions that are failing,
    // identified by matching the function id mentioned in each violation.
    let failing_function_ids: HashSet<&str> = violations
        .iter()
        .filter_map(|v| v.subject_id())
        .collect();

    let failing_functions: Vec<&crate::ir::Function> = ir
        .functions
        .iter()
        .filter(|f| failing_function_ids.contains(f.id.as_str()))
        .collect();

    // Primary signal: are any of the impl_ids standalone and absent?
    let has_missing_standalone = failing_functions.iter().any(|f| {
        f.impl_id.ends_with(".standalone") && !impl_ids_in_ir.contains(f.impl_id.as_str())
    });

    if has_missing_standalone {
        DefectClass::MissingEmit
    } else {
        DefectClass::WrongValue
    }
}

// ── Rule 26 ───────────────────────────────────────────────────────────────────
//
// The validator checks that impl_block.trait_id resolves in idx.traits.
//
// Heuristic (IR-based):
//   1. Collect impl blocks that have a non-empty trait_id absent from ir.traits.
//   2. If any impl_block.trait_id is empty → standalone impl hit the validator
//      unconditionally → ValidatorOverConstraint.
//   3. If the trait name (last segment of the id) exists in ir.traits under a
//      different module prefix → WrongValue (wrong module stamp).
//   4. Otherwise → WrongValue (general id mismatch).

fn classify_rule26(violations: &[&Violation], ir: &CanonicalIr) -> DefectClass {
    let failing_impl_ids: HashSet<&str> = violations
        .iter()
        .filter_map(|v| v.subject_id())
        .collect();

    let failing_impls: Vec<&crate::ir::ImplBlock> = ir
        .impl_blocks
        .iter()
        .filter(|b| failing_impl_ids.contains(b.id.as_str()))
        .collect();

    // Signal A: any failing impl has an empty trait_id (standalone inherent impl).
    let has_empty_trait_id = failing_impls.iter().any(|b| b.trait_id.is_empty());
    if has_empty_trait_id {
        return DefectClass::ValidatorOverConstraint;
    }

    // Signal B: the trait name portion of the id exists in ir.traits, just
    // under a different module prefix.
    let trait_names_in_ir: HashSet<String> = ir
        .traits
        .iter()
        .map(|t| t.name.as_str().to_ascii_lowercase())
        .collect();

    let trait_id_in_ir: HashSet<&str> =
        ir.traits.iter().map(|t| t.id.as_str()).collect();

    let wrong_module = failing_impls.iter().any(|b| {
        if trait_id_in_ir.contains(b.trait_id.as_str()) {
            return false; // correctly resolved — not a wrong-module case
        }
        let name = b.trait_id.split('.').last().unwrap_or("").to_ascii_lowercase();
        trait_names_in_ir.contains(&name)
    });

    if wrong_module {
        // Trait exists but under different module prefix
        DefectClass::WrongValue
    } else {
        // General id mismatch or trait not present
        DefectClass::WrongValue
    }
}

// ── Rules 43 / 13 ─────────────────────────────────────────────────────────────
//
// The validator checks that module_edges cover every cross-module call.
//
// Heuristic (IR-based):
//   1. No module edges at all → MissingEmit.
//   2. Caller module ids appear as edge *targets* but not *sources*
//      → direction is inverted → WrongDirection.
//   3. Some edges exist but specific pairs are missing → MissingContext.

fn classify_rule43_13(violations: &[&Violation], ir: &CanonicalIr) -> DefectClass {
    if ir.module_edges.is_empty() {
        return DefectClass::MissingEmit;
    }

    // Collect caller module ids from the IR functions named in violations.
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

    let edge_sources: HashSet<&str> =
        ir.module_edges.iter().map(|e| e.source.as_str()).collect();
    let edge_targets: HashSet<&str> =
        ir.module_edges.iter().map(|e| e.target.as_str()).collect();

    // If caller modules appear as targets but not as sources, direction is inverted.
    let inverted = caller_module_ids
        .iter()
        .any(|id| edge_targets.contains(*id) && !edge_sources.contains(*id));

    if inverted {
        DefectClass::WrongDirection
    } else {
        DefectClass::MissingContext
    }
}
