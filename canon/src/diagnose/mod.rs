use std::collections::HashMap;

use crate::ir::CanonicalIr;
use crate::validate::{CanonRule, ValidationErrors, Violation};

mod brief;
mod classify;
mod pipeline;
mod predicate;

// ── Public types ──────────────────────────────────────────────────────────────

/// A structured brief produced by tracing one cluster of violations back to
/// its root cause in the ingest pipeline. This is the unit of work sent to
/// an LLM patch-writer.
#[derive(Debug, Clone)]
pub struct RootCauseBrief {
    /// The canon rule that fired.
    pub rule: CanonRule,
    /// How many violations this single root cause is responsible for.
    pub violation_count: usize,
    /// Which structural defect class this is.
    pub defect_class: DefectClass,
    /// The IR field that the validator reads and finds wrong.
    pub ir_field: String,
    /// The ingest function (file + fn name) that writes that field.
    pub fix_site: String,
    /// Concrete examples drawn from the violation messages (up to 3).
    pub examples: Vec<String>,
    /// A self-contained brief paragraph suitable for an LLM prompt.
    pub brief: String,
}

/// Four classes cover all structural defects in a pipeline-to-validator system.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefectClass {
    /// A node is referenced but never written to the IR collection.
    MissingEmit,
    /// A node exists but a field is computed with the wrong value.
    WrongValue,
    /// An edge exists but source and target are swapped.
    WrongDirection,
    /// The correct value exists in an upstream stage but is not threaded to
    /// the stage that needs it.
    MissingContext,
    /// The validator predicate is stricter than the IR model requires.
    ValidatorOverConstraint,
    /// The violation requires domain-level input that code cannot supply.
    RequiresDomainInput,
}

impl std::fmt::Display for DefectClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefectClass::MissingEmit             => write!(f, "MissingEmit"),
            DefectClass::WrongValue              => write!(f, "WrongValue"),
            DefectClass::WrongDirection          => write!(f, "WrongDirection"),
            DefectClass::MissingContext          => write!(f, "MissingContext"),
            DefectClass::ValidatorOverConstraint => write!(f, "ValidatorOverConstraint"),
            DefectClass::RequiresDomainInput     => write!(f, "RequiresDomainInput"),
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Consumes a `ValidationErrors` and the IR that produced them.
/// Returns one `RootCauseBrief` per distinct root cause (not per violation).
/// Results are sorted by violation count descending.
pub fn trace_root_causes(
    errors: &ValidationErrors,
    ir:     &CanonicalIr,
) -> Vec<RootCauseBrief> {
    let clusters = cluster_by_rule(errors.violations());

    let mut briefs: Vec<RootCauseBrief> = clusters
        .iter()
        .map(|(rule, violations)| trace_cluster(*rule, violations, ir))
        .collect();

    briefs.sort_by(|a, b| b.violation_count.cmp(&a.violation_count));
    briefs
}

// ── Internal ──────────────────────────────────────────────────────────────────

fn cluster_by_rule<'a>(
    violations: &'a [Violation],
) -> HashMap<CanonRule, Vec<&'a Violation>> {
    let mut map: HashMap<CanonRule, Vec<&'a Violation>> = HashMap::new();
    for v in violations {
        map.entry(v.rule()).or_default().push(v);
    }
    map
}

fn trace_cluster(
    rule:       CanonRule,
    violations: &[&Violation],
    ir:         &CanonicalIr,
) -> RootCauseBrief {
    let count    = violations.len();
    let examples: Vec<String> = violations.iter().take(3).map(|v| v.detail().to_owned()).collect();

    // Layer 1: what does the validator check?
    let pred = predicate::lookup(rule);

    // Layer 2: which ingest function writes that field?
    let pipe = pred
        .as_ref()
        .and_then(|p| pipeline::entry_for(p.ir_field));

    // Layer 3: which defect class does the IR evidence point to?
    let defect = classify::classify(rule, violations, ir);

    // Layer 4: compose the brief from structured inputs.
    let brief_text = brief::render(
        pred.as_ref().unwrap_or(&fallback_predicate(rule)),
        pipe.as_ref(),
        &defect,
        count,
        &examples,
    );

    // ir_field and fix_site are derived from the structured layers; fall back
    // to "unknown" only when no predicate or pipeline entry exists.
    let ir_field = pred
        .as_ref()
        .map(|p| p.ir_field.to_owned())
        .unwrap_or_else(|| "unknown".to_owned());

    let fix_site = pipe
        .as_ref()
        .map(|pe| format!("{} :: {}", pe.file, pe.ingest_fn))
        .unwrap_or_else(|| "unknown".to_owned());

    RootCauseBrief {
        rule,
        violation_count: count,
        defect_class: defect,
        ir_field,
        fix_site,
        examples,
        brief: brief_text,
    }
}

/// Synthetic predicate used only when a rule has no registered entry yet,
/// so that the brief generator always has something to work with.
fn fallback_predicate(rule: CanonRule) -> predicate::RulePredicate {
    predicate::RulePredicate {
        rule,
        ir_collection:  "unknown",
        ir_field:       "unknown",
        pass_condition: "no predicate registered for this rule",
    }
}
