use std::collections::HashMap;
use crate::ir::SystemState;
use crate::validate::{CanonRule, ValidationErrors, Violation};
mod brief;
mod classify;
mod pipeline;
mod predicate;
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
    /// Optional structural cycle information (for DAG violations).
    pub cycle: Option<Vec<String>>,
    /// Structured diagnostic payload (rule-specific).
    pub structured_report: Option<StructuredReport>,
}
/// Rule-specific structured diagnostic payload.
#[derive(Debug, Clone)]
pub enum StructuredReport {
    Rule26(Rule26Report),
}
impl std::fmt::Display for StructuredReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StructuredReport::Rule26(r) => {
                writeln!(f, "Structured Report (Rule 26: ImplBinding)")?;
                writeln!(f, "  impl_id:  {}", r.impl_id)?;
                writeln!(f, "  trait_id: {}", r.trait_id)?;
                writeln!(f, "  struct_id: {}", r.struct_id)?;
                writeln!(f, "  trait_resolves: {}", r.trait_resolves)?;
                writeln!(f, "  struct_resolves: {}", r.struct_resolves)?;
                writeln!(
                    f, "  trait_name_exists_elsewhere: {}", r.trait_name_exists_elsewhere
                )?;
                Ok(())
            }
        }
    }
}
/// Detailed diagnostic report for Rule 26 (ImplBinding).
#[derive(Debug, Clone)]
pub struct Rule26Report {
    pub impl_id: String,
    pub trait_id: String,
    pub struct_id: String,
    pub trait_resolves: bool,
    pub struct_resolves: bool,
    pub trait_name_exists_elsewhere: bool,
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
    /// The module graph contains a directed cycle.
    CycleDetected,
}
impl std::fmt::Display for DefectClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefectClass::MissingEmit => write!(f, "MissingEmit"),
            DefectClass::WrongValue => write!(f, "WrongValue"),
            DefectClass::WrongDirection => write!(f, "WrongDirection"),
            DefectClass::MissingContext => write!(f, "MissingContext"),
            DefectClass::ValidatorOverConstraint => write!(f, "ValidatorOverConstraint"),
            DefectClass::RequiresDomainInput => write!(f, "RequiresDomainInput"),
            DefectClass::CycleDetected => write!(f, "CycleDetected"),
        }
    }
}
/// Consumes a `ValidationErrors` and the IR that produced them.
/// Returns one `RootCauseBrief` per distinct root cause (not per violation).
/// Results are sorted by violation count descending.
pub fn trace_root_causes(
    errors: &ValidationErrors,
    ir: &SystemState,
) -> Vec<RootCauseBrief> {
    for v in errors.violations() {
        if v.rule() == CanonRule::ExplicitArtifacts {
            println!("EXPLICIT ARTIFACT VIOLATION → {}", v.detail());
        }
    }
    let clusters = cluster_by_rule(errors.violations());
    let mut briefs: Vec<RootCauseBrief> = clusters
        .iter()
        .map(|(rule, violations)| trace_cluster(*rule, violations, ir))
        .collect();
    briefs.sort_by(|a, b| b.violation_count.cmp(&a.violation_count));
    briefs
}
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
    rule: CanonRule,
    violations: &[&Violation],
    ir: &SystemState,
) -> RootCauseBrief {
    let count = violations.len();
    let examples: Vec<String> = violations
        .iter()
        .take(3)
        .map(|v| format!("{:#?}", v))
        .collect();
    let pred = predicate::lookup(rule);
    let pipe = pred.as_ref().and_then(|p| pipeline::entry_for(p.ir_field));
    let (defect, _) = classify::classify(rule, violations, ir);
    let structured_report = match rule {
        CanonRule::ImplBinding => {
            let failing_impl_id = violations
                .iter()
                .filter_map(|v| v.subject_id())
                .next();
            if let Some(impl_id) = failing_impl_id {
                if let Some(block) = ir.impls.iter().find(|b| b.id == impl_id) {
                    let trait_resolves = ir
                        .traits
                        .iter()
                        .any(|t| t.id == block.trait_id);
                    let struct_resolves = ir
                        .structs
                        .iter()
                        .any(|s| s.id == block.struct_id);
                    let trait_name_exists_elsewhere = ir
                        .traits
                        .iter()
                        .any(|t| {
                            t
                                .name
                                .as_str()
                                .eq_ignore_ascii_case(
                                    block.trait_id.split('.').last().unwrap_or(""),
                                )
                        });
                    Some(
                        StructuredReport::Rule26(Rule26Report {
                            impl_id: block.id.clone(),
                            trait_id: block.trait_id.clone(),
                            struct_id: block.struct_id.clone(),
                            trait_resolves,
                            struct_resolves,
                            trait_name_exists_elsewhere,
                        }),
                    )
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    };
    let cycle = if defect == DefectClass::CycleDetected {
        compute_module_cycle(ir)
    } else {
        None
    };
    let tick_slice = if defect == DefectClass::CycleDetected {
        Some(render_tick_executor_edges(ir))
    } else {
        None
    };
    let brief_text = brief::render(
        pred.as_ref().unwrap_or(&fallback_predicate(rule)),
        pipe.as_ref(),
        &defect,
        count,
        &examples,
        cycle.as_deref(),
        tick_slice.as_deref(),
    );
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
        cycle,
        structured_report,
    }
}
fn compute_module_cycle(ir: &SystemState) -> Option<Vec<String>> {
    use std::collections::{HashMap, HashSet};
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in &ir.module_edges {
        adj.entry(e.source.as_str()).or_default().push(e.target.as_str());
        adj.entry(e.target.as_str()).or_default();
    }
    let mut visiting: HashSet<&str> = HashSet::new();
    let mut visited: HashSet<&str> = HashSet::new();
    let mut parent: HashMap<&str, &str> = HashMap::new();
    fn dfs<'a>(
        u: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visiting: &mut HashSet<&'a str>,
        visited: &mut HashSet<&'a str>,
        parent: &mut HashMap<&'a str, &'a str>,
    ) -> Option<Vec<String>> {
        visiting.insert(u);
        if let Some(neighbors) = adj.get(u) {
            for &v in neighbors {
                if !visited.contains(v) && !visiting.contains(v) {
                    parent.insert(v, u);
                    if let Some(path) = dfs(v, adj, visiting, visited, parent) {
                        return Some(path);
                    }
                } else if visiting.contains(v) {
                    let mut path = vec![v.to_string()];
                    let mut cur = u;
                    while cur != v {
                        path.push(cur.to_string());
                        if let Some(&p) = parent.get(cur) {
                            cur = p;
                        } else {
                            break;
                        }
                    }
                    path.push(v.to_string());
                    path.reverse();
                    return Some(path);
                }
            }
        }
        visiting.remove(u);
        visited.insert(u);
        None
    }
    for &node in adj.keys() {
        if !visited.contains(node) {
            parent.clear();
            if let Some(path) = dfs(
                node,
                &adj,
                &mut visiting,
                &mut visited,
                &mut parent,
            ) {
                return Some(path);
            }
        }
    }
    None
}
fn render_tick_executor_edges(ir: &SystemState) -> String {
    let mut lines: Vec<String> = Vec::new();
    for e in &ir.module_edges {
        let a = e.source.as_str();
        let b = e.target.as_str();
        if a.contains("tick_executor") || b.contains("tick_executor") {
            let head: Vec<String> = e.imported_types.iter().take(3).cloned().collect();
            lines.push(format!("{a} -> {b} {:?}", head));
        }
    }
    if lines.is_empty() {
        "(no tick_executor edges found)".to_owned()
    } else {
        lines.join("\n")
    }
}
/// Synthetic predicate used only when a rule has no registered entry yet,
/// so that the brief generator always has something to work with.
fn fallback_predicate(rule: CanonRule) -> predicate::RulePredicate {
    predicate::RulePredicate {
        rule,
        ir_collection: "unknown",
        ir_field: "unknown",
        pass_condition: "no predicate registered for this rule",
    }
}
