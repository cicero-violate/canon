use super::{pipeline::PipelineEntry, predicate::RulePredicate, DefectClass};

/// Layer 4 — Brief Generator.
///
/// Composes the human-readable brief from structured inputs only.
/// No ad-hoc format strings live in the tracers; all templates are here.
pub fn render(predicate: &RulePredicate, pipeline: Option<&PipelineEntry>, defect: &DefectClass, count: usize, examples: &[String], cycle: Option<&[String]>, tick_slice: Option<&str>) -> String {
    match defect {
        DefectClass::MissingEmit => render_missing_emit(predicate, pipeline, count),
        DefectClass::WrongValue => render_wrong_value(predicate, pipeline, count),
        DefectClass::WrongDirection => render_wrong_direction(predicate, pipeline, count),
        DefectClass::MissingContext => render_missing_context(predicate, pipeline, count),
        DefectClass::ValidatorOverConstraint => render_over_constraint(predicate, pipeline, count),
        DefectClass::RequiresDomainInput => render_domain_input(predicate, count),
        DefectClass::CycleDetected => render_cycle(predicate, count, examples, cycle, tick_slice),
    }
}

// ── Templates ─────────────────────────────────────────────────────────────────

fn render_missing_emit(p: &RulePredicate, pipeline: Option<&PipelineEntry>, count: usize) -> String {
    let rule_code = p.rule.code();
    let coll = p.ir_collection;
    let field = p.ir_field;
    match pipeline {
        Some(pe) => format!(
            "Rule {rule_code} fires on {count} item(s) because a node is referenced via \
             `{field}` but never written into `{coll}`. \
             The ingest function `{fn}` in `{file}` does not construct or push this node. \
             Call-site context: {site}. \
             Fix: construct the missing node in `{fn}` and push it into `{coll}`.",
            fn   = pe.ingest_fn,
            file = pe.file,
            site = pe.call_site,
        ),
        None => format!(
            "Rule {rule_code} fires on {count} item(s) because a node is referenced via \
             `{field}` but never written into `{coll}`. \
             No pipeline map entry found for this field; inspect the ingest builder manually.",
        ),
    }
}

fn render_wrong_value(p: &RulePredicate, pipeline: Option<&PipelineEntry>, count: usize) -> String {
    let rule_code = p.rule.code();
    let field = p.ir_field;
    let cond = p.pass_condition;
    match pipeline {
        Some(pe) => format!(
            "Rule {rule_code} fires on {count} item(s) because `{field}` holds a value that \
             does not satisfy: {cond}. \
             The field is written by `{fn}` in `{file}` (call-site: {site}). \
             Fix: correct the id construction logic in `{fn}` so the produced value \
             matches the format used when registering the target artifact.",
            fn   = pe.ingest_fn,
            file = pe.file,
            site = pe.call_site,
        ),
        None => format!(
            "Rule {rule_code} fires on {count} item(s) because `{field}` holds a value that \
             does not satisfy: {cond}. \
             No pipeline map entry found for this field; inspect the ingest builder manually.",
        ),
    }
}

fn render_wrong_direction(p: &RulePredicate, pipeline: Option<&PipelineEntry>, count: usize) -> String {
    let rule_code = p.rule.code();
    let field = p.ir_field;
    match pipeline {
        Some(pe) => format!(
            "Rule {rule_code} fires on {count} item(s) because the edge recorded in `{field}` \
             has source and target swapped. \
             The edge is constructed in `{fn}` in `{file}` (call-site: {site}). \
             Fix: swap `source` and `target` when inserting into the accumulator in `{fn}`.",
            fn   = pe.ingest_fn,
            file = pe.file,
            site = pe.call_site,
        ),
        None => format!(
            "Rule {rule_code} fires on {count} item(s) because `{field}` has source and target \
             swapped. No pipeline map entry found; inspect the ingest builder manually.",
        ),
    }
}

fn render_missing_context(p: &RulePredicate, pipeline: Option<&PipelineEntry>, count: usize) -> String {
    let rule_code = p.rule.code();
    let field = p.ir_field;
    let coll = p.ir_collection;
    match pipeline {
        Some(pe) => format!(
            "Rule {rule_code} fires on {count} item(s) because some `{coll}` entries are \
             present but the specific pairs required by the failing items are absent from \
             `{field}`. \
             The field is populated by `{fn}` in `{file}` (call-site: {site}). \
             The upstream value exists but is not being threaded through to `{fn}`. \
             Fix: verify that the key format used when writing `{field}` matches the key \
             format used when looking it up.",
            fn   = pe.ingest_fn,
            file = pe.file,
            site = pe.call_site,
        ),
        None => format!(
            "Rule {rule_code} fires on {count} item(s). Some `{coll}` entries exist but \
             the required pairs for `{field}` are absent. \
             No pipeline map entry found; inspect the ingest builder manually.",
        ),
    }
}

fn render_over_constraint(p: &RulePredicate, pipeline: Option<&PipelineEntry>, count: usize) -> String {
    let rule_code = p.rule.code();
    let field = p.ir_field;
    let cond = p.pass_condition;
    match pipeline {
        Some(pe) => format!(
            "Rule {rule_code} fires on {count} item(s) because the validator checks `{field}` \
             unconditionally, but the pass condition ({cond}) does not apply to all items \
             in `{coll}`. \
             The validator is located in `{file}` (call-site: {site}). \
             Fix: guard the check in the validator so it only applies when the condition \
             is relevant (e.g. skip the trait_id lookup for standalone inherent impls).",
            coll = p.ir_collection,
            file = pe.file,
            site = pe.call_site,
        ),
        None => format!(
            "Rule {rule_code} fires on {count} item(s). The validator checks `{field}` \
             unconditionally but the pass condition ({cond}) does not apply to all items. \
             No pipeline map entry found; inspect the validator source manually.",
        ),
    }
}

fn render_domain_input(p: &RulePredicate, count: usize) -> String {
    let rule_code = p.rule.code();
    let coll = p.ir_collection;
    let cond = p.pass_condition;
    format!(
        "Rule {rule_code} fires on {count} item(s). The pass condition ({cond}) requires \
         a human-authored artifact in `{coll}`. \
         This cannot be satisfied by the ingest pipeline. \
         A human operator must author the required node and reference it in `{coll}`. \
         No code fix is possible.",
    )
}

fn render_cycle(p: &RulePredicate, count: usize, _examples: &[String], cycle: Option<&[String]>, tick_slice: Option<&str>) -> String {
    let rule_code = p.rule.code();
    let cycle_part = match cycle {
        Some(path) if !path.is_empty() => format!("Cycle path:\n{}\n\n", path.join(" -> ")),
        _ => "Cycle path:\n(no cycle path computed)\n\n".to_owned(),
    };

    let tick_part = match tick_slice {
        Some(s) => {
            let t = s.trim_end();
            if t.is_empty() {
                "tick_executor edges:\n(no tick_executor edges found)".to_owned()
            } else if t.starts_with("tick_executor edges:") {
                t.to_owned()
            } else {
                format!("tick_executor edges:\n{t}")
            }
        }
        None => "tick_executor edges:\n(no tick_executor edges found)".to_owned(),
    };

    format!(
        "Rule {rule_code} fires on {count} item(s) because `ir.module_edges` \
contains a directed cycle among modules.\n\n\
Cycle detail (tick_executor slice):\n{cycle_part}{tick_part}\n\n\
Fix: break the cycle by removing or refactoring at least one \
cross-module dependency edge."
    )
}
