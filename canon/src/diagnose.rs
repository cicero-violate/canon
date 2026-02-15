use std::collections::HashMap;

use crate::ir::CanonicalIr;
use crate::validate::{CanonRule, ValidationErrors, Violation};

// ── Public API ────────────────────────────────────────────────────────────────

/// A structured brief produced by tracing one cluster of violations back to
/// its root cause in the ingest pipeline.  This is the unit of work sent to
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
            DefectClass::MissingEmit          => write!(f, "MissingEmit"),
            DefectClass::WrongValue           => write!(f, "WrongValue"),
            DefectClass::WrongDirection       => write!(f, "WrongDirection"),
            DefectClass::MissingContext       => write!(f, "MissingContext"),
            DefectClass::ValidatorOverConstraint => write!(f, "ValidatorOverConstraint"),
            DefectClass::RequiresDomainInput  => write!(f, "RequiresDomainInput"),
        }
    }
}

/// Entry point.  Consumes a `ValidationErrors` and the IR that produced them,
/// returns one `RootCauseBrief` per distinct root cause (not per violation).
pub fn trace_root_causes(
    errors: &ValidationErrors,
    ir: &CanonicalIr,
) -> Vec<RootCauseBrief> {
    // Step 1: cluster violations by rule.
    let clusters = cluster_by_rule(errors.violations());

    // Step 2: for each cluster, trace to a root cause brief.
    let mut briefs = Vec::new();
    for (rule, violations) in &clusters {
        let brief = trace_cluster(*rule, violations, ir);
        briefs.push(brief);
    }

    // Step 3: sort by violation count descending so the highest-impact
    // root causes appear first.
    briefs.sort_by(|a, b| b.violation_count.cmp(&a.violation_count));
    briefs
}

// ── Step 1: Cluster ───────────────────────────────────────────────────────────

fn cluster_by_rule<'a>(
    violations: &'a [Violation],
) -> HashMap<CanonRule, Vec<&'a Violation>> {
    let mut map: HashMap<CanonRule, Vec<&'a Violation>> = HashMap::new();
    for v in violations {
        map.entry(v.rule()).or_default().push(v);
    }
    map
}

// ── Step 2: Trace one cluster ─────────────────────────────────────────────────

fn trace_cluster(
    rule: CanonRule,
    violations: &[&Violation],
    ir: &CanonicalIr,
) -> RootCauseBrief {
    let count = violations.len();
    let examples: Vec<String> = violations
        .iter()
        .take(3)
        .map(|v| v.detail().to_owned())
        .collect();

    match rule {
        CanonRule::ExecutionOnlyInImpl => trace_rule27(count, examples, violations, ir),
        CanonRule::ImplBinding         => trace_rule26(count, examples, violations, ir),
        CanonRule::CallGraphRespectsDag | CanonRule::ModuleDag => {
            trace_rule43_13(rule, count, examples, ir)
        }
        CanonRule::VersionEvolution    => trace_rule99(count, examples),
        _                              => trace_unknown(rule, count, examples),
    }
}

// ── Rule 27: Execution may only occur inside impl blocks ──────────────────────
//
// Validator reads: function.impl_id → looks up in idx.impls
// Written by:      build_impls_and_functions / build_standalone in syn_conv.rs
//
// Two sub-causes:
//   A) impl_id is non-empty but no ImplBlock with that id was emitted
//      → DefectClass::MissingEmit
//   B) impl_id is empty for a method that should be in an impl
//      → DefectClass::WrongValue

fn trace_rule27(
    count: usize,
    examples: Vec<String>,
    violations: &[&Violation],
    ir: &CanonicalIr,
) -> RootCauseBrief {
    let rule = CanonRule::ExecutionOnlyInImpl;
    // Collect the missing impl ids referenced in the messages.
    let missing_impl_ids: Vec<&str> = violations
        .iter()
        .filter_map(|v| extract_backtick_pair(v.detail(), "impl `", "`"))
        .collect();

    // Check whether any of those ids are standalone impls.
    let standalone_count = missing_impl_ids
        .iter()
        .filter(|id| id.ends_with(".standalone"))
        .count();

    // Check whether any of those impl ids exist in the IR.
    let impl_ids_in_ir: std::collections::HashSet<&str> =
        ir.impl_blocks.iter().map(|b| b.id.as_str()).collect();
    let truly_missing = missing_impl_ids
        .iter()
        .filter(|id| !impl_ids_in_ir.contains(*id))
        .count();

    let (defect_class, ir_field, fix_site, brief) = if standalone_count > 0 && truly_missing > 0 {
        (
            DefectClass::MissingEmit,
            "ir.impl_blocks (standalone inherent impl nodes)".to_owned(),
            "src/ingest/builder/functions/syn_conv.rs :: build_standalone()\n\
             src/ingest/builder/functions/mod.rs :: build_impls_and_functions() Standalone arm"
                .to_owned(),
            format!(
                "Rule 27 fires on {count} functions whose impl_id ends with `.standalone`. \
                 The ImplBlock node for standalone inherent impls is never emitted into \
                 ir.impl_blocks. build_standalone() in syn_conv.rs returns \
                 ImplMapping::Standalone(funcs) but never constructs an ImplBlock. \
                 The Standalone arm in build_impls_and_functions() discards the block. \
                 Fix: construct and push an ImplBlock in build_standalone(), change \
                 ImplMapping::Standalone to carry (ImplBlock, Vec<Function>), and push \
                 the block in the Standalone arm of build_impls_and_functions()."
            ),
        )
    } else {
        (
            DefectClass::WrongValue,
            "function.impl_id".to_owned(),
            "src/ingest/builder/functions/syn_conv.rs :: function_from_syn()".to_owned(),
            format!(
                "Rule 27 fires on {count} functions with an impl_id that does not resolve \
                 to any registered ImplBlock. The impl_id is set at construction time in \
                 function_from_syn(). Check that the id format used when setting impl_id \
                 matches the id format used when constructing ImplBlock."
            ),
        )
    };

    RootCauseBrief {
        rule,
        violation_count: count,
        defect_class,
        ir_field,
        fix_site,
        examples,
        brief,
    }
}

// ── Rule 26: Impl blocks must bind nouns to verbs lawfully ────────────────────
//
// Validator reads: impl.trait_id → looks up in idx.traits
//                  impl.struct_id → looks up in idx.structs
//                  binding.trait_fn → checks it belongs to the impl's trait
// Written by:      impl_block_from_syn() in syn_conv.rs
//                  trait_path_to_trait_id() in ids.rs
//
// Two sub-causes:
//   A) trait_id is local-module-stamped for an external/cross-module trait
//      → DefectClass::WrongValue  (trait_path_to_trait_id stamps wrong module)
//   B) trait_id is empty string (standalone impl) and validator rejects it
//      → DefectClass::ValidatorOverConstraint

fn trace_rule26(
    count: usize,
    examples: Vec<String>,
    violations: &[&Violation],
    ir: &CanonicalIr,
) -> RootCauseBrief {
    let rule = CanonRule::ImplBinding;
    // Collect trait ids referenced as missing.
    let missing_trait_ids: Vec<&str> = violations
        .iter()
        .filter_map(|v| extract_backtick_pair(v.detail(), "missing trait `", "`"))
        .collect();

    // Check which missing trait names exist in the IR under a different module.
    let trait_names_in_ir: HashMap<String, &str> = ir
        .traits
        .iter()
        .map(|t| (t.name.as_str().to_ascii_lowercase(), t.id.as_str()))
        .collect();

    let wrong_module_count = missing_trait_ids
        .iter()
        .filter(|id| {
            // Extract the trait name from the id: trait.module_foo.traitname
            let name = id.split('.').last().unwrap_or("");
            trait_names_in_ir.contains_key(name)
        })
        .count();

    let empty_trait_count = violations
        .iter()
        .filter(|v| v.detail().contains("missing trait ``"))
        .count();

    let (defect_class, ir_field, fix_site, brief) = if empty_trait_count > 0 {
        (
            DefectClass::ValidatorOverConstraint,
            "impl_block.trait_id (empty for standalone inherent impls)".to_owned(),
            "src/validate/check_artifacts/impls.rs :: check_impls()".to_owned(),
            format!(
                "Rule 26 fires on {empty_trait_count} standalone inherent impl blocks whose \
                 trait_id is empty string. The validator unconditionally requires trait_id \
                 to resolve in idx.traits, but standalone impls legitimately have no trait. \
                 Fix: guard the trait_id lookup with `if !block.trait_id.is_empty()` in \
                 check_impls() in src/validate/check_artifacts/impls.rs."
            ),
        )
    } else if wrong_module_count > 0 {
        (
            DefectClass::WrongValue,
            "impl_block.trait_id (module segment is wrong)".to_owned(),
            "src/ingest/builder/functions/ids.rs :: trait_path_to_trait_id()".to_owned(),
            format!(
                "Rule 26 fires on {count} impl blocks where the trait_id is stamped with \
                 the calling module instead of the module where the trait is declared. \
                 trait_path_to_trait_id() always uses module_id (the impl's module) to \
                 build the trait id, even for traits declared in other modules. \
                 Fix: build a name→id lookup from the registered traits after build_traits() \
                 completes, pass it into build_impls_and_functions(), thread it to \
                 impl_block_from_syn() and trait_path_to_trait_id(), and resolve the \
                 correct id from the lookup before falling back to local-module stamping."
            ),
        )
    } else {
        (
            DefectClass::WrongValue,
            "impl_block.trait_id or impl_block.struct_id".to_owned(),
            "src/ingest/builder/functions/syn_conv.rs :: impl_block_from_syn()".to_owned(),
            format!(
                "Rule 26 fires on {count} impl blocks with unresolvable trait or struct ids. \
                 Inspect the id construction logic in impl_block_from_syn() and verify that \
                 the slugify format matches the format used when registering structs and traits."
            ),
        )
    };

    RootCauseBrief {
        rule,
        violation_count: count,
        defect_class,
        ir_field,
        fix_site,
        examples,
        brief,
    }
}

// ── Rule 43 / Rule 13: Call graph must respect module import permissions ───────
//
// Validator reads: module_edges (source=importer, target=importee)
//                  call_edges (caller module, callee module)
// Written by:      build_module_edges() in modules.rs
//
// Sub-causes:
//   A) edge direction inverted: (importee, importer) instead of (importer, importee)
//      → DefectClass::WrongDirection
//   B) use statements not translated to module edges at all
//      → DefectClass::MissingEmit

fn trace_rule43_13(
    rule: CanonRule,
    count: usize,
    examples: Vec<String>,
    ir: &CanonicalIr,
) -> RootCauseBrief {
    // Heuristic: collect caller modules from violation messages and check
    // whether ANY module edge exists with them as target (inverted) vs source.
    let caller_ids: Vec<&str> = examples
        .iter()
        .filter_map(|e| extract_backtick_pair(e, "module `", "`"))
        .collect();

    let edge_sources: std::collections::HashSet<&str> =
        ir.module_edges.iter().map(|e| e.source.as_str()).collect();
    let edge_targets: std::collections::HashSet<&str> =
        ir.module_edges.iter().map(|e| e.target.as_str()).collect();

    // If callers appear as targets but not sources, the direction is inverted.
    let inverted_count = caller_ids
        .iter()
        .filter(|id| edge_targets.contains(*id) && !edge_sources.contains(*id))
        .count();

    let no_edges_at_all = ir.module_edges.is_empty();

    let (defect_class, ir_field, fix_site, brief) = if no_edges_at_all {
        (
            DefectClass::MissingEmit,
            "ir.module_edges".to_owned(),
            "src/ingest/builder/modules.rs :: build_module_edges()".to_owned(),
            format!(
                "Rules 43/13 fire on {count} cross-module calls with no module edges at all. \
                 build_module_edges() is not emitting any edges. Check that use statements \
                 are being parsed and that resolve_use_entry() is returning results."
            ),
        )
    } else if inverted_count > 0 {
        (
            DefectClass::WrongDirection,
            "module_edge.source / module_edge.target".to_owned(),
            "src/ingest/builder/modules.rs :: build_module_edges() line building acc key"
                .to_owned(),
            format!(
                "Rules 43/13 fire on {count} cross-module calls. Module edges exist but \
                 caller modules appear as edge targets, not sources. The edge key is \
                 constructed as (importee, importer) but should be (importer, importee). \
                 ModuleEdge.source = the module that imports; ModuleEdge.target = the module \
                 being imported from. Fix: swap source_id and target_id when inserting into \
                 the accumulator in build_module_edges()."
            ),
        )
    } else {
        (
            DefectClass::MissingContext,
            "ir.module_edges".to_owned(),
            "src/ingest/builder/modules.rs :: build_module_edges()".to_owned(),
            format!(
                "Rules 43/13 fire on {count} cross-module calls. Some module edges exist \
                 but the specific caller→callee pairs are missing. The use statements for \
                 these modules may not be resolving via resolve_use_entry(). Check that \
                 the module key format used in flatten_use_tree matches the keys in \
                 module_lookup."
            ),
        )
    };

    RootCauseBrief {
        rule,
        violation_count: count,
        defect_class,
        ir_field,
        fix_site,
        examples,
        brief,
    }
}

// ── Rule 99: Version upgrades require explicit migration proofs ───────────────
//
// This requires domain input (a human-authored proof node). Code cannot fix it.

fn trace_rule99(count: usize, examples: Vec<String>) -> RootCauseBrief {
    RootCauseBrief {
        rule: CanonRule::VersionEvolution,
        violation_count: count,
        defect_class: DefectClass::RequiresDomainInput,
        ir_field: "version_contract.migration_proofs".to_owned(),
        fix_site: "canon.ir.json :: version_contract.migration_proofs[]".to_owned(),
        examples,
        brief: "Rule 99 requires at least one law-scoped proof attached to the version \
                contract. This cannot be generated by the ingest pipeline. A human operator \
                must author a Proof node with ProofScope::Law and reference it in \
                version_contract.migration_proofs. No code fix is possible."
            .to_owned(),
    }
}

// ── Unknown rule fallback ─────────────────────────────────────────────────────

fn trace_unknown(rule: CanonRule, count: usize, examples: Vec<String>) -> RootCauseBrief {
    RootCauseBrief {
        rule,
        violation_count: count,
        defect_class: DefectClass::WrongValue,
        ir_field: "unknown".to_owned(),
        fix_site: "unknown".to_owned(),
        examples,
        brief: format!(
            "{count} violations for rule {:?}. No tracer implemented for this rule yet. \
             Read the validator source for this rule, identify which IR field it reads, \
             then trace to the ingest function that writes that field.",
            rule
        ),
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Extract the content between `prefix` and the next `suffix` in `s`.
/// Used to pull artifact ids out of violation detail strings.
fn extract_backtick_pair<'a>(s: &'a str, prefix: &str, suffix: &str) -> Option<&'a str> {
    let start = s.find(prefix)? + prefix.len();
    let end   = s[start..].find(suffix)? + start;
    Some(&s[start..end])
}
