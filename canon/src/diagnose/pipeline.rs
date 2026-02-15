/// Layer 2 — Pipeline Map.
///
/// For each IR field named by Layer 1, declares *which ingest function writes
/// it* and at which call-site in the build pipeline.
///
/// This is stable across rule changes: it only changes when the ingest code
/// is restructured.
#[derive(Debug, Clone)]
pub struct PipelineEntry {
    /// The IR field this entry describes (matches `RulePredicate::ir_field`).
    pub ir_field:  &'static str,
    /// Human-readable name of the ingest function that writes the field.
    pub ingest_fn: &'static str,
    /// Source file that contains `ingest_fn`.
    pub file:      &'static str,
    /// The call-site context (the outer function that invokes `ingest_fn`).
    pub call_site: &'static str,
}

/// Return the pipeline entry for the IR field that a predicate names.
/// Returns `None` when no mapping has been registered yet.
pub fn entry_for(ir_field: &str) -> Option<PipelineEntry> {
    // Match on the leading token of the ir_field string so that compound
    // fields like "impl_block.trait_id / impl_block.struct_id" still resolve.
    let key = ir_field
        .split('/')
        .next()
        .map(str::trim)
        .unwrap_or("");

    match key {
        // Rule 27
        "function.impl_id" => Some(PipelineEntry {
            ir_field:  "function.impl_id",
            ingest_fn: "function_from_syn / build_standalone",
            file:      "src/ingest/builder/functions/syn_conv.rs",
            call_site: "build_impls_and_functions() in src/ingest/builder/functions/mod.rs",
        }),
        // Rule 27 sub-cause: the ImplBlock node itself
        "ir.impl_blocks" => Some(PipelineEntry {
            ir_field:  "ir.impl_blocks",
            ingest_fn: "build_standalone",
            file:      "src/ingest/builder/functions/syn_conv.rs",
            call_site: "build_impls_and_functions() Standalone arm in src/ingest/builder/functions/mod.rs",
        }),
        // Rule 26
        "impl_block.trait_id" => Some(PipelineEntry {
            ir_field:  "impl_block.trait_id",
            ingest_fn: "impl_block_from_syn / trait_path_to_trait_id",
            file:      "src/ingest/builder/functions/syn_conv.rs",
            call_site: "build_impls_and_functions() in src/ingest/builder/functions/mod.rs",
        }),
        // Rules 43 / 13
        "module_edge.source" => Some(PipelineEntry {
            ir_field:  "module_edge.source / module_edge.target",
            ingest_fn: "build_module_edges",
            file:      "src/ingest/builder/modules.rs",
            call_site: "build_modules() in src/ingest/builder/modules.rs",
        }),
        // Rule 99
        "version_contract.migration_proofs" => Some(PipelineEntry {
            ir_field:  "version_contract.migration_proofs",
            ingest_fn: "(none — requires human-authored proof node)",
            file:      "canon.ir.json",
            call_site: "version_contract.migration_proofs[]",
        }),
        _ => None,
    }
}
