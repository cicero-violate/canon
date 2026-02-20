use crate::validate::CanonRule;

/// Layer 1 — Predicate Registry.
///
/// For each CanonRule, declares *what the validator checks*:
///   - which IR collection it reads
///   - which field on each item it inspects
///   - the pass condition that must hold
///
/// This is invariant: it does not change when the ingest pipeline changes.
#[derive(Debug, Clone)]
pub struct RulePredicate {
    pub rule: CanonRule,
    pub ir_collection: &'static str,
    pub ir_field: &'static str,
    pub pass_condition: &'static str,
}

/// Return the predicate descriptor for a given rule.
/// Returns `None` for rules that have no tracer yet.
pub fn lookup(rule: CanonRule) -> Option<RulePredicate> {
    match rule {
        CanonRule::ExecutionOnlyInImpl => {
            Some(RulePredicate { rule, ir_collection: "ir.impls", ir_field: "function.impl_id", pass_condition: "function.impl_id resolves to a registered ImplBlock in idx.impls" })
        }
        CanonRule::ImplBinding => Some(RulePredicate {
            rule,
            ir_collection: "ir.impls",
            ir_field: "impl_block.trait_id / impl_block.struct_id",
            pass_condition: "trait_id resolves in idx.traits and struct_id resolves in idx.structs; \
                             each binding.trait_fn belongs to the impl's trait",
        }),
        CanonRule::CallGraphRespectsDag | CanonRule::ModuleDag => Some(RulePredicate {
            rule,
            ir_collection: "ir.module_edges",
            ir_field: "module_edge.source / module_edge.target",
            pass_condition: "for every cross-module call edge (caller → callee), a module_edge \
                             (source=caller, target=callee) must exist in ir.module_edges",
        }),
        CanonRule::VersionEvolution => Some(RulePredicate {
            rule,
            ir_collection: "ir.version_contract",
            ir_field: "version_contract.migration_proofs",
            pass_condition: "at least one Proof with ProofScope::Law is referenced in \
                             version_contract.migration_proofs",
        }),
        CanonRule::ExplicitArtifacts => Some(RulePredicate {
            rule,
            ir_collection: "multiple IR collections",
            ir_field: "referenced artifact id",
            pass_condition: "every referenced artifact id resolves uniquely \
                             in its corresponding IR collection",
        }),
        _ => None,
    }
}
