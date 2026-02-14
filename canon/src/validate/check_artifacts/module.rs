use super::super::error::Violation;
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{CanonicalIr, ProofScope};

pub fn check_version_proofs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for proof_id in &ir.version_contract.migration_proofs {
        match idx.proofs.get(proof_id.as_str()) {
            Some(p) if p.scope == ProofScope::Law => {}
            Some(_) => violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version migration proof `{proof_id}` must have law scope"),
            )),
            None => violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version migration proof `{proof_id}` was not found"),
            )),
        }
    }
}

pub fn check_module_edges(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    for edge in &ir.module_edges {
        for imported in &edge.imported_types {
            if imported.trim().is_empty() {
                violations.push(Violation::new(
                    CanonRule::ExplicitArtifacts,
                    format!(
                        "module edge `{}` -> `{}` contains an empty imported_types entry",
                        edge.source, edge.target
                    ),
                ));
            }
        }
    }
}
