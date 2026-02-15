use super::super::error::{Violation, ViolationDetail};
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{CanonicalIr, ProofScope};

pub fn check_version_proofs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for proof_id in &ir.version_contract.migration_proofs {
        match idx.proofs.get(proof_id.as_str()) {
            Some(p) if p.scope == ProofScope::Law => {}
            Some(_) => violations.push(Violation::structured(
                CanonRule::VersionEvolution,
                "ir.version_contract",
                ViolationDetail::VersionProofWrongScope {
                    proof: proof_id.clone(),
                },
            )),
            None => violations.push(Violation::structured(
                CanonRule::VersionEvolution,
                "ir.version_contract",
                ViolationDetail::VersionProofMissing {
                    proof: proof_id.clone(),
                },
            )),
        }
    }
}

pub fn check_module_edges(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    for edge in &ir.module_edges {
        for imported in &edge.imported_types {
            if imported.trim().is_empty() {
                violations.push(Violation::structured(
                    CanonRule::ExplicitArtifacts,
                    format!("module_edge:{}->{}", edge.source, edge.target),
                    ViolationDetail::ModuleEdgeEmptyImport {
                        source: edge.source.clone(),
                        target: edge.target.clone(),
                    },
                ));
            }
        }
    }
}
