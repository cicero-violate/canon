use super::super::error::{Violation, ViolationDetail};
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{SystemState, ProofScope};
pub fn check_version_proofs(
    ir: &SystemState,
    idx: &Indexes,
    violations: &mut Vec<Violation>,
) {
    let _ = (ir, idx, violations);
}
pub fn check_module_edges(ir: &SystemState, violations: &mut Vec<Violation>) {
    for edge in &ir.module_edges {
        for imported in &edge.imported_types {
            if imported.trim().is_empty() {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::ExplicitArtifacts,
                            format!("module_edge:{}->{}", edge.source, edge.target),
                            ViolationDetail::ModuleEdgeEmptyImport {
                                source: edge.source.clone(),
                                target: edge.target.clone(),
                            },
                        ),
                    );
            }
        }
    }
}
