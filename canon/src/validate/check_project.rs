use super::error::{Violation, ViolationDetail};
use super::rules::CanonRule;
use crate::ir::{CanonicalIr, Language};
use std::collections::HashSet;

pub fn check(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    if ir.project.language != Language::Rust {
        violations.push(Violation::structured(
            CanonRule::ProjectEnvelope,
            "project",
            ViolationDetail::ProjectLanguageInvalid,
        ));
    }
    if ir.project.version.trim().is_empty() {
        violations.push(Violation::structured(
            CanonRule::ProjectEnvelope,
            "project",
            ViolationDetail::ProjectVersionMissing,
        ));
    }
    if ir.version_contract.current != ir.meta.version {
        violations.push(Violation::structured(
            CanonRule::VersionEvolution,
            "ir.version_contract",
            ViolationDetail::VersionMismatch {
                expected: ir.meta.version.clone(),
                found: ir.version_contract.current.clone(),
            },
        ));
    }
    let mut seen = HashSet::new();
    for v in &ir.version_contract.compatible_with {
        if !seen.insert(v) {
            violations.push(Violation::structured(
                CanonRule::VersionEvolution,
                "ir.version_contract",
                ViolationDetail::VersionDuplicateCompatibility { version: v.clone() },
            ));
        }
    }
    // VersionEvolution migration proof requirement disabled.
    let mut dep_names = HashSet::new();
    for dep in &ir.dependencies {
        if dep.version.trim().is_empty() {
            violations.push(Violation::structured(
                CanonRule::ExternalDependencies,
                dep.name.to_string(),
                ViolationDetail::DependencyMissingVersion {
                    dependency: dep.name.to_string(),
                },
            ));
        }
        if !dep_names.insert(dep.name.as_str()) {
            violations.push(Violation::structured(
                CanonRule::ExternalDependencies,
                dep.name.to_string(),
                ViolationDetail::DependencyDuplicate {
                    dependency: dep.name.to_string(),
                },
            ));
        }
    }
}
