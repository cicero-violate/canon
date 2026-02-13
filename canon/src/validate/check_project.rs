use super::error::Violation;
use super::rules::CanonRule;
use crate::ir::{CanonicalIr, Language};
use std::collections::HashSet;

pub fn check(ir: &CanonicalIr, violations: &mut Vec<Violation>) {
    if ir.project.language != Language::Rust {
        violations.push(Violation::new(
            CanonRule::ProjectEnvelope,
            "project language must currently be Rust",
        ));
    }
    if ir.project.version.trim().is_empty() {
        violations.push(Violation::new(
            CanonRule::ProjectEnvelope,
            "project version must be declared",
        ));
    }
    if ir.version_contract.current != ir.meta.version {
        violations.push(Violation::new(
            CanonRule::VersionEvolution,
            format!(
                "version contract `{}` must match meta version `{}`",
                ir.version_contract.current, ir.meta.version
            ),
        ));
    }
    let mut seen = HashSet::new();
    for v in &ir.version_contract.compatible_with {
        if !seen.insert(v) {
            violations.push(Violation::new(
                CanonRule::VersionEvolution,
                format!("version `{v}` listed multiple times in compatibility"),
            ));
        }
    }
    if ir.version_contract.migration_proofs.is_empty() {
        violations.push(Violation::new(
            CanonRule::VersionEvolution,
            "version contract must provide at least one migration proof",
        ));
    }
    let mut dep_names = HashSet::new();
    for dep in &ir.dependencies {
        if dep.version.trim().is_empty() {
            violations.push(Violation::new(
                CanonRule::ExternalDependencies,
                format!("dependency `{}` must provide a version", dep.name),
            ));
        }
        if !dep_names.insert(dep.name.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExternalDependencies,
                format!("dependency `{}` declared multiple times", dep.name),
            ));
        }
    }
}
