use crate::CanonicalIr;
use std::collections::BTreeSet;

pub fn diff_ir(before: &CanonicalIr, after: &CanonicalIr) -> String {
    let mut lines = Vec::new();
    diff_section(
        "module",
        before.modules.iter().map(|m| m.id.clone()),
        after.modules.iter().map(|m| m.id.clone()),
        &mut lines,
    );
    diff_section(
        "struct",
        before.structs.iter().map(|s| s.id.clone()),
        after.structs.iter().map(|s| s.id.clone()),
        &mut lines,
    );
    lines.join("\n")
}

fn diff_section(
    label: &str,
    before: impl Iterator<Item = String>,
    after: impl Iterator<Item = String>,
    lines: &mut Vec<String>,
) {
    let before_set: BTreeSet<_> = before.collect();
    let after_set: BTreeSet<_> = after.collect();
    for item in before_set.difference(&after_set) {
        lines.push(format!("- {:<8} {}", label, item));
    }
    for item in after_set.difference(&before_set) {
        lines.push(format!("+ {:<8} {}", label, item));
    }
}
