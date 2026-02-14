use super::super::error::Violation;
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{CanonicalIr, StructKind};
use std::collections::{HashMap, HashSet};

pub fn check_structs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for s in &ir.structs {
        if idx.modules.get(s.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("struct `{}` references unknown module `{}`", s.name, s.module),
            ));
        }
        if matches!(s.kind, StructKind::Tuple) && s.fields.is_empty() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("tuple struct `{}` must declare at least one field", s.name),
            ));
        }
        for derive in &s.derives {
            if derive.trim().is_empty() {
                violations.push(Violation::new(
                    CanonRule::ExplicitArtifacts,
                    format!("struct `{}` contains an empty derive entry", s.name),
                ));
            }
        }
    }
}

pub fn check_enums(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for e in &ir.enums {
        if idx.modules.get(e.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("enum `{}` references unknown module `{}`", e.name, e.module),
            ));
        }
    }
}

pub fn check_traits<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    let mut trait_fns: HashMap<&str, &str> = HashMap::new();
    let known_traits: HashSet<&str> = ir.traits.iter().map(|t| t.id.as_str()).collect();
    for t in &ir.traits {
        if idx.modules.get(t.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("trait `{}` references unknown module `{}`", t.name, t.module),
            ));
        }
        for bound in &t.supertraits {
            if !known_traits.contains(bound.as_str()) && !bound.contains("::") {
                violations.push(Violation::new(
                    CanonRule::TraitVerbs,
                    format!("trait `{}` references unknown supertrait `{}`", t.name, bound),
                ));
            }
        }
        for f in &t.functions {
            if trait_fns.insert(f.id.as_str(), t.id.as_str()).is_some() {
                violations.push(Violation::new(
                    CanonRule::TraitVerbs,
                    format!("trait function `{}` must be unique across all traits", f.id),
                ));
            }
        }
    }
}
