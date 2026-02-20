use super::super::error::{Violation, ViolationDetail};
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{CanonicalIr, StructKind};
use std::collections::{HashMap, HashSet};

pub fn check_structs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for s in &ir.structs {
        if idx.modules.get(s.module.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::ExplicitArtifacts, s.id.clone(), ViolationDetail::StructMissingModule { struct_id: s.id.clone(), module: s.module.clone() }));
        }
        if matches!(s.kind, StructKind::Tuple) && s.fields.is_empty() {
            violations.push(Violation::structured(CanonRule::ExplicitArtifacts, s.id.clone(), ViolationDetail::TupleStructEmpty { struct_id: s.id.clone() }));
        }
        for derive in &s.derives {
            if derive.trim().is_empty() {
                violations.push(Violation::structured(CanonRule::ExplicitArtifacts, s.id.clone(), ViolationDetail::StructEmptyDerive { struct_id: s.id.clone() }));
            }
        }
    }
}

pub fn check_enums(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for e in &ir.enums {
        if idx.modules.get(e.module.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::ExplicitArtifacts, e.id.clone(), ViolationDetail::EnumMissingModule { enum_id: e.id.clone(), module: e.module.clone() }));
        }
    }
}

pub fn check_traits<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    let mut trait_fns: HashMap<&str, &str> = HashMap::new();
    let known_traits: HashSet<&str> = ir.traits.iter().map(|t| t.id.as_str()).collect();
    for t in &ir.traits {
        if idx.modules.get(t.module.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::ExplicitArtifacts, t.id.clone(), ViolationDetail::TraitMissingModule { trait_id: t.id.clone(), module: t.module.clone() }));
        }
        for bound in &t.supertraits {
            if !known_traits.contains(bound.as_str()) && !bound.contains("::") {
                violations.push(Violation::structured(CanonRule::TraitVerbs, t.id.clone(), ViolationDetail::TraitMissingSupertrait { trait_id: t.id.clone(), supertrait: bound.clone() }));
            }
        }
        for f in &t.functions {
            if trait_fns.insert(f.id.as_str(), t.id.as_str()).is_some() {
                violations.push(Violation::structured(CanonRule::TraitVerbs, f.id.clone(), ViolationDetail::TraitFunctionDuplicate { function_id: f.id.clone() }));
            }
        }
    }
}
