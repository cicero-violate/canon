use std::collections::{HashMap, HashSet};
use crate::ir::*;
use super::error::Violation;
use super::helpers::Indexes;
use super::rules::CanonRule;

pub fn check<'a>(
    ir: &'a CanonicalIr,
    idx: &Indexes<'a>,
    violations: &mut Vec<Violation>,
) {
    check_version_proofs(ir, idx, violations);
    check_structs(ir, idx, violations);
    check_traits(ir, idx, violations);
    check_impls(ir, idx, violations);
    check_functions(ir, idx, violations);
}

fn check_version_proofs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for proof_id in &ir.version_contract.migration_proofs {
        match idx.proofs.get(proof_id.as_str()) {
            Some(p) if p.scope == ProofScope::Law => {}
            Some(_) => violations.push(Violation::new(CanonRule::VersionEvolution, format!("version migration proof `{proof_id}` must have law scope"))),
            None    => violations.push(Violation::new(CanonRule::VersionEvolution, format!("version migration proof `{proof_id}` was not found"))),
        }
    }
}

fn check_structs(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    for s in &ir.structs {
        if idx.modules.get(s.module.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::ExplicitArtifacts, format!("struct `{}` references unknown module `{}`", s.name, s.module)));
        }
    }
}

fn check_traits<'a>(ir: &'a CanonicalIr, idx: &Indexes<'a>, violations: &mut Vec<Violation>) {
    let mut trait_fns: HashMap<&str, &str> = HashMap::new();
    for t in &ir.traits {
        if idx.modules.get(t.module.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::ExplicitArtifacts, format!("trait `{}` references unknown module `{}`", t.name, t.module)));
        }
        for f in &t.functions {
            if trait_fns.insert(f.id.as_str(), t.id.as_str()).is_some() {
                violations.push(Violation::new(CanonRule::TraitVerbs, format!("trait function `{}` must be unique across all traits", f.id)));
            }
        }
    }
}

fn check_impls(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let trait_fn_to_trait: HashMap<&str, &str> = ir.traits.iter()
        .flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str())))
        .collect();

    let mut binding_lookup: HashMap<&str, &str> = HashMap::new();
    for block in &ir.impl_blocks {
        let struct_opt = idx.structs.get(block.struct_id.as_str());
        let trait_opt  = idx.traits.get(block.trait_id.as_str());
        if struct_opt.is_none() {
            violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` references missing struct `{}`", block.id, block.struct_id)));
        }
        if trait_opt.is_none() {
            violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` references missing trait `{}`", block.id, block.trait_id)));
        }
        if idx.modules.get(block.module.as_str()).is_none() {
            violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` references unknown module `{}`", block.id, block.module)));
        }
        if let Some(s) = struct_opt {
            if block.module != s.module {
                violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` must live in same module as struct `{}`", block.id, s.name)));
            }
        }
        if let Some(t) = trait_opt {
            if t.module != block.module {
                violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` binding to trait `{}` must share module", block.id, t.name)));
            }
        }
        for binding in &block.functions {
            if binding_lookup.insert(binding.function.as_str(), block.id.as_str()).is_some() {
                violations.push(Violation::new(CanonRule::ExecutionOnlyInImpl, format!("function `{}` bound more than once", binding.function)));
            }
            if trait_fn_to_trait.get(binding.trait_fn.as_str()).copied() != Some(block.trait_id.as_str()) {
                violations.push(Violation::new(CanonRule::ImplBinding, format!("impl `{}` cannot bind trait function `{}` from another trait", block.id, binding.trait_fn)));
            }
        }
    }
}

fn check_functions(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let trait_fn_to_trait: HashMap<&str, &str> = ir.traits.iter()
        .flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str())))
        .collect();

    for f in &ir.functions {
        let Some(block) = idx.impls.get(f.impl_id.as_str()) else {
            violations.push(Violation::new(CanonRule::ExecutionOnlyInImpl, format!("function `{}` references missing impl `{}`", f.name, f.impl_id)));
            continue;
        };
        if f.module != block.module {
            violations.push(Violation::new(CanonRule::ExecutionOnlyInImpl, format!("function `{}` must live in module `{}`", f.name, block.module)));
        }
        match trait_fn_to_trait.get(f.trait_function.as_str()) {
            Some(&tid) if tid == block.trait_id.as_str() => {}
            Some(_) => violations.push(Violation::new(CanonRule::ImplBinding, format!("function `{}` implements trait fn from wrong trait", f.name))),
            None    => violations.push(Violation::new(CanonRule::ImplBinding, format!("function `{}` references unknown trait function `{}`", f.name, f.trait_function))),
        }
        if !f.contract.total || !f.contract.deterministic || !f.contract.explicit_inputs || !f.contract.explicit_outputs || !f.contract.effects_are_deltas {
            violations.push(Violation::new(CanonRule::FunctionContracts, format!("function `{}` must assert totality, determinism, explicit IO, and delta effects", f.name)));
        }
        if f.outputs.is_empty() {
            violations.push(Violation::new(CanonRule::FunctionContracts, format!("function `{}` must explicitly enumerate outputs", f.name)));
        }
        for d in &f.deltas {
            if idx.deltas.get(d.delta.as_str()).is_none() {
                violations.push(Violation::new(CanonRule::EffectsAreDeltas, format!("function `{}` references missing delta `{}`", f.name, d.delta)));
            }
        }
    }
}
