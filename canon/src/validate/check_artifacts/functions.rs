use super::super::error::Violation;
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{CanonicalIr, Function};
use std::collections::{HashMap, HashSet};

pub fn check_functions(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let trait_fn_to_trait: HashMap<&str, &str> = ir
        .traits
        .iter()
        .flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str())))
        .collect();

    for f in &ir.functions {
        if f.impl_id.is_empty() {
            check_function_generics(f, violations);
            check_function_deltas(f, idx, violations);
        } else {
            let Some(block) = idx.impls.get(f.impl_id.as_str()) else {
                violations.push(Violation::new(
                    CanonRule::ExecutionOnlyInImpl,
                    format!("function `{}` references missing impl `{}`", f.name, f.impl_id),
                ));
                check_function_generics(f, violations);
                check_function_deltas(f, idx, violations);
                continue;
            };
            if f.module != block.module {
                violations.push(Violation::new(
                    CanonRule::ExecutionOnlyInImpl,
                    format!("function `{}` must live in module `{}`", f.name, block.module),
                ));
            }
            match trait_fn_to_trait.get(f.trait_function.as_str()) {
                Some(&tid) if tid == block.trait_id.as_str() => {}
                Some(_) => violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!("function `{}` implements trait fn from wrong trait", f.name),
                )),
                None if !f.trait_function.is_empty() => violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!(
                        "function `{}` references unknown trait function `{}`",
                        f.name, f.trait_function
                    ),
                )),
                None => {}
            }
            if !f.contract.total
                || !f.contract.deterministic
                || !f.contract.explicit_inputs
                || !f.contract.explicit_outputs
                || !f.contract.effects_are_deltas
            {
                violations.push(Violation::new(
                    CanonRule::FunctionContracts,
                    format!(
                        "function `{}` must assert totality, determinism, explicit IO, and delta effects",
                        f.name
                    ),
                ));
            }
            if f.outputs.is_empty() {
                violations.push(Violation::new(
                    CanonRule::FunctionContracts,
                    format!("function `{}` must explicitly enumerate outputs", f.name),
                ));
            }
            check_function_generics(f, violations);
            check_function_deltas(f, idx, violations);
        }
    }
}

fn check_function_generics(f: &Function, violations: &mut Vec<Violation>) {
    let mut seen_generics: HashSet<&str> = HashSet::new();
    for param in &f.generics {
        if !seen_generics.insert(param.name.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!(
                    "function `{}` declares duplicate generic parameter `{}`",
                    f.name, param.name
                ),
            ));
        }
    }
    let mut seen_lifetimes: HashSet<&str> = HashSet::new();
    for lt in &f.lifetime_params {
        if !seen_lifetimes.insert(lt.as_str()) {
            violations.push(Violation::new(
                CanonRule::ExplicitArtifacts,
                format!("function `{}` declares duplicate lifetime parameter `{lt}`", f.name),
            ));
        }
    }
}

fn check_function_deltas(f: &Function, idx: &Indexes, violations: &mut Vec<Violation>) {
    for d in &f.deltas {
        if idx.deltas.get(d.delta.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::EffectsAreDeltas,
                format!("function `{}` references missing delta `{}`", f.name, d.delta),
            ));
        }
    }
}
