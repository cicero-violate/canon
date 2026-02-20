use super::super::error::Violation;
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::{SystemState, Function};
use crate::validate::error::ViolationDetail;
use std::collections::{HashMap, HashSet};
pub fn check_functions(
    ir: &SystemState,
    idx: &Indexes,
    violations: &mut Vec<Violation>,
) {
    let trait_fn_to_trait: HashMap<&str, &str> = ir
        .traits
        .iter()
        .flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str())))
        .collect();
    for f in &ir.functions {
        if f.impl_id.is_empty() {
            check_function_generics(f, violations);
            check_function_deltas(f, idx, violations);
            continue;
        }
        if idx.impls.get(f.impl_id.as_str()).is_none() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ExplicitArtifacts,
                        f.id.clone(),
                        ViolationDetail::FunctionMissingImpl {
                            function_id: f.id.clone(),
                            impl_id: f.impl_id.clone(),
                        },
                    ),
                );
        }
        let Some(block) = idx.impls.get(f.impl_id.as_str()) else {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ExecutionOnlyInImpl,
                        f.id.clone(),
                        ViolationDetail::FunctionMissingImpl {
                            function_id: f.id.clone(),
                            impl_id: f.impl_id.clone(),
                        },
                    ),
                );
            check_function_generics(f, violations);
            check_function_deltas(f, idx, violations);
            continue;
        };
        if f.module != block.module {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ExecutionOnlyInImpl,
                        f.id.clone(),
                        ViolationDetail::FunctionWrongModule {
                            function_id: f.id.clone(),
                            module: block.module.clone(),
                        },
                    ),
                );
        }
        match trait_fn_to_trait.get(f.trait_function.as_str()) {
            Some(&tid) if tid == block.trait_id.as_str() => {}
            Some(_) => {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::ImplBinding,
                            f.id.clone(),
                            ViolationDetail::FunctionWrongTraitBinding {
                                function_id: f.id.clone(),
                            },
                        ),
                    )
            }
            None if !f.trait_function.is_empty() => {
                violations
                    .push(
                        Violation::structured(
                            CanonRule::ImplBinding,
                            f.id.clone(),
                            ViolationDetail::FunctionUnknownTraitFunction {
                                function_id: f.id.clone(),
                                trait_fn: f.trait_function.clone(),
                            },
                        ),
                    )
            }
            None => {}
        }
        if !f.contract.total || !f.contract.deterministic || !f.contract.explicit_inputs
            || !f.contract.explicit_outputs || !f.contract.effects_are_deltas
        {
            violations
                .push(
                    Violation::structured(
                        CanonRule::FunctionContracts,
                        f.id.clone(),
                        ViolationDetail::FunctionContractViolation {
                            function_id: f.id.clone(),
                        },
                    ),
                );
        }
        if f.outputs.is_empty() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::FunctionContracts,
                        f.id.clone(),
                        ViolationDetail::FunctionMissingOutputs {
                            function_id: f.id.clone(),
                        },
                    ),
                );
        }
        check_function_generics(f, violations);
        check_function_deltas(f, idx, violations);
    }
}
fn check_function_generics(f: &Function, violations: &mut Vec<Violation>) {
    let mut seen_generics: HashSet<&str> = HashSet::new();
    for param in &f.generics {
        if !seen_generics.insert(param.name.as_str()) {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ExplicitArtifacts,
                        f.id.clone(),
                        ViolationDetail::FunctionDuplicateGeneric {
                            function_id: f.id.clone(),
                            generic: param.name.to_string(),
                        },
                    ),
                );
        }
    }
    let mut seen_lifetimes: HashSet<&str> = HashSet::new();
    for lt in &f.lifetime_params {
        if !seen_lifetimes.insert(lt.as_str()) {
            violations
                .push(
                    Violation::structured(
                        CanonRule::ExplicitArtifacts,
                        f.id.clone(),
                        ViolationDetail::FunctionDuplicateLifetime {
                            function_id: f.id.clone(),
                            lifetime: lt.to_string(),
                        },
                    ),
                );
        }
    }
}
fn check_function_deltas(f: &Function, idx: &Indexes, violations: &mut Vec<Violation>) {
    for d in &f.deltas {
        if idx.deltas.get(d.delta.as_str()).is_none() {
            violations
                .push(
                    Violation::structured(
                        CanonRule::EffectsAreDeltas,
                        f.id.clone(),
                        ViolationDetail::FunctionMissingDelta {
                            function_id: f.id.clone(),
                            delta: d.delta.to_string(),
                        },
                    ),
                );
        }
    }
}
