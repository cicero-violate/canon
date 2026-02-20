use super::super::error::{Violation, ViolationDetail};
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::CanonicalIr;
use std::collections::HashMap;

pub fn check_impls(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let enum_ids: std::collections::HashSet<&str> = ir.enums.iter().map(|e| e.id.as_str()).collect();

    let trait_fn_to_trait: HashMap<&str, &str> = ir.traits.iter().flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str()))).collect();

    let mut binding_lookup: HashMap<&str, &str> = HashMap::new();
    for block in &ir.impls {
        let struct_opt = idx.structs.get(block.struct_id.as_str());
        let trait_opt = idx.traits.get(block.trait_id.as_str());
        let struct_id_valid = struct_opt.is_some() || enum_ids.contains(block.struct_id.as_str());
        if !struct_id_valid {
            violations.push(Violation::structured(CanonRule::ImplBinding, block.id.clone(), ViolationDetail::ImplMissingStruct { impl_id: block.id.clone(), struct_id: block.struct_id.clone() }));
        }
        // Only validate trait existence if trait_id is present
        if !block.trait_id.is_empty() && trait_opt.is_none() {
            violations.push(Violation::structured(CanonRule::ImplBinding, block.id.clone(), ViolationDetail::ImplMissingTrait { impl_id: block.id.clone(), trait_id: block.trait_id.clone() }));
        }
        if idx.modules.get(block.module.as_str()).is_none() {
            violations.push(Violation::structured(CanonRule::ImplBinding, block.id.clone(), ViolationDetail::ImplMissingModule { impl_id: block.id.clone(), module: block.module.clone() }));
        }
        if let Some(s) = struct_opt {
            if block.module != s.module {
                // Cross-module inherent impls are valid Rust; only enforce
                // module co-location for trait impls.
                if !block.trait_id.is_empty() {
                    violations.push(Violation::structured(CanonRule::ImplBinding, block.id.clone(), ViolationDetail::ImplWrongModuleForStruct { impl_id: block.id.clone() }));
                }
            }
        }
        // ImplWrongModuleForTrait is intentionally not checked: Rust allows
        // implementing a trait defined in one module for a type in another.
        for binding in &block.functions {
            if binding_lookup.insert(binding.function.as_str(), block.id.as_str()).is_some() {
                violations.push(Violation::structured(CanonRule::ExecutionOnlyInImpl, block.id.clone(), ViolationDetail::ImplFunctionDuplicate { function_id: binding.function.clone() }));
            }
            // Only enforce trait function binding when trait_id exists
            if !block.trait_id.is_empty() {
                if trait_fn_to_trait.get(binding.trait_fn.as_str()).copied() != Some(block.trait_id.as_str()) {
                    violations.push(Violation::structured(
                        CanonRule::ImplBinding,
                        block.id.clone(),
                        ViolationDetail::ImplTraitMismatch { impl_id: block.id.clone(), trait_fn: binding.trait_fn.clone() },
                    ));
                }
            }
        }
    }
}
