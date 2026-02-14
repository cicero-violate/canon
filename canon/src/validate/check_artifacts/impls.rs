use super::super::error::Violation;
use super::super::helpers::Indexes;
use super::super::rules::CanonRule;
use crate::ir::CanonicalIr;
use std::collections::HashMap;

pub fn check_impls(ir: &CanonicalIr, idx: &Indexes, violations: &mut Vec<Violation>) {
    let trait_fn_to_trait: HashMap<&str, &str> = ir
        .traits
        .iter()
        .flat_map(|t| t.functions.iter().map(move |f| (f.id.as_str(), t.id.as_str())))
        .collect();

    let mut binding_lookup: HashMap<&str, &str> = HashMap::new();
    for block in &ir.impl_blocks {
        let struct_opt = idx.structs.get(block.struct_id.as_str());
        let trait_opt  = idx.traits.get(block.trait_id.as_str());
        if struct_opt.is_none() {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!("impl `{}` references missing struct `{}`", block.id, block.struct_id),
            ));
        }
        if trait_opt.is_none() {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!("impl `{}` references missing trait `{}`", block.id, block.trait_id),
            ));
        }
        if idx.modules.get(block.module.as_str()).is_none() {
            violations.push(Violation::new(
                CanonRule::ImplBinding,
                format!("impl `{}` references unknown module `{}`", block.id, block.module),
            ));
        }
        if let Some(s) = struct_opt {
            if block.module != s.module {
                violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!("impl `{}` must live in same module as struct `{}`", block.id, s.name),
                ));
            }
        }
        if let Some(t) = trait_opt {
            if t.module != block.module {
                violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!(
                        "impl `{}` binding to trait `{}` must share module",
                        block.id, t.name
                    ),
                ));
            }
        }
        for binding in &block.functions {
            if binding_lookup.insert(binding.function.as_str(), block.id.as_str()).is_some() {
                violations.push(Violation::new(
                    CanonRule::ExecutionOnlyInImpl,
                    format!("function `{}` bound more than once", binding.function),
                ));
            }
            if trait_fn_to_trait.get(binding.trait_fn.as_str()).copied()
                != Some(block.trait_id.as_str())
            {
                violations.push(Violation::new(
                    CanonRule::ImplBinding,
                    format!(
                        "impl `{}` cannot bind trait function `{}` from another trait",
                        block.id, binding.trait_fn
                    ),
                ));
            }
        }
    }
}
