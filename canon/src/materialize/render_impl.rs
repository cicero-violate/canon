use std::collections::HashMap;
use crate::ir::{Function, ImplBlock, Struct, Trait};
use super::render_fn::render_impl_function;

pub fn render_impl(
    block: &ImplBlock,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
) -> String {
    let struct_name = struct_map
        .get(block.struct_id.as_str())
        .map(|s| s.name.as_str())
        .unwrap_or("<UnknownStruct>");
    let trait_name = trait_map
        .get(block.trait_id.as_str())
        .map(|t| t.name.as_str())
        .unwrap_or("<UnknownTrait>");

    let mut lines = Vec::new();
    lines.push(format!("impl {} for {} {{", trait_name, struct_name));

    let mut bindings = block.functions.clone();
    bindings.sort_by(|a, b| a.trait_fn.as_str().cmp(b.trait_fn.as_str()));

    if bindings.is_empty() {
        lines.push("    // no bindings".to_owned());
    } else {
        for binding in &bindings {
            if let Some(function) = function_map.get(binding.function.as_str()) {
                lines.push(render_impl_function(function));
            } else {
                lines.push(format!(
                    "    // missing function `{}` for trait fn `{}`",
                    binding.function, binding.trait_fn
                ));
            }
        }
    }

    lines.push("}".to_owned());
    lines.join("\n\n")
}
