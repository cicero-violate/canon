use super::render_fn::{render_ast_body, render_fn_signature, render_generics, render_type};
use super::render_struct::render_visibility;
use crate::ir::Trait;

pub fn render_trait(trait_def: &Trait) -> String {
    let mut lines = Vec::new();
    let mut named_generics = Vec::new();
    let mut anonymous_bounds: Vec<String> = Vec::new();
    for param in &trait_def.generic_params {
        if param.name.as_str().is_empty() {
            anonymous_bounds.extend(param.bounds.iter().cloned());
        } else {
            named_generics.push(param.clone());
        }
    }
    let generics = render_generics(&[], &named_generics);
    let mut supertraits_list = trait_def.supertraits.clone();
    if !anonymous_bounds.is_empty() {
        supertraits_list.extend(anonymous_bounds);
    }
    let supertraits = if supertraits_list.is_empty() {
        String::new()
    } else {
        format!(": {}", supertraits_list.join(" + "))
    };
    lines.push(format!(
        "{}trait {}{}{} {{",
        render_visibility(trait_def.visibility),
        trait_def.name,
        generics,
        supertraits
    ));
    for assoc_ty in &trait_def.associated_types {
        lines.push(format!("    type {};", assoc_ty.name));
    }
    for assoc_const in &trait_def.associated_consts {
        lines.push(format!(
            "    const {}: {};",
            assoc_const.name,
            render_type(&assoc_const.ty)
        ));
    }
    let mut functions = trait_def.functions.clone();
    functions.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    if functions.is_empty() {
        lines.push("    // no capabilities".to_owned());
    } else {
        for func in functions {
            let signature = render_fn_signature(&func.inputs, &func.outputs);
            if let Some(body) = &func.default_body {
                lines.push(format!("    fn {}{} {{", func.name, signature));
                let rendered = render_ast_body(body, 2);
                if !rendered.trim().is_empty() {
                    for body_line in rendered.trim_end().lines() {
                        lines.push(body_line.to_owned());
                    }
                }
                lines.push("    }".to_owned());
            } else {
                lines.push(format!("    fn {}{};", func.name, signature));
            }
        }
    }
    lines.push("}".to_owned());
    lines.join("\n")
}
