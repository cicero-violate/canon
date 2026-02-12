use crate::ir::Trait;
use super::render_fn::render_fn_signature;
use super::render_struct::render_visibility;

pub fn render_trait(trait_def: &Trait) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{}trait {} {{",
        render_visibility(trait_def.visibility),
        trait_def.name
    ));
    let mut functions = trait_def.functions.clone();
    functions.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    if functions.is_empty() {
        lines.push("    // no capabilities".to_owned());
    } else {
        for func in functions {
            lines.push(format!(
                "    fn {}{};",
                func.name,
                render_fn_signature(&func.inputs, &func.outputs)
            ));
        }
    }
    lines.push("}".to_owned());
    lines.join("\n")
}
