use crate::ir::{Struct, Visibility};
use super::render_fn::render_type;

pub fn render_struct(structure: &Struct) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{}struct {} {{",
        render_visibility(structure.visibility),
        structure.name
    ));
    let mut fields = structure.fields.clone();
    fields.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    if fields.is_empty() {
        lines.push("    // no fields".to_owned());
    } else {
        for field in fields {
            lines.push(format!(
                "    {}{}: {},",
                render_visibility(field.visibility),
                field.name,
                render_type(&field.ty)
            ));
        }
    }
    lines.push("}".to_owned());
    lines.join("\n")
}

pub fn render_visibility(vis: Visibility) -> &'static str {
    match vis {
        Visibility::Public => "pub ",
        Visibility::Private => "",
    }
}
