use super::render_common::render_type;
use crate::ir::{EnumNode, EnumVariantFields, Struct, StructKind, Visibility};

pub fn render_struct(structure: &Struct) -> String {
    let mut lines = Vec::new();
    if let Some(doc) = &structure.doc {
        for doc_line in doc.lines() {
            lines.push(format!("/// {doc_line}"));
        }
    }
    if !structure.derives.is_empty() {
        lines.push(format!("#[derive({})]", structure.derives.join(", ")));
    }
    match structure.kind {
        StructKind::Unit => {
            lines.push(format!("{}struct {};", render_visibility(structure.visibility), structure.name));
        }
        StructKind::Tuple => {
            let tuple_fields = structure.fields.iter().map(|field| format!("{}{}", render_visibility(field.visibility), render_type(&field.ty))).collect::<Vec<_>>().join(", ");
            lines.push(format!("{}struct {}({});", render_visibility(structure.visibility), structure.name, tuple_fields));
        }
        StructKind::Normal => {
            lines.push(format!("{}struct {} {{", render_visibility(structure.visibility), structure.name));
            let mut fields = structure.fields.clone();
            fields.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
            if fields.is_empty() {
                lines.push("    // no fields".to_owned());
            } else {
                for field in fields {
                    if let Some(doc) = &field.doc {
                        for doc_line in doc.lines() {
                            lines.push(format!("    /// {doc_line}"));
                        }
                    }
                    lines.push(format!("    {}{}: {},", render_visibility(field.visibility), field.name, render_type(&field.ty)));
                }
            }
            lines.push("}".to_owned());
        }
    }
    lines.join("\n")
}

pub use super::render_common::render_visibility;

pub fn render_enum(en: &EnumNode) -> String {
    let mut lines = Vec::new();
    lines.push(format!("{}enum {} {{", render_visibility(en.visibility), en.name));
    if en.variants.is_empty() {
        lines.push("    // no variants".to_owned());
    } else {
        for v in &en.variants {
            let body = match &v.fields {
                EnumVariantFields::Unit => format!("    {},", v.name),
                EnumVariantFields::Tuple { types } => {
                    let inner = types.iter().map(render_type).collect::<Vec<_>>().join(", ");
                    format!("    {}({}),", v.name, inner)
                }
                EnumVariantFields::Struct { fields } => {
                    let inner = fields.iter().map(|f| format!("        {}: {}", f.name, render_type(&f.ty))).collect::<Vec<_>>().join(",\n");
                    format!("    {} {{\n{}\n    }},", v.name, inner)
                }
            };
            lines.push(body);
        }
    }
    lines.push("}".to_owned());
    lines.join("\n")
}
