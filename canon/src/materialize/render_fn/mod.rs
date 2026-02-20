mod ast;

use crate::ir::{Function, GenericParam, Receiver, RefKind, TypeKind, TypeRef, ValuePort, WhereClause};
use crate::materialize::render_common::{render_type as render_type_common, render_visibility};

pub use ast::render_ast_body;

// ── type rendering (delegated to render_common) ───────────────────────────────

pub fn render_type(ty: &TypeRef) -> String {
    render_type_common(ty)
}

fn format_lifetime_param(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        "'_".to_owned()
    } else if trimmed.starts_with('\'') {
        trimmed.to_owned()
    } else {
        format!("'{}", trimmed)
    }
}

// ── function signature ───────────────────────────────────────────────────────

pub fn render_fn_signature(inputs: &[ValuePort], outputs: &[ValuePort]) -> String {
    render_fn_signature_with_receiver(Receiver::None, inputs, outputs, &[], &[], &[])
}

pub fn render_fn_signature_with_receiver(
    receiver: Receiver, inputs: &[ValuePort], outputs: &[ValuePort], lifetime_params: &[String], generics: &[GenericParam], where_clauses: &[WhereClause],
) -> String {
    let generics_block = render_generics(lifetime_params, generics);
    let params = inputs.iter().map(|p| format!("{}: {}", p.name, render_type(&p.ty))).collect::<Vec<_>>().join(", ");

    let receiver_token = match receiver {
        Receiver::None => String::new(),
        Receiver::SelfVal => "self".to_owned(),
        Receiver::SelfRef => "&self".to_owned(),
        Receiver::SelfMutRef => "&mut self".to_owned(),
    };

    let full_params = match (receiver_token.is_empty(), params.is_empty()) {
        (true, _) => params,
        (false, true) => receiver_token,
        (false, false) => format!("{receiver_token}, {params}"),
    };

    let mut sig = format!("{generics_block}({full_params})");
    if let Some(ret) = render_output_types(outputs) {
        sig.push_str(" -> ");
        sig.push_str(&ret);
    }
    let where_suffix = render_where_suffix(where_clauses);
    if !where_suffix.is_empty() {
        sig.push(' ');
        sig.push_str(&where_suffix);
    }
    sig
}

fn render_output_types(outputs: &[ValuePort]) -> Option<String> {
    match outputs.len() {
        0 => None,
        1 => Some(render_type(&outputs[0].ty)),
        _ => Some(format!("({})", outputs.iter().map(|o| render_type(&o.ty)).collect::<Vec<_>>().join(", "))),
    }
}

pub(crate) fn render_generics(lifetime_params: &[String], generics: &[GenericParam]) -> String {
    if lifetime_params.is_empty() && generics.is_empty() {
        String::new()
    } else {
        let mut parts = Vec::new();
        for lt in lifetime_params {
            parts.push(format_lifetime_param(lt));
        }
        parts.extend(generics.iter().map(|param| if param.bounds.is_empty() { param.name.to_string() } else { format!("{}: {}", param.name, param.bounds.join(" + ")) }));
        format!("<{}>", parts.join(", "))
    }
}

pub(crate) fn render_where_suffix(where_clauses: &[WhereClause]) -> String {
    if where_clauses.is_empty() {
        String::new()
    } else {
        let clauses =
            where_clauses.iter().map(|clause| if clause.bounds.is_empty() { clause.ty.clone() } else { format!("{}: {}", clause.ty, clause.bounds.join(" + ")) }).collect::<Vec<_>>().join(", ");
        format!("where {clauses}")
    }
}

// ── impl function ─────────────────────────────────────────────────────────────

pub fn render_impl_function(function: &Function) -> String {
    let sig = render_fn_signature_with_receiver(function.receiver, &function.inputs, &function.outputs, &function.lifetime_params, &function.generics, &function.where_clauses);
    let async_kw = if function.is_async { "async " } else { "" };
    let unsafe_kw = if function.is_unsafe { "unsafe " } else { "" };
    let doc_block = function
        .doc
        .as_ref()
        .map(|doc| {
            let rendered = doc.lines().map(|line| if line.trim().is_empty() { "    ///".to_owned() } else { format!("    /// {}", line) }).collect::<Vec<_>>().join("\n");
            if rendered.is_empty() {
                rendered
            } else {
                rendered + "\n"
            }
        })
        .unwrap_or_default();
    let body = match &function.metadata.ast {
        Some(ast) => render_ast_body(ast, 1),
        None => format!("    // Canon runtime stub\n    canon_runtime::execute_function(\"{}\");\n", function.id),
    };
    format!("{}    {}{}{}fn {}{} {{\n{body}    }}", doc_block, render_visibility(function.visibility), async_kw, unsafe_kw, function.name, sig,)
}
