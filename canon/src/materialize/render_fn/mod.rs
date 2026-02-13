mod ast;

use super::render_struct::render_visibility;
use crate::ir::{
    Function, GenericParam, Receiver, RefKind, TypeKind, TypeRef, ValuePort, WhereClause,
};

pub use ast::render_ast_body;

// ── type rendering ───────────────────────────────────────────────────────────

pub fn render_type(ty: &TypeRef) -> String {
    if matches!(ty.kind, TypeKind::Slice) {
        return render_slice_type(ty);
    }
    let inner = render_type_core(ty);
    apply_ref(inner, ty.ref_kind, ty.lifetime.as_deref())
}

fn render_type_core(ty: &TypeRef) -> String {
    match ty.kind {
        TypeKind::Tuple => render_tuple_type(&ty.params),
        TypeKind::FnPtr => render_fn_ptr_type(&ty.params),
        TypeKind::Never => "!".to_owned(),
        TypeKind::Slice => render_slice_type(ty),
        TypeKind::SelfType => "Self".to_owned(),
        TypeKind::ImplTrait => render_impl_dyn_trait("impl", &ty.params),
        TypeKind::DynTrait => render_impl_dyn_trait("dyn", &ty.params),
        _ => render_named_type(ty),
    }
}

fn render_named_type(ty: &TypeRef) -> String {
    if ty.params.is_empty() {
        ty.name.as_str().to_owned()
    } else {
        let args = ty
            .params
            .iter()
            .map(render_type)
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}<{}>", ty.name, args)
    }
}

fn render_tuple_type(params: &[TypeRef]) -> String {
    match params.len() {
        0 => "()".to_owned(),
        1 => format!("({},)", render_type(&params[0])),
        _ => format!(
            "({})",
            params
                .iter()
                .map(render_type)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn render_fn_ptr_type(params: &[TypeRef]) -> String {
    if params.is_empty() {
        return "fn()".to_owned();
    }
    let (inputs, output) = params.split_at(params.len().saturating_sub(1));
    let rendered_inputs = inputs
        .iter()
        .map(render_type)
        .collect::<Vec<_>>()
        .join(", ");
    let mut sig = format!("fn({rendered_inputs})");
    if let Some(ret) = output.last() {
        sig.push_str(" -> ");
        sig.push_str(&render_type(ret));
    }
    sig
}

fn render_slice_type(ty: &TypeRef) -> String {
    let elem = ty
        .params
        .get(0)
        .map(render_type)
        .unwrap_or_else(|| "()".to_owned());
    match ty.ref_kind {
        RefKind::None => format!("[{elem}]"),
        RefKind::Ref => format!("&{}[{elem}]", lifetime_fragment(ty.lifetime.as_deref())),
        RefKind::MutRef => format!("&{}mut [{elem}]", lifetime_fragment(ty.lifetime.as_deref())),
    }
}

fn apply_ref(inner: String, ref_kind: RefKind, lifetime: Option<&str>) -> String {
    match ref_kind {
        RefKind::None => inner,
        RefKind::Ref => format!("&{}{inner}", lifetime_fragment(lifetime)),
        RefKind::MutRef => format!("&{}mut {inner}", lifetime_fragment(lifetime)),
    }
}

fn lifetime_fragment(lifetime: Option<&str>) -> String {
    lifetime
        .and_then(|lt| {
            let trimmed = lt.trim();
            if trimmed.is_empty() {
                None
            } else if trimmed.starts_with('\'') {
                Some(format!("{trimmed} "))
            } else {
                Some(format!("'{} ", trimmed))
            }
        })
        .unwrap_or_default()
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

fn render_impl_dyn_trait(prefix: &str, params: &[TypeRef]) -> String {
    if params.is_empty() {
        return format!("{prefix} _");
    }
    let bounds = params
        .iter()
        .map(render_type)
        .collect::<Vec<_>>()
        .join(" + ");
    format!("{prefix} {bounds}")
}

// ── function signature ───────────────────────────────────────────────────────

pub fn render_fn_signature(inputs: &[ValuePort], outputs: &[ValuePort]) -> String {
    render_fn_signature_with_receiver(Receiver::None, inputs, outputs, &[], &[], &[])
}

pub fn render_fn_signature_with_receiver(
    receiver: Receiver,
    inputs: &[ValuePort],
    outputs: &[ValuePort],
    lifetime_params: &[String],
    generics: &[GenericParam],
    where_clauses: &[WhereClause],
) -> String {
    let generics_block = render_generics(lifetime_params, generics);
    let params = inputs
        .iter()
        .map(|p| format!("{}: {}", p.name, render_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");

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
        _ => Some(format!(
            "({})",
            outputs
                .iter()
                .map(|o| render_type(&o.ty))
                .collect::<Vec<_>>()
                .join(", ")
        )),
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
        parts.extend(generics.iter().map(|param| {
            if param.bounds.is_empty() {
                param.name.to_string()
            } else {
                format!("{}: {}", param.name, param.bounds.join(" + "))
            }
        }));
        format!("<{}>", parts.join(", "))
    }
}

pub(crate) fn render_where_suffix(where_clauses: &[WhereClause]) -> String {
    if where_clauses.is_empty() {
        String::new()
    } else {
        let clauses = where_clauses
            .iter()
            .map(|clause| {
                if clause.bounds.is_empty() {
                    clause.ty.clone()
                } else {
                    format!("{}: {}", clause.ty, clause.bounds.join(" + "))
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("where {clauses}")
    }
}

// ── impl function ─────────────────────────────────────────────────────────────

pub fn render_impl_function(function: &Function) -> String {
    let sig = render_fn_signature_with_receiver(
        function.receiver,
        &function.inputs,
        &function.outputs,
        &function.lifetime_params,
        &function.generics,
        &function.where_clauses,
    );
    let async_kw = if function.is_async { "async " } else { "" };
    let unsafe_kw = if function.is_unsafe { "unsafe " } else { "" };
    let doc_block = function
        .doc
        .as_ref()
        .map(|doc| {
            let rendered = doc
                .lines()
                .map(|line| {
                    if line.trim().is_empty() {
                        "    ///".to_owned()
                    } else {
                        format!("    /// {}", line)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if rendered.is_empty() {
                rendered
            } else {
                rendered + "\n"
            }
        })
        .unwrap_or_default();
    let body = match &function.metadata.ast {
        Some(ast) => render_ast_body(ast, 1),
        None => format!(
            "    // Canon runtime stub\n    canon_runtime::execute_function(\"{}\");\n",
            function.id
        ),
    };
    format!(
        "{}    {}{}{}fn {}{} {{\n{body}    }}",
        doc_block,
        render_visibility(function.visibility),
        async_kw,
        unsafe_kw,
        function.name,
        sig,
    )
}
