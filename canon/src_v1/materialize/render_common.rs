use crate::ir::{RefKind, TypeKind, TypeRef, Visibility};

pub fn render_visibility(vis: Visibility) -> &'static str {
    match vis {
        Visibility::Public => "pub ",
        Visibility::Private => "",
        Visibility::PubCrate => "pub(crate) ",
        Visibility::PubSuper => "pub(super) ",
    }
}

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
    match params.split_last() {
        None => "fn()".to_owned(),
        Some((ret, inputs)) => {
            let rendered_inputs = inputs
                .iter()
                .map(render_type)
                .collect::<Vec<_>>()
                .join(", ");
            let mut sig = format!("fn({rendered_inputs})");
            if !matches!(ret.kind, TypeKind::Tuple) || !ret.params.is_empty() {
                sig.push_str(" -> ");
                sig.push_str(&render_type(ret));
            }
            sig
        }
    }
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
