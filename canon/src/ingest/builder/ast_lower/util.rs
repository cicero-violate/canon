use serde_json::Value as JsonValue;

pub(crate) fn path_to_str(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

pub(crate) fn type_to_str(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p)      => path_to_str(&p.path),
        syn::Type::Reference(_) => "&_".to_owned(),
        syn::Type::Slice(_)     => "[_]".to_owned(),
        syn::Type::Array(_)     => "[_; N]".to_owned(),
        syn::Type::Tuple(_)     => "()".to_owned(),
        _                       => "_".to_owned(),
    }
}

pub(crate) fn expr_to_call_str(expr: &syn::Expr) -> String {
    match expr {
        syn::Expr::Path(p) => path_to_str(&p.path),
        syn::Expr::Field(f) => {
            let base   = expr_to_call_str(&f.base);
            let member = match &f.member {
                syn::Member::Named(i)   => i.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            format!("{base}.{member}")
        }
        _ => "fn_ptr".to_owned(),
    }
}
