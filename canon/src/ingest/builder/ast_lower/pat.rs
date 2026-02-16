use super::util::path_to_str;

pub(crate) fn pat_to_string(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(i) => i.ident.to_string(),
        syn::Pat::Wild(_) => "_".to_owned(),
        syn::Pat::Tuple(t) => {
            let parts: Vec<_> = t.elems.iter().map(pat_to_string).collect();
            format!("({})", parts.join(", "))
        }
        syn::Pat::TupleStruct(ts) => {
            let name = path_to_str(&ts.path);
            let fields: Vec<_> = ts.elems.iter().map(pat_to_string).collect();
            format!("{}({})", name, fields.join(", "))
        }
        syn::Pat::Struct(s) => {
            let name = path_to_str(&s.path);
            let fields: Vec<_> = s
                .fields
                .iter()
                .map(|f| match &f.member {
                    syn::Member::Named(i) => i.to_string(),
                    syn::Member::Unnamed(i) => i.index.to_string(),
                })
                .collect();
            format!("{} {{ {} }}", name, fields.join(", "))
        }
        syn::Pat::Path(p) => path_to_str(&p.path),
        syn::Pat::Lit(l) => match &l.lit {
            syn::Lit::Int(i) => i.base10_digits().to_owned(),
            syn::Lit::Str(s) => format!("\"{}\"", s.value()),
            syn::Lit::Bool(b) => b.value.to_string(),
            syn::Lit::Char(c) => format!("'{}'", c.value()),
            syn::Lit::Byte(b) => b.value().to_string(),
            _ => "_".to_owned(),
        },
        syn::Pat::Range(_) => "_..=_".to_owned(),
        syn::Pat::Reference(r) => format!("&{}", pat_to_string(&r.pat)),
        syn::Pat::Or(o) => {
            let parts: Vec<_> = o.cases.iter().map(pat_to_string).collect();
            parts.join(" | ")
        }
        syn::Pat::Slice(s) => {
            let parts: Vec<_> = s.elems.iter().map(pat_to_string).collect();
            format!("[{}]", parts.join(", "))
        }
        _ => "_".to_owned(),
    }
}

pub(crate) fn pat_is_mut(pat: &syn::Pat) -> bool {
    matches!(pat, syn::Pat::Ident(i) if i.mutability.is_some())
}
