use crate::ir::Word;

pub(crate) fn slugify(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "root".to_owned()
    } else {
        out
    }
}

pub(crate) fn to_pascal_case(input: &str) -> String {
    let mut out = String::new();
    let mut capitalize = true;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            if out.is_empty() || capitalize {
                out.push(ch.to_ascii_uppercase());
                capitalize = false;
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        } else {
            capitalize = true;
        }
    }
    if out.is_empty() {
        "Module".to_owned()
    } else {
        out
    }
}

pub(crate) fn word_from_ident(ident: &syn::Ident, fallback: &str) -> Word {
    word_from_string(&ident.to_string(), fallback)
}

pub(crate) fn word_from_string(value: &str, fallback: &str) -> Word {
    Word::new(value).or_else(|_| Word::new(to_pascal_case(value))).unwrap_or_else(|_| Word::new(fallback).unwrap())
}

pub(crate) fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}

pub(crate) fn expr_to_string(expr: &syn::Expr) -> String {
    use quote::ToTokens;
    expr.to_token_stream().to_string()
}

pub(crate) fn attribute_to_string(attr: &syn::Attribute) -> Option<String> {
    match &attr.meta {
        syn::Meta::Path(path) => Some(path_to_string(path)),
        syn::Meta::List(list) => {
            let path = path_to_string(&list.path);
            let tokens = list.tokens.to_string();
            if tokens.is_empty() {
                Some(path)
            } else {
                Some(format!("{path}{tokens}"))
            }
        }
        syn::Meta::NameValue(name_value) => Some(format!("{} = {}", path_to_string(&name_value.path), expr_to_string(&name_value.value))),
    }
}

pub(crate) fn collect_doc_string(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(meta) = &attr.meta {
            if let syn::Expr::Lit(expr_lit) = &meta.value {
                if let syn::Lit::Str(lit) = &expr_lit.lit {
                    docs.push(lit.value());
                }
            }
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

pub(crate) fn collect_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut derives = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("derive") {
            continue;
        }
        if let Ok(list) = attr.parse_args_with(|input: syn::parse::ParseStream| {
            let punct: syn::punctuated::Punctuated<syn::Path, syn::Token![,]> = syn::punctuated::Punctuated::parse_terminated(input)?;
            Ok(punct)
        }) {
            for path in list {
                derives.push(path_to_string(&path));
            }
        }
    }
    derives
}
