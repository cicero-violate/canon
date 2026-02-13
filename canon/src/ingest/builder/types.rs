use quote::ToTokens;

use crate::ir::{
    Field, GenericParam, Receiver, StructKind, TypeKind, TypeRef, ValuePort, Visibility, Word,
};

// ── String helpers ────────────────────────────────────────────────────────────

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
    Word::new(value)
        .or_else(|_| Word::new(to_pascal_case(value)))
        .unwrap_or_else(|_| Word::new(fallback).unwrap())
}

pub(crate) fn path_to_string(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

pub(crate) fn expr_to_string(expr: &syn::Expr) -> String {
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
        syn::Meta::NameValue(name_value) => Some(format!(
            "{} = {}",
            path_to_string(&name_value.path),
            expr_to_string(&name_value.value)
        )),
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
            let punct: syn::punctuated::Punctuated<syn::Path, syn::Token![,]> =
                syn::punctuated::Punctuated::parse_terminated(input)?;
            Ok(punct)
        }) {
            for path in list {
                derives.push(path_to_string(&path));
            }
        }
    }
    derives
}

// ── Visibility ────────────────────────────────────────────────────────────────

pub(crate) fn map_visibility(vis: &syn::Visibility) -> Visibility {
    match vis {
        syn::Visibility::Public(_) => Visibility::Public,
        syn::Visibility::Restricted(restricted) => {
            if let Some(path) = &restricted.in_token {
                let _ = path;
            }
            if restricted.path.is_ident("crate") {
                Visibility::PubCrate
            } else if restricted.path.is_ident("super") {
                Visibility::PubSuper
            } else {
                Visibility::Private
            }
        }
        _ => Visibility::Private,
    }
}

// ── Type conversion ───────────────────────────────────────────────────────────

pub(crate) fn convert_type(ty: &syn::Type) -> TypeRef {
    match ty {
        syn::Type::Reference(r) => {
            let mut inner = convert_type(&r.elem);
            inner.ref_kind = if r.mutability.is_some() {
                crate::ir::RefKind::MutRef
            } else {
                crate::ir::RefKind::Ref
            };
            inner.lifetime = r.lifetime.as_ref().map(|lt| lt.to_string());
            inner
        }
        syn::Type::Path(path) => path_type(path),
        syn::Type::Tuple(tuple) => TypeRef {
            name: Word::new("Tuple").unwrap(),
            kind: TypeKind::Tuple,
            params: tuple.elems.iter().map(convert_type).collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::Never(_) => TypeRef {
            name: Word::new("Never").unwrap(),
            kind: TypeKind::Never,
            params: Vec::new(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::ImplTrait(impl_trait) => TypeRef {
            name: Word::new("ImplTrait").unwrap(),
            kind: TypeKind::ImplTrait,
            params: impl_trait
                .bounds
                .iter()
                .filter_map(type_from_bound)
                .collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::TraitObject(obj) => TypeRef {
            name: Word::new("DynTrait").unwrap(),
            kind: TypeKind::DynTrait,
            params: obj.bounds.iter().filter_map(type_from_bound).collect(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
        syn::Type::Paren(paren) => convert_type(&paren.elem),
        _ => TypeRef {
            name: Word::new("External").unwrap(),
            kind: TypeKind::External,
            params: Vec::new(),
            ref_kind: crate::ir::RefKind::None,
            lifetime: None,
        },
    }
}

pub(crate) fn path_type(type_path: &syn::TypePath) -> TypeRef {
    let ident = type_path
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Type".to_owned());
    let kind = if ident == "Self" {
        TypeKind::SelfType
    } else {
        TypeKind::External
    };
    let mut params = Vec::new();
    if let Some(segment) = type_path.path.segments.last() {
        if let syn::PathArguments::AngleBracketed(args) = &segment.arguments {
            for arg in &args.args {
                if let syn::GenericArgument::Type(arg_ty) = arg {
                    params.push(convert_type(arg_ty));
                }
            }
        }
    }
    TypeRef {
        name: word_from_string(&ident, "Type"),
        kind,
        params,
        ref_kind: crate::ir::RefKind::None,
        lifetime: None,
    }
}

pub(crate) fn type_from_bound(bound: &syn::TypeParamBound) -> Option<TypeRef> {
    match bound {
        syn::TypeParamBound::Trait(trait_bound) => {
            let path = trait_bound.path.clone();
            Some(TypeRef {
                name: word_from_string(
                    &path
                        .segments
                        .last()
                        .map(|seg| seg.ident.to_string())
                        .unwrap_or_else(|| "Trait".to_owned()),
                    "Trait",
                ),
                kind: TypeKind::External,
                params: Vec::new(),
                ref_kind: crate::ir::RefKind::None,
                lifetime: None,
            })
        }
        _ => None,
    }
}

// ── Field / generics / ports ──────────────────────────────────────────────────

pub(crate) fn convert_fields(fields: &syn::Fields) -> (StructKind, Vec<Field>) {
    match fields {
        syn::Fields::Named(named) => {
            let mut out = Vec::new();
            for field in &named.named {
                let name = field
                    .ident
                    .as_ref()
                    .map(|ident| word_from_ident(ident, "Field"))
                    .unwrap_or_else(|| Word::new("Field").unwrap());
                out.push(Field {
                    name,
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: None,
                });
            }
            (StructKind::Normal, out)
        }
        syn::Fields::Unnamed(unnamed) => {
            let mut out = Vec::new();
            for (idx, field) in unnamed.unnamed.iter().enumerate() {
                let fallback = format!("Field{idx}");
                out.push(Field {
                    name: word_from_string(&fallback, "Field"),
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: None,
                });
            }
            (StructKind::Tuple, out)
        }
        syn::Fields::Unit => (StructKind::Unit, Vec::new()),
    }
}

pub(crate) fn convert_generics(generics: &syn::Generics) -> Vec<GenericParam> {
    let mut params = Vec::new();
    for param in &generics.params {
        if let syn::GenericParam::Type(ty) = param {
            let bounds = ty
                .bounds
                .iter()
                .filter_map(|bound| match bound {
                    syn::TypeParamBound::Trait(trait_bound) => {
                        Some(path_to_string(&trait_bound.path))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            params.push(GenericParam {
                name: word_from_ident(&ty.ident, "Param"),
                bounds,
            });
        }
    }
    params
}

pub(crate) fn convert_inputs(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
) -> Vec<ValuePort> {
    let mut result = Vec::new();
    for (idx, arg) in inputs.iter().enumerate() {
        match arg {
            syn::FnArg::Receiver(_) => continue,
            syn::FnArg::Typed(pat_ty) => {
                let name = match &*pat_ty.pat {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => format!("param{idx}"),
                };
                result.push(ValuePort {
                    name: word_from_string(&name, "Param"),
                    ty: convert_type(&pat_ty.ty),
                });
            }
        }
    }
    result
}

pub(crate) fn convert_return_type(ret: &syn::ReturnType) -> Vec<ValuePort> {
    match ret {
        syn::ReturnType::Default => vec![ValuePort {
            name: word_from_string("Output", "Output"),
            ty: TypeRef {
                name: Word::new("Unit").unwrap(),
                kind: TypeKind::Tuple,
                params: Vec::new(),
                ref_kind: crate::ir::RefKind::None,
                lifetime: None,
            },
        }],
        syn::ReturnType::Type(_, ty) => vec![ValuePort {
            name: word_from_string("Output", "Output"),
            ty: convert_type(ty),
        }],
    }
}

pub(crate) fn convert_receiver(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
) -> Receiver {
    if let Some(syn::FnArg::Receiver(receiver)) = inputs.first() {
        match (receiver.reference.as_ref(), receiver.mutability.is_some()) {
            (None, _) => Receiver::SelfVal,
            (Some(_), false) => Receiver::SelfRef,
            (Some(_), true) => Receiver::SelfMutRef,
        }
    } else {
        Receiver::None
    }
}
use super::edges::UseEntry;

pub(crate) fn module_segments_from_key(key: &str) -> Vec<String> {
    if key.is_empty() || key == "crate" {
        Vec::new()
    } else {
        key.split("::").map(|s| s.to_string()).collect()
    }
}

pub(crate) fn render_use_item(item: &syn::ItemUse) -> String {
    let body = use_tree_to_string(&item.tree);
    if item.leading_colon.is_some() {
        format!("::{body}")
    } else {
        body
    }
}

fn use_tree_to_string(tree: &syn::UseTree) -> String {
    match tree {
        syn::UseTree::Path(path) => {
            let rest = use_tree_to_string(&path.tree);
            if rest.is_empty() {
                path.ident.to_string()
            } else {
                format!("{}::{}", path.ident, rest)
            }
        }
        syn::UseTree::Name(name) => name.ident.to_string(),
        syn::UseTree::Rename(rename) => format!("{} as {}", rename.ident, rename.rename),
        syn::UseTree::Glob(_) => "*".to_owned(),
        syn::UseTree::Group(group) => {
            let parts = group.items.iter().map(use_tree_to_string).collect::<Vec<_>>();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

pub(crate) fn flatten_use_tree(
    prefix: Vec<String>,
    tree: &syn::UseTree,
    leading_colon: bool,
    acc: &mut Vec<UseEntry>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix;
            next.push(path.ident.to_string());
            flatten_use_tree(next, &path.tree, leading_colon, acc);
        }
        syn::UseTree::Name(name) => {
            let mut segments = prefix;
            segments.push(name.ident.to_string());
            acc.push(UseEntry { segments, alias: None, is_glob: false, leading_colon });
        }
        syn::UseTree::Rename(rename) => {
            let mut segments = prefix;
            segments.push(rename.ident.to_string());
            acc.push(UseEntry {
                segments,
                alias: Some(rename.rename.to_string()),
                is_glob: false,
                leading_colon,
            });
        }
        syn::UseTree::Glob(_) => {
            acc.push(UseEntry { segments: prefix, alias: None, is_glob: true, leading_colon });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(prefix.clone(), item, leading_colon, acc);
            }
        }
    }
}

pub(crate) fn resolve_use_entry(entry: &UseEntry, module_key: &str) -> Option<(String, String)> {
    let mut segments = entry.segments.clone();
    let mut base = if entry.leading_colon {
        Vec::new()
    } else {
        module_segments_from_key(module_key)
    };
    if let Some(first) = segments.first() {
        match first.as_str() {
            "crate" => { base.clear(); segments.remove(0); }
            "self" => { base = module_segments_from_key(module_key); segments.remove(0); }
            "super" => {
                base = module_segments_from_key(module_key);
                while let Some(seg) = segments.first() {
                    if seg == "super" {
                        segments.remove(0);
                        if !base.is_empty() { base.pop(); }
                    } else { break; }
                }
            }
            _ => {}
        }
    }
    base.extend(segments);
    if entry.is_glob {
        let module_name = if base.is_empty() { module_key.to_owned() } else { base.join("::") };
        if module_name == module_key { return None; }
        return Some((module_name, "*".to_owned()));
    }
    if base.is_empty() { return None; }
    let item_name = base.pop()?;
    if item_name == "self" { return None; }
    let module_name = if base.is_empty() { "crate".to_owned() } else { base.join("::") };
    if module_name == module_key { return None; }
    let imported = if let Some(alias) = &entry.alias {
        format!("{item_name} as {alias}")
    } else {
        item_name
    };
    Some((module_name, imported))
}
