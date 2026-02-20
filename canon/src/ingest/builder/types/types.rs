use super::strings::{path_to_string, word_from_string};
use crate::ir::{RefKind, TypeKind, TypeRef, Word};

pub(crate) fn convert_type(ty: &syn::Type) -> TypeRef {
    match ty {
        syn::Type::Reference(r) => {
            let mut inner = convert_type(&r.elem);
            inner.ref_kind = if r.mutability.is_some() { RefKind::MutRef } else { RefKind::Ref };
            inner.lifetime = r.lifetime.as_ref().map(|lt| lt.to_string());
            inner
        }
        syn::Type::Path(path) => path_type(path),
        syn::Type::Tuple(tuple) => {
            TypeRef { name: Word::new("Tuple").unwrap(), kind: TypeKind::Tuple, params: tuple.elems.iter().map(convert_type).collect(), ref_kind: RefKind::None, lifetime: None }
        }
        syn::Type::Never(_) => TypeRef { name: Word::new("Never").unwrap(), kind: TypeKind::Never, params: Vec::new(), ref_kind: RefKind::None, lifetime: None },
        syn::Type::ImplTrait(impl_trait) => TypeRef {
            name: Word::new("ImplTrait").unwrap(),
            kind: TypeKind::ImplTrait,
            params: impl_trait.bounds.iter().filter_map(type_from_bound).collect(),
            ref_kind: RefKind::None,
            lifetime: None,
        },
        syn::Type::TraitObject(obj) => {
            TypeRef { name: Word::new("DynTrait").unwrap(), kind: TypeKind::DynTrait, params: obj.bounds.iter().filter_map(type_from_bound).collect(), ref_kind: RefKind::None, lifetime: None }
        }
        syn::Type::Paren(paren) => convert_type(&paren.elem),
        _ => TypeRef { name: Word::new("External").unwrap(), kind: TypeKind::External, params: Vec::new(), ref_kind: RefKind::None, lifetime: None },
    }
}

pub(crate) fn path_type(type_path: &syn::TypePath) -> TypeRef {
    // Preserve the FULL type path (e.g. crate::foo::Bar)
    let full_path = path_to_string(&type_path.path);

    // Last segment still used to detect `Self`
    let last_ident = type_path.path.segments.last().map(|seg| seg.ident.to_string()).unwrap_or_else(|| "Type".to_owned());

    let kind = if last_ident == "Self" { TypeKind::SelfType } else { TypeKind::External };
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
        // Store full path for stable codegen
        name: word_from_string(&full_path, "Type"),
        kind,
        params,
        ref_kind: RefKind::None,
        lifetime: None,
    }
}

pub(crate) fn type_from_bound(bound: &syn::TypeParamBound) -> Option<TypeRef> {
    match bound {
        syn::TypeParamBound::Trait(trait_bound) => {
            let path = trait_bound.path.clone();
            Some(TypeRef {
                name: word_from_string(&path.segments.last().map(|seg| seg.ident.to_string()).unwrap_or_else(|| "Trait".to_owned()), "Trait"),
                kind: TypeKind::External,
                params: Vec::new(),
                ref_kind: RefKind::None,
                lifetime: None,
            })
        }
        _ => None,
    }
}
