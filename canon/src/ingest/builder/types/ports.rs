use super::strings::{word_from_ident, word_from_string};
use super::types::convert_type;
use super::visibility::map_visibility;
use crate::ir::{Field, GenericParam, Receiver, StructKind, TypeKind, TypeRef, ValuePort, Word};

pub(crate) fn convert_fields(fields: &syn::Fields) -> (StructKind, Vec<Field>) {
    match fields {
        syn::Fields::Named(named) => {
            let mut out = Vec::new();
            for field in &named.named {
                let name = field.ident.as_ref().map(|ident| word_from_ident(ident, "Field")).unwrap_or_else(|| Word::new("Field").unwrap());
                out.push(Field { name, ty: convert_type(&field.ty), visibility: map_visibility(&field.vis), doc: None });
            }
            (StructKind::Normal, out)
        }
        syn::Fields::Unnamed(unnamed) => {
            let mut out = Vec::new();
            for (idx, field) in unnamed.unnamed.iter().enumerate() {
                let fallback = format!("Field{idx}");
                out.push(Field { name: word_from_string(&fallback, "Field"), ty: convert_type(&field.ty), visibility: map_visibility(&field.vis), doc: None });
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
                    syn::TypeParamBound::Trait(trait_bound) => Some(super::strings::path_to_string(&trait_bound.path)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            params.push(GenericParam { name: word_from_ident(&ty.ident, "Param"), bounds });
        }
    }
    params
}

pub(crate) fn convert_inputs(inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>) -> Vec<ValuePort> {
    let mut result = Vec::new();
    for (idx, arg) in inputs.iter().enumerate() {
        match arg {
            syn::FnArg::Receiver(_) => continue,
            syn::FnArg::Typed(pat_ty) => {
                let name = match &*pat_ty.pat {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => format!("param{idx}"),
                };
                result.push(ValuePort { name: word_from_string(&name, "Param"), ty: convert_type(&pat_ty.ty) });
            }
        }
    }
    result
}

pub(crate) fn convert_return_type(ret: &syn::ReturnType) -> Vec<ValuePort> {
    match ret {
        syn::ReturnType::Default => vec![ValuePort {
            name: word_from_string("Output", "Output"),
            ty: TypeRef { name: Word::new("Unit").unwrap(), kind: TypeKind::Tuple, params: Vec::new(), ref_kind: crate::ir::RefKind::None, lifetime: None },
        }],
        syn::ReturnType::Type(_, ty) => vec![ValuePort { name: word_from_string("Output", "Output"), ty: convert_type(ty) }],
    }
}

pub(crate) fn convert_receiver(inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>) -> Receiver {
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
