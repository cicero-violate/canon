use crate::ir::{
    EnumNode, EnumVariant, EnumVariantFields, Field, Function, FunctionContract, ImplBlock,
    ImplFunctionBinding, Struct, Trait, TraitFunction, Word,
};

use super::super::ast_lower;
use super::super::types::{
    collect_derives, collect_doc_string, convert_fields, convert_generics, convert_inputs,
    convert_receiver, convert_return_type, convert_type, map_visibility, path_to_string, slugify,
    word_from_ident,
};
use super::ImplMapping;
use super::ids::{trait_path_to_trait_fn_id, trait_path_to_trait_id, type_path_to_struct_id};

pub(crate) fn struct_from_syn(module_id: &str, item: &syn::ItemStruct) -> Struct {
    let (kind, fields) = convert_fields(&item.fields);
    Struct {
        id: format!(
            "struct.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name: word_from_ident(&item.ident, "Struct"),
        module: slugify(module_id),
        visibility: map_visibility(&item.vis),
        derives: collect_derives(&item.attrs),
        doc: None,
        kind,
        fields,
        history: Vec::new(),
    }
}

pub(crate) fn enum_from_syn(module_id: &str, item: &syn::ItemEnum) -> EnumNode {
    EnumNode {
        id: format!(
            "enum.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name: word_from_ident(&item.ident, "Enum"),
        module: slugify(module_id),
        visibility: map_visibility(&item.vis),
        variants: item.variants.iter().map(enum_variant_from_syn).collect(),
        history: Vec::new(),
    }
}

pub(crate) fn enum_variant_from_syn(variant: &syn::Variant) -> EnumVariant {
    let fields = match &variant.fields {
        syn::Fields::Unit => EnumVariantFields::Unit,
        syn::Fields::Unnamed(u) => EnumVariantFields::Tuple {
            types: u.unnamed.iter().map(|f| convert_type(&f.ty)).collect(),
        },
        syn::Fields::Named(n) => EnumVariantFields::Struct {
            fields: n
                .named
                .iter()
                .map(|field| Field {
                    name: field
                        .ident
                        .as_ref()
                        .map(|id| word_from_ident(id, "Field"))
                        .unwrap_or_else(|| Word::new("Field").unwrap()),
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: collect_doc_string(&field.attrs),
                })
                .collect(),
        },
    };
    EnumVariant {
        name: word_from_ident(&variant.ident, "Variant"),
        fields,
    }
}

pub(crate) fn trait_from_syn(module_id: &str, item: &syn::ItemTrait) -> Trait {
    let trait_id = format!(
        "trait.{}.{}",
        slugify(module_id),
        slugify(&item.ident.to_string())
    );
    let supertraits = item
        .supertraits
        .iter()
        .filter_map(|bound| match bound {
            syn::TypeParamBound::Trait(tb) => Some(path_to_string(&tb.path)),
            _ => None,
        })
        .collect();
    let functions = item
        .items
        .iter()
        .filter_map(|ti| {
            if let syn::TraitItem::Fn(fn_item) = ti {
                Some(trait_fn_from_syn(&trait_id, fn_item))
            } else {
                None
            }
        })
        .collect();
    Trait {
        id: trait_id,
        name: word_from_ident(&item.ident, "Trait"),
        module: slugify(module_id),
        visibility: map_visibility(&item.vis),
        generic_params: convert_generics(&item.generics),
        functions,
        supertraits,
        associated_types: Vec::new(),
        associated_consts: Vec::new(),
    }
}

pub(crate) fn trait_fn_from_syn(trait_id: &str, item: &syn::TraitItemFn) -> TraitFunction {
    let fn_slug = slugify(&item.sig.ident.to_string());
    TraitFunction {
        id: format!("trait_fn.{}.{}", trait_id, fn_slug),
        name: word_from_ident(&item.sig.ident, "TraitFn"),
        inputs: convert_inputs(&item.sig.inputs),
        outputs: convert_return_type(&item.sig.output),
        default_body: None,
    }
}

pub(crate) fn function_from_syn(
    module_id: &str,
    item: &syn::ItemFn,
    impl_context: Option<(&str, Option<&syn::Path>)>,
    trait_name_to_id: &std::collections::HashMap<String, String>,
) -> Function {
    let (impl_id, trait_function) = match impl_context {
        Some((id, Some(path))) => (
            id.to_owned(),
            Some(trait_path_to_trait_fn_id(
                path,
                module_id,
                &item.sig.ident,
                trait_name_to_id,
            )),
        ),
        Some((id, None)) => (id.to_owned(), None),
        None => (String::new(), None),
    };
    let fn_id = {
        let fn_slug = slugify(&item.sig.ident.to_string());
        let mod_slug = slugify(module_id);
        if impl_id.is_empty() {
            format!("function.{mod_slug}.{fn_slug}")
        } else {
            let struct_slug = impl_id.splitn(4, '.').nth(2).unwrap_or("unknown");
            format!("function.{mod_slug}.{struct_slug}.{fn_slug}")
        }
    };
    Function {
        id: fn_id,
        name: word_from_ident(&item.sig.ident, "Function"),
        module: slugify(module_id),
        impl_id,
        trait_function: trait_function.unwrap_or_default(),
        visibility: map_visibility(&item.vis),
        doc: collect_doc_string(&item.attrs),
        lifetime_params: item
            .sig
            .generics
            .lifetimes()
            .map(|lt| lt.lifetime.to_string())
            .collect(),
        receiver: convert_receiver(&item.sig.inputs),
        is_async: item.sig.asyncness.is_some(),
        is_unsafe: item.sig.unsafety.is_some(),
        generics: convert_generics(&item.sig.generics),
        where_clauses: Vec::new(),
        inputs: convert_inputs(&item.sig.inputs),
        outputs: convert_return_type(&item.sig.output),
        deltas: Vec::new(),
        contract: FunctionContract {
            total: true,
            deterministic: true,
            explicit_inputs: true,
            explicit_outputs: true,
            effects_are_deltas: true,
        },
        metadata: crate::ir::FunctionMetadata {
            ast: ast_lower::lower_block(&item.block),
            ..Default::default()
        },
    }
}

pub(crate) fn function_from_impl_item(
    module_id: &str,
    method: &syn::ImplItemFn,
    context: Option<(&str, Option<&syn::Path>)>,
    trait_name_to_id: &std::collections::HashMap<String, String>,
) -> Function {
    let item_fn = syn::ItemFn {
        attrs: method.attrs.clone(),
        vis: syn::Visibility::Inherited,
        sig: method.sig.clone(),
        block: Box::new(method.block.clone()),
    };
    function_from_syn(module_id, &item_fn, context, trait_name_to_id)
}

pub(crate) fn impl_block_from_syn(
    module_id: &str,
    block: &syn::ItemImpl,
    trait_name_to_id: &std::collections::HashMap<String, String>,
    type_slug_to_id: &std::collections::HashMap<String, String>,
) -> ImplMapping {
    let Some((_, trait_path, _)) = &block.trait_ else {
        return build_standalone(module_id, block, trait_name_to_id, type_slug_to_id);
    };
    let syn::Type::Path(self_path) = block.self_ty.as_ref() else {
        return ImplMapping::Unsupported;
    };
    let struct_id = type_path_to_struct_id(self_path, module_id, type_slug_to_id);
    let trait_id = trait_path_to_trait_id(trait_path, module_id, trait_name_to_id);
    // Skip impl blocks for external traits (std::fmt::Display, Default, From,
    // etc.) that were not ingested as Trait entries. The fallback id produced
    // by trait_path_to_trait_id will not resolve in idx.traits, so emitting
    // the block would only produce ImplBinding violations.
    let trait_name = trait_path
        .segments
        .last()
        .map(|s| s.ident.to_string().to_ascii_lowercase())
        .unwrap_or_default();
    if !trait_name_to_id.contains_key(&trait_name) {
        return ImplMapping::Unsupported;
    }
    let struct_slug = struct_id
        .split('.')
        .last()
        .unwrap_or("unknown");

    let trait_slug = trait_id
        .split('.')
        .last()
        .unwrap_or("unknown");

    // Canonical impl identity must derive from the struct's owning module,
    // not the file/module where the impl block appears.
    let struct_module = module_id.to_string();

    let impl_id = make_impl_id(struct_module, struct_slug, Some(trait_slug));
    let mut bindings = Vec::new();
    let mut functions = Vec::new();
    for item in &block.items {
        if let syn::ImplItem::Fn(method) = item {
            let function = function_from_impl_item(
                module_id,
                method,
                Some((&impl_id, Some(trait_path))),
                trait_name_to_id,
            );
            bindings.push(ImplFunctionBinding {
                trait_fn: trait_path_to_trait_fn_id(
                    trait_path,
                    module_id,
                    &method.sig.ident,
                    trait_name_to_id,
                ),
                function: function.id.clone(),
            });
            functions.push(function);
        }
    }
    if bindings.is_empty() {
        return ImplMapping::Unsupported;
    }
    // Ensure impl block is recorded under the struct's module,
    // not the current file's module.
        let struct_module = struct_id
            .split('.')
            .nth(1)
            .unwrap_or(&slugify(module_id))
            .to_string();

    ImplMapping::ImplBlock(
        ImplBlock {
            id: impl_id,
            module: struct_module,
            struct_id,
            trait_id,
            functions: bindings,
        },
        functions,
    )
}

fn build_standalone(
    module_id: &str,
    block: &syn::ItemImpl,
    trait_name_to_id: &std::collections::HashMap<String, String>,
    type_slug_to_id: &std::collections::HashMap<String, String>,
) -> ImplMapping {
    let self_ty_slug = match block.self_ty.as_ref() {
        syn::Type::Path(p) => slugify(
            &p.path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default(),
        ),
        _ => "unknown".to_owned(),
    };
    // Standalone impl identity must also derive from struct ownership.
    let struct_module = type_slug_to_id
        .get(self_ty_slug.as_str())
        .and_then(|id| id.split('.').nth(1))
        .unwrap_or(module_id);

    let standalone_impl_id = make_impl_id(struct_module, &self_ty_slug, None);
    let struct_id = type_slug_to_id
        .get(self_ty_slug.as_str())
        .cloned()
        .unwrap_or_else(|| format!("struct.{}.{}", slugify(module_id), self_ty_slug));
    let funcs = block
        .items
        .iter()
        .filter_map(|item| {
            if let syn::ImplItem::Fn(method) = item {
                Some(function_from_impl_item(
                    module_id,
                    method,
                    Some((&standalone_impl_id, None)),
                    trait_name_to_id,
                ))
            } else {
                None
            }
        })
        .collect();
    let impl_block = ImplBlock {
        id: standalone_impl_id,
        module: slugify(module_id),
        struct_id,
        trait_id: String::new(),
        functions: Vec::new(),
    };
    ImplMapping::Standalone(impl_block, funcs)
}
// -----------------------------------------------------------------------------
// Canonical Impl ID Constructor
// -----------------------------------------------------------------------------

fn make_impl_id(module_id: &str, struct_slug: &str, trait_slug: Option<&str>) -> String {
    let module_slug = slugify(module_id);

    match trait_slug {
        Some(trait_slug) => {
            format!(
                "impl.{}.{}.{}",
                module_slug,
                struct_slug,
                trait_slug
            )
        }
        None => {
            format!(
                "impl.{}.{}.standalone",
                module_slug,
                struct_slug
            )
        }
    }
}
