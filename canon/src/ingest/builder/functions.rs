use std::collections::HashMap;

use crate::ir::{
    EnumNode, EnumVariant, EnumVariantFields, Field, Function, FunctionContract, ImplBlock,
    ImplFunctionBinding, Struct, Trait, TraitFunction, Word,
};
use crate::layout::{LayoutNode};

use super::layout::LayoutAccumulator;
use super::super::parser::ParsedWorkspace;
use super::types::{
    collect_derives, collect_doc_string, convert_fields, convert_generics, convert_inputs,
    convert_receiver, convert_return_type, convert_type, map_visibility, path_to_string, slugify,
    word_from_ident, word_from_string,
};
use super::modules::module_key;
use super::ast_lower;

// ── Top-level builders ────────────────────────────────────────────────────────

pub(crate) fn build_structs(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
    layout: &mut LayoutAccumulator,
) -> Vec<Struct> {
    let mut structs = Vec::new();
    for file in &parsed.files {
        let mk = module_key(file);
        let Some(module_id) = module_lookup.get(&mk) else { continue };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Struct(item_struct) = item {
                let structure = struct_from_syn(module_id, item_struct);
                layout.assign(LayoutNode::Struct(structure.id.clone()), file_id.clone());
                structs.push(structure);
            }
        }
    }
    structs
}

pub(crate) fn build_enums(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
    layout: &mut LayoutAccumulator,
) -> Vec<EnumNode> {
    let mut enums = Vec::new();
    for file in &parsed.files {
        let mk = module_key(file);
        let Some(module_id) = module_lookup.get(&mk) else { continue };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Enum(item_enum) = item {
                let enum_node = enum_from_syn(module_id, item_enum);
                layout.assign(LayoutNode::Enum(enum_node.id.clone()), file_id.clone());
                enums.push(enum_node);
            }
        }
    }
    enums
}

pub(crate) fn build_traits(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
    layout: &mut LayoutAccumulator,
) -> Vec<Trait> {
    let mut traits = Vec::new();
    for file in &parsed.files {
        let mk = module_key(file);
        let Some(module_id) = module_lookup.get(&mk) else { continue };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Trait(trait_item) = item {
                let tr = trait_from_syn(module_id, trait_item);
                layout.assign(LayoutNode::Trait(tr.id.clone()), file_id.clone());
                traits.push(tr);
            }
        }
    }
    traits
}

pub(crate) fn build_impls_and_functions(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    file_lookup: &HashMap<String, String>,
    layout: &mut LayoutAccumulator,
) -> (Vec<ImplBlock>, Vec<Function>) {
    let mut impls = Vec::new();
    let mut functions = Vec::new();
    for file in &parsed.files {
        let mk = module_key(file);
        let Some(module_id) = module_lookup.get(&mk) else { continue };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for syn_item in &file.ast.items {
            match syn_item {
                syn::Item::Fn(item_fn) => {
                    let function = function_from_syn(module_id, item_fn, None);
                    layout.assign(LayoutNode::Function(function.id.clone()), file_id.clone());
                    functions.push(function);
                }
                syn::Item::Impl(impl_block) => {
                    match impl_block_from_syn(module_id, impl_block) {
                        ImplMapping::Standalone(funcs) => {
                            for function in funcs {
                                layout.assign(
                                    LayoutNode::Function(function.id.clone()),
                                    file_id.clone(),
                                );
                                functions.push(function);
                            }
                        }
                        ImplMapping::ImplBlock(block, funcs) => {
                            for function in funcs {
                                layout.assign(
                                    LayoutNode::Function(function.id.clone()),
                                    file_id.clone(),
                                );
                                functions.push(function);
                            }
                            impls.push(block);
                        }
                        ImplMapping::Unsupported => {}
                    }
                }
                _ => {}
            }
        }
    }
    (impls, functions)
}

// ── syn → IR converters ───────────────────────────────────────────────────────

pub(crate) fn function_from_syn(
    module_id: &str,
    item: &syn::ItemFn,
    impl_context: Option<(&str, Option<&syn::Path>)>,
) -> Function {
    let name = word_from_ident(&item.sig.ident, "Function");
    let visibility = map_visibility(&item.vis);
    let inputs = convert_inputs(&item.sig.inputs);
    let outputs = convert_return_type(&item.sig.output);
    let (impl_id, trait_function) = if let Some((impl_id, trait_path)) = impl_context {
        (
            impl_id.to_owned(),
            trait_path.map(|path| trait_path_to_trait_fn_id(path, module_id, &item.sig.ident)),
        )
    } else {
        (String::new(), None)
    };
    // Disambiguate impl method IDs by injecting the struct slug.
    // impl_id: "impl.<module>.<struct>.<trait>" → extract segment 2.
    // Free functions get no prefix; their names are unique within a module.
    let fn_id = {
        let fn_slug = slugify(&item.sig.ident.to_string());
        let mod_slug = slugify(module_id);
        if impl_id.is_empty() {
            format!("function.{mod_slug}.{fn_slug}")
        } else {
            let struct_slug = impl_id
                .splitn(4, '.')
                .nth(2)
                .unwrap_or("unknown");
            format!("function.{mod_slug}.{struct_slug}.{fn_slug}")
        }
    };
    Function {
        id: fn_id,
        name,
        module: module_id.to_owned(),
        impl_id,
        trait_function: trait_function.unwrap_or_default(),
        visibility,
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
        inputs,
        outputs,
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

pub(crate) fn impl_block_from_syn(
    module_id: &str,
    block: &syn::ItemImpl,
) -> ImplMapping {
    let Some((_, trait_path, _)) = &block.trait_ else {
        let mut standalone = Vec::new();
        let self_ty_slug = match block.self_ty.as_ref() {
            syn::Type::Path(p) => slugify(
                &p.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
            ),
            _ => "unknown".to_owned(),
        };
        let standalone_impl_id = format!("impl.{}.{}.standalone", slugify(module_id), self_ty_slug);
        for item in &block.items {
            if let syn::ImplItem::Fn(method) = item {
                standalone.push(function_from_impl_item(
                    module_id,
                    method,
                    Some((&standalone_impl_id, None)),
                ));
            }
        }
        return ImplMapping::Standalone(standalone);
    };
    let syn::Type::Path(self_path) = block.self_ty.as_ref() else {
        return ImplMapping::Unsupported;
    };
    let struct_id = type_path_to_struct_id(self_path, module_id);
    let trait_id = trait_path_to_trait_id(trait_path, module_id);
    let impl_id = format!(
        "impl.{}.{}.{}",
        slugify(module_id),
        slugify(&struct_id),
        slugify(&trait_id)
    );
    let mut bindings = Vec::new();
    let mut functions = Vec::new();
    for item in &block.items {
        if let syn::ImplItem::Fn(method) = item {
            let function = function_from_impl_item(
                module_id,
                method,
                Some((&impl_id, Some(trait_path))),
            );
            let fn_id = function.id.clone();
            let trait_fn_id =
                trait_path_to_trait_fn_id(trait_path, module_id, &method.sig.ident);
            bindings.push(ImplFunctionBinding {
                trait_fn: trait_fn_id,
                function: fn_id,
            });
            functions.push(function);
        }
    }
    if bindings.is_empty() {
        return ImplMapping::Unsupported;
    }
    ImplMapping::ImplBlock(
        ImplBlock {
            id: impl_id,
            module: module_id.to_owned(),
            struct_id,
            trait_id,
            functions: bindings,
        },
        functions,
    )
}

pub(crate) fn function_from_impl_item(
    module_id: &str,
    method: &syn::ImplItemFn,
    context: Option<(&str, Option<&syn::Path>)>,
) -> Function {
    let item_fn = syn::ItemFn {
        attrs: method.attrs.clone(),
        vis: syn::Visibility::Inherited,
        sig: method.sig.clone(),
        block: Box::new(method.block.clone()),
    };
    function_from_syn(module_id, &item_fn, context)
}

pub(crate) fn enum_from_syn(module_id: &str, item: &syn::ItemEnum) -> EnumNode {
    let name = word_from_ident(&item.ident, "Enum");
    let variants = item.variants.iter().map(enum_variant_from_syn).collect();
    EnumNode {
        id: format!(
            "enum.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name,
        module: module_id.to_owned(),
        visibility: map_visibility(&item.vis),
        variants,
        history: Vec::new(),
    }
}

pub(crate) fn enum_variant_from_syn(variant: &syn::Variant) -> EnumVariant {
    let name = word_from_ident(&variant.ident, "Variant");
    let fields = match &variant.fields {
        syn::Fields::Unit => EnumVariantFields::Unit,
        syn::Fields::Unnamed(unnamed) => {
            let types = unnamed
                .unnamed
                .iter()
                .map(|f| convert_type(&f.ty))
                .collect();
            EnumVariantFields::Tuple { types }
        }
        syn::Fields::Named(named) => {
            let mut fields = Vec::new();
            for field in &named.named {
                let field_name = field
                    .ident
                    .as_ref()
                    .map(|ident| word_from_ident(ident, "Field"))
                    .unwrap_or_else(|| Word::new("Field").unwrap());
                fields.push(Field {
                    name: field_name,
                    ty: convert_type(&field.ty),
                    visibility: map_visibility(&field.vis),
                    doc: collect_doc_string(&field.attrs),
                });
            }
            EnumVariantFields::Struct { fields }
        }
    };
    EnumVariant { name, fields }
}

pub(crate) fn struct_from_syn(
    module_id: &str,
    item: &syn::ItemStruct,
) -> Struct {
    let name = word_from_ident(&item.ident, "Struct");
    let visibility = map_visibility(&item.vis);
    let derives = collect_derives(&item.attrs);
    let (kind, fields) = convert_fields(&item.fields);
    Struct {
        id: format!(
            "struct.{}.{}",
            slugify(module_id),
            slugify(&item.ident.to_string())
        ),
        name,
        module: module_id.to_owned(),
        visibility,
        derives,
        doc: None,
        kind,
        fields,
        history: Vec::new(),
    }
}

pub(crate) fn trait_from_syn(
    module_id: &str,
    item: &syn::ItemTrait,
) -> Trait {
    let name = word_from_ident(&item.ident, "Trait");
    let visibility = map_visibility(&item.vis);
    let trait_slug = slugify(&item.ident.to_string());
    let trait_id = format!("trait.{}.{}", slugify(module_id), trait_slug);
    let supertraits = item
        .supertraits
        .iter()
        .filter_map(|bound| match bound {
            syn::TypeParamBound::Trait(trait_bound) => Some(path_to_string(&trait_bound.path)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let functions = item
        .items
        .iter()
        .filter_map(|trait_item| {
            if let syn::TraitItem::Fn(fn_item) = trait_item {
                Some(trait_fn_from_syn(&trait_id, &item.ident, fn_item))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    Trait {
        id: trait_id,
        name,
        module: module_id.to_owned(),
        visibility,
        generic_params: convert_generics(&item.generics),
        functions,
        supertraits,
        associated_types: Vec::new(),
        associated_consts: Vec::new(),
    }
}

pub(crate) fn trait_fn_from_syn(
    trait_id: &str,
    _trait_name: &syn::Ident,
    item: &syn::TraitItemFn,
) -> TraitFunction {
    let fn_name = word_from_ident(&item.sig.ident, "TraitFn");
    let inputs = convert_inputs(&item.sig.inputs);
    let outputs = convert_return_type(&item.sig.output);
    let fn_slug = slugify(&item.sig.ident.to_string());
    TraitFunction {
        id: format!("trait_fn.{}.{}", trait_id, fn_slug),
        name: fn_name,
        inputs,
        outputs,
        default_body: None,
    }
}

// ── ID derivation helpers ─────────────────────────────────────────────────────

pub(crate) fn trait_path_to_trait_fn_id(
    trait_path: &syn::Path,
    module_id: &str,
    fn_ident: &syn::Ident,
) -> String {
    let trait_id = trait_path_to_trait_id(trait_path, module_id);
    format!("trait_fn.{}.{}", trait_id, slugify(&fn_ident.to_string()))
}

pub(crate) fn trait_path_to_trait_id(path: &syn::Path, module_id: &str) -> String {
    let trait_name = path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Trait".to_owned());
    format!("trait.{}.{}", slugify(module_id), slugify(&trait_name))
}

pub(crate) fn type_path_to_struct_id(path: &syn::TypePath, module_id: &str) -> String {
    let struct_name = path
        .path
        .segments
        .last()
        .map(|seg| seg.ident.to_string())
        .unwrap_or_else(|| "Struct".to_owned());
    format!("struct.{}.{}", slugify(module_id), slugify(&struct_name))
}

pub(crate) enum ImplMapping {
    Standalone(Vec<Function>),
    ImplBlock(ImplBlock, Vec<Function>),
    Unsupported,
}
