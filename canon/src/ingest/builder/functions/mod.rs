use std::collections::HashMap;

use crate::ir::{EnumNode, Function, ImplBlock, Struct, Trait};
use crate::layout::LayoutNode;

use super::super::parser::ParsedWorkspace;
use super::layout::LayoutAccumulator;
use super::modules::module_key;

mod ids;
mod syn_conv;

pub(crate) use ids::{trait_path_to_trait_fn_id, trait_path_to_trait_id, type_path_to_struct_id};
pub(crate) use syn_conv::{enum_from_syn, enum_variant_from_syn, function_from_impl_item, function_from_syn, impl_block_from_syn, struct_from_syn, trait_fn_from_syn, trait_from_syn};

pub(crate) enum ImplMapping {
    Standalone(ImplBlock, Vec<Function>),
    ImplBlock(ImplBlock, Vec<Function>),
    Unsupported,
}

pub(crate) fn build_structs(parsed: &ParsedWorkspace, module_lookup: &HashMap<String, String>, file_lookup: &HashMap<String, String>, layout: &mut LayoutAccumulator) -> Vec<Struct> {
    let mut structs = Vec::new();
    for file in &parsed.files {
        let Some(module_id) = module_lookup.get(&module_key(file)) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Struct(s) = item {
                let structure = struct_from_syn(module_id, s);
                layout.assign(LayoutNode::Struct(structure.id.clone()), file_id.clone());
                structs.push(structure);
            }
        }
    }
    structs
}

pub(crate) fn build_enums(parsed: &ParsedWorkspace, module_lookup: &HashMap<String, String>, file_lookup: &HashMap<String, String>, layout: &mut LayoutAccumulator) -> Vec<EnumNode> {
    let mut enums = Vec::new();
    for file in &parsed.files {
        let Some(module_id) = module_lookup.get(&module_key(file)) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Enum(e) = item {
                let enum_node = enum_from_syn(module_id, e);
                layout.assign(LayoutNode::Enum(enum_node.id.clone()), file_id.clone());
                enums.push(enum_node);
            }
        }
    }
    enums
}

pub(crate) fn build_traits(parsed: &ParsedWorkspace, module_lookup: &HashMap<String, String>, file_lookup: &HashMap<String, String>, layout: &mut LayoutAccumulator) -> Vec<Trait> {
    let mut traits = Vec::new();
    for file in &parsed.files {
        let Some(module_id) = module_lookup.get(&module_key(file)) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for item in &file.ast.items {
            if let syn::Item::Trait(t) = item {
                let tr = trait_from_syn(module_id, t);
                layout.assign(LayoutNode::Trait(tr.id.clone()), file_id.clone());
                traits.push(tr);
            }
        }
    }
    traits
}

pub(crate) fn build_impls_and_functions(
    parsed: &ParsedWorkspace, module_lookup: &HashMap<String, String>, file_lookup: &HashMap<String, String>, layout: &mut LayoutAccumulator, trait_name_to_id: &HashMap<String, String>,
    type_slug_to_id: &HashMap<String, String>, type_id_to_module: &HashMap<String, String>,
) -> (Vec<ImplBlock>, Vec<Function>) {
    let mut impls = Vec::new();
    let mut functions = Vec::new();
    for file in &parsed.files {
        let Some(module_id) = module_lookup.get(&module_key(file)) else {
            continue;
        };
        let file_id = file_lookup.get(&file.path_string()).cloned();
        for syn_item in &file.ast.items {
            match syn_item {
                syn::Item::Fn(item_fn) => {
                    let function = function_from_syn(module_id, item_fn, None, trait_name_to_id);
                    layout.assign(LayoutNode::Function(function.id.clone()), file_id.clone());
                    functions.push(function);
                }
                syn::Item::Impl(impl_block) => match impl_block_from_syn(module_id, impl_block, trait_name_to_id, type_slug_to_id, type_id_to_module) {
                    ImplMapping::Standalone(block, funcs) => {
                        for f in &funcs {
                            layout.assign(LayoutNode::Function(f.id.clone()), file_id.clone());
                        }
                        functions.extend(funcs);
                        impls.push(block);
                    }
                    ImplMapping::ImplBlock(block, funcs) => {
                        for f in &funcs {
                            layout.assign(LayoutNode::Function(f.id.clone()), file_id.clone());
                        }
                        functions.extend(funcs);
                        impls.push(block);
                    }
                    ImplMapping::Unsupported => {}
                },
                _ => {}
            }
        }
    }
    (impls, functions)
}
