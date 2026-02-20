use std::collections::HashMap;

use anyhow::Result;
use quote::ToTokens;

use crate::state::{NodeHandle, NodeKind};
use crate::structured::{FieldMutation, NodeOp};

pub(super) fn apply_node_op(ast: &mut syn::File, handles: &HashMap<String, NodeHandle>, symbol_id: &str, op: &NodeOp) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => replace_node(ast, handle, new_node.clone()),
        NodeOp::InsertBefore { handle, new_node } => insert_node(ast, handle, new_node.clone(), true),
        NodeOp::InsertAfter { handle, new_node } => insert_node(ast, handle, new_node.clone(), false),
        NodeOp::DeleteNode { handle } => delete_node(ast, handle),
        NodeOp::ReorderItems { file: _, new_order } => reorder_items(ast, handles, new_order),
        NodeOp::MutateField { handle, mutation } => apply_field_mutation(ast, handle, symbol_id, mutation),
        NodeOp::MoveSymbol { .. } => Ok(false),
    }
}

fn apply_field_mutation(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, mutation: &FieldMutation) -> Result<bool> {
    match mutation {
        FieldMutation::RenameIdent(new_name) => {
            if rename_ident(ast, handle, symbol_id, new_name) {
                Ok(true)
            } else {
                anyhow::bail!("rename failed for {}", symbol_id);
            }
        }
        FieldMutation::ChangeVisibility(new_vis) => {
            if change_visibility(ast, handle, symbol_id, new_vis) {
                Ok(true)
            } else {
                anyhow::bail!("visibility change failed for {}", symbol_id);
            }
        }
        FieldMutation::AddAttribute(attr) => {
            if add_attribute(ast, handle, symbol_id, attr.clone()) {
                Ok(true)
            } else {
                anyhow::bail!("add attribute failed for {}", symbol_id);
            }
        }
        FieldMutation::RemoveAttribute(name) => {
            if remove_attribute(ast, handle, symbol_id, name) {
                Ok(true)
            } else {
                anyhow::bail!("remove attribute failed for {}", symbol_id);
            }
        }
        FieldMutation::ReplaceSignature(sig) => {
            if replace_signature(ast, handle, symbol_id, sig.clone()) {
                Ok(true)
            } else {
                anyhow::bail!("replace signature failed for {}", symbol_id);
            }
        }
        FieldMutation::AddStructField(field) => {
            if add_struct_field(ast, handle, symbol_id, field.clone()) {
                Ok(true)
            } else {
                anyhow::bail!("add struct field failed for {}", symbol_id);
            }
        }
        FieldMutation::RemoveStructField(name) => {
            if remove_struct_field(ast, handle, symbol_id, name) {
                Ok(true)
            } else {
                anyhow::bail!("remove struct field failed for {}", symbol_id);
            }
        }
        FieldMutation::AddVariant(variant) => {
            if add_variant(ast, handle, symbol_id, variant.clone()) {
                Ok(true)
            } else {
                anyhow::bail!("add variant failed for {}", symbol_id);
            }
        }
        FieldMutation::RemoveVariant(name) => {
            if remove_variant(ast, handle, symbol_id, name) {
                Ok(true)
            } else {
                anyhow::bail!("remove variant failed for {}", symbol_id);
            }
        }
    }
}

fn rename_ident(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, new_name: &str) -> bool {
    let target = symbol_id.rsplit("::").next().unwrap_or(symbol_id);

    match resolve_target_mut(ast, handle, symbol_id) {
        Some(TargetItemMut::Top(item)) => rename_ident_in_item(item, target, new_name),
        Some(TargetItemMut::ImplFn(impl_fn)) => {
            if impl_fn.sig.ident == target {
                impl_fn.sig.ident = syn::Ident::new(new_name, impl_fn.sig.ident.span());
                true
            } else {
                false
            }
        }
        None => false,
    }
}

fn replace_node(ast: &mut syn::File, handle: &NodeHandle, new_node: syn::Item) -> Result<bool> {
    if !handle.nested_path.is_empty() {
        anyhow::bail!("replace node not supported for nested items");
    }
    let item = ast.items.get_mut(handle.item_index).ok_or_else(|| anyhow::anyhow!("item index out of bounds"))?;
    *item = new_node;
    Ok(true)
}

fn insert_node(ast: &mut syn::File, handle: &NodeHandle, new_node: syn::Item, before: bool) -> Result<bool> {
    if !handle.nested_path.is_empty() {
        anyhow::bail!("insert node not supported for nested items");
    }
    let mut idx = handle.item_index;
    if !before {
        idx = idx.saturating_add(1);
    }
    if idx > ast.items.len() {
        anyhow::bail!("insert index out of bounds");
    }
    ast.items.insert(idx, new_node);
    Ok(true)
}

fn delete_node(ast: &mut syn::File, handle: &NodeHandle) -> Result<bool> {
    if !handle.nested_path.is_empty() {
        anyhow::bail!("delete node not supported for nested items");
    }
    if handle.item_index >= ast.items.len() {
        anyhow::bail!("delete index out of bounds");
    }
    ast.items.remove(handle.item_index);
    Ok(true)
}

fn reorder_items(ast: &mut syn::File, handles: &HashMap<String, NodeHandle>, new_order: &[String]) -> Result<bool> {
    let mut container_path: Option<Vec<usize>> = None;
    for symbol_id in new_order {
        let handle = handles.get(symbol_id).ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.kind == NodeKind::ImplFn {
            anyhow::bail!("reorder not supported for impl items");
        }
        match &container_path {
            Some(existing) if existing.as_slice() != handle.nested_path.as_slice() => {
                anyhow::bail!("reorder requires a single container scope");
            }
            None => container_path = Some(handle.nested_path.clone()),
            _ => {}
        }
    }

    let container_path = container_path.unwrap_or_default();
    let items = resolve_items_container_mut(ast, &container_path).ok_or_else(|| anyhow::anyhow!("failed to resolve container for reorder"))?;

    let mut taken = vec![false; items.len()];
    let mut reordered = Vec::with_capacity(items.len());

    for symbol_id in new_order {
        let handle = handles.get(symbol_id).ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.item_index >= items.len() || taken[handle.item_index] {
            continue;
        }
        reordered.push(items[handle.item_index].clone());
        taken[handle.item_index] = true;
    }

    for (idx, item) in items.iter().cloned().enumerate() {
        if !taken[idx] {
            reordered.push(item);
        }
    }

    *items = reordered;
    Ok(true)
}

fn change_visibility(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, new_vis: &syn::Visibility) -> bool {
    if let Some(target) = resolve_target_mut(ast, handle, symbol_id) {
        match target {
            TargetItemMut::Top(item) => match item {
                syn::Item::Fn(item_fn) => {
                    item_fn.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Struct(item_struct) => {
                    item_struct.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Enum(item_enum) => {
                    item_enum.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Trait(item_trait) => {
                    item_trait.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Type(item_type) => {
                    item_type.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Const(item_const) => {
                    item_const.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Mod(item_mod) => {
                    item_mod.vis = new_vis.clone();
                    return true;
                }
                syn::Item::Impl(_item_impl) => {
                    return false;
                }
                _ => {}
            },
            TargetItemMut::ImplFn(item_fn) => {
                item_fn.vis = new_vis.clone();
                return true;
            }
        }
    }
    false
}

fn add_attribute(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, attr: syn::Attribute) -> bool {
    if let Some(target) = resolve_target_mut(ast, handle, symbol_id) {
        match target {
            TargetItemMut::Top(item) => match item {
                syn::Item::Fn(item_fn) => item_fn.attrs.push(attr),
                syn::Item::Struct(item_struct) => item_struct.attrs.push(attr),
                syn::Item::Enum(item_enum) => item_enum.attrs.push(attr),
                syn::Item::Trait(item_trait) => item_trait.attrs.push(attr),
                syn::Item::Type(item_type) => item_type.attrs.push(attr),
                syn::Item::Const(item_const) => item_const.attrs.push(attr),
                syn::Item::Mod(item_mod) => item_mod.attrs.push(attr),
                syn::Item::Impl(item_impl) => item_impl.attrs.push(attr),
                _ => return false,
            },
            TargetItemMut::ImplFn(item_fn) => item_fn.attrs.push(attr),
        }
        return true;
    }
    false
}

fn remove_attribute(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, name: &str) -> bool {
    let mut removed = false;
    if let Some(target) = resolve_target_mut(ast, handle, symbol_id) {
        let matcher = |attr: &syn::Attribute| attr.path().is_ident(name) || attr.path().to_token_stream().to_string() == name;
        match target {
            TargetItemMut::Top(item) => match item {
                syn::Item::Fn(item_fn) => item_fn.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Struct(item_struct) => item_struct.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Enum(item_enum) => item_enum.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Trait(item_trait) => item_trait.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Type(item_type) => item_type.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Const(item_const) => item_const.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Mod(item_mod) => item_mod.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                syn::Item::Impl(item_impl) => item_impl.attrs.retain(|a| {
                    let keep = !matcher(a);
                    removed |= !keep;
                    keep
                }),
                _ => {}
            },
            TargetItemMut::ImplFn(item_fn) => item_fn.attrs.retain(|a| {
                let keep = !matcher(a);
                removed |= !keep;
                keep
            }),
        }
    }
    removed
}

fn replace_signature(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, sig: syn::Signature) -> bool {
    if let Some(target) = resolve_target_mut(ast, handle, symbol_id) {
        match target {
            TargetItemMut::Top(item) => {
                if let syn::Item::Fn(item_fn) = item {
                    item_fn.sig = sig;
                    return true;
                }
            }
            TargetItemMut::ImplFn(item_fn) => {
                item_fn.sig = sig;
                return true;
            }
        }
    }
    false
}

fn add_struct_field(ast: &mut syn::File, handle: &NodeHandle, _symbol_id: &str, field: syn::Field) -> bool {
    if let Some(TargetItemMut::Top(item)) = resolve_target_mut(ast, handle, _symbol_id) {
        if let syn::Item::Struct(item_struct) = item {
            match &mut item_struct.fields {
                syn::Fields::Named(named) => {
                    named.named.push(field);
                    return true;
                }
                syn::Fields::Unnamed(unnamed) => {
                    unnamed.unnamed.push(field);
                    return true;
                }
                syn::Fields::Unit => return false,
            }
        }
    }
    false
}

fn remove_struct_field(ast: &mut syn::File, handle: &NodeHandle, _symbol_id: &str, name: &str) -> bool {
    if let Some(TargetItemMut::Top(item)) = resolve_target_mut(ast, handle, _symbol_id) {
        if let syn::Item::Struct(item_struct) = item {
            match &mut item_struct.fields {
                syn::Fields::Named(named) => {
                    let before = named.named.len();
                    let filtered: syn::punctuated::Punctuated<syn::Field, syn::token::Comma> = named.named.iter().cloned().filter(|f| f.ident.as_ref().map(|i| i != name).unwrap_or(true)).collect();
                    named.named = filtered;
                    return named.named.len() != before;
                }
                syn::Fields::Unnamed(_unnamed) => return false,
                syn::Fields::Unit => return false,
            }
        }
    }
    false
}

fn add_variant(ast: &mut syn::File, handle: &NodeHandle, _symbol_id: &str, variant: syn::Variant) -> bool {
    if let Some(TargetItemMut::Top(item)) = resolve_target_mut(ast, handle, _symbol_id) {
        if let syn::Item::Enum(item_enum) = item {
            item_enum.variants.push(variant);
            return true;
        }
    }
    false
}

fn remove_variant(ast: &mut syn::File, handle: &NodeHandle, _symbol_id: &str, name: &str) -> bool {
    if let Some(TargetItemMut::Top(item)) = resolve_target_mut(ast, handle, _symbol_id) {
        if let syn::Item::Enum(item_enum) = item {
            let before = item_enum.variants.len();
            item_enum.variants = item_enum.variants.iter().cloned().filter(|v| v.ident != name).collect();
            return item_enum.variants.len() != before;
        }
    }
    false
}

enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}

fn resolve_target_mut<'a>(ast: &'a mut syn::File, handle: &NodeHandle, symbol_id: &str) -> Option<TargetItemMut<'a>> {
    let target = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
    if handle.kind == NodeKind::ImplFn {
        let (module_path, impl_index) = split_impl_path(handle)?;
        let item = get_item_mut(&mut ast.items, module_path, impl_index)?;
        let impl_item = match item {
            syn::Item::Impl(item_impl) => item_impl,
            _ => return None,
        };
        for item in &mut impl_item.items {
            if let syn::ImplItem::Fn(impl_fn) = item {
                if impl_fn.sig.ident == target {
                    return Some(TargetItemMut::ImplFn(impl_fn));
                }
            }
        }
        return None;
    }

    let item = get_item_mut(&mut ast.items, &handle.nested_path, handle.item_index)?;
    Some(TargetItemMut::Top(item))
}

fn rename_ident_in_item(item: &mut syn::Item, target: &str, new_name: &str) -> bool {
    match item {
        syn::Item::Fn(item_fn) => {
            if item_fn.sig.ident == target {
                item_fn.sig.ident = syn::Ident::new(new_name, item_fn.sig.ident.span());
                return true;
            }
        }
        syn::Item::Struct(item_struct) => {
            if item_struct.ident == target {
                item_struct.ident = syn::Ident::new(new_name, item_struct.ident.span());
                return true;
            }
        }
        syn::Item::Enum(item_enum) => {
            if item_enum.ident == target {
                item_enum.ident = syn::Ident::new(new_name, item_enum.ident.span());
                return true;
            }
        }
        syn::Item::Trait(item_trait) => {
            if item_trait.ident == target {
                item_trait.ident = syn::Ident::new(new_name, item_trait.ident.span());
                return true;
            }
        }
        syn::Item::Type(item_type) => {
            if item_type.ident == target {
                item_type.ident = syn::Ident::new(new_name, item_type.ident.span());
                return true;
            }
        }
        syn::Item::Const(item_const) => {
            if item_const.ident == target {
                item_const.ident = syn::Ident::new(new_name, item_const.ident.span());
                return true;
            }
        }
        syn::Item::Mod(item_mod) => {
            if item_mod.ident == target {
                item_mod.ident = syn::Ident::new(new_name, item_mod.ident.span());
                return true;
            }
        }
        _ => {}
    }
    false
}

fn split_impl_path(handle: &NodeHandle) -> Option<(&[usize], usize)> {
    if handle.nested_path.is_empty() {
        return Some((&[], handle.item_index));
    }
    let (path, last) = handle.nested_path.split_at(handle.nested_path.len() - 1);
    let impl_index = *last.first()?;
    Some((path, impl_index))
}

fn get_item_mut<'a>(items: &'a mut Vec<syn::Item>, module_path: &[usize], item_index: usize) -> Option<&'a mut syn::Item> {
    if let Some((first, rest)) = module_path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return get_item_mut(inner, rest, item_index);
    }
    items.get_mut(item_index)
}

fn resolve_items_container_mut<'a>(ast: &'a mut syn::File, module_path: &[usize]) -> Option<&'a mut Vec<syn::Item>> {
    resolve_items_container_from(&mut ast.items, module_path)
}

fn resolve_items_container_from<'a>(items: &'a mut Vec<syn::Item>, module_path: &[usize]) -> Option<&'a mut Vec<syn::Item>> {
    if let Some((first, rest)) = module_path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return resolve_items_container_from(inner, rest);
    }
    Some(items)
}
