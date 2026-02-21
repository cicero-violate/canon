use anyhow::Result;
use quote::ToTokens;

use crate::state::NodeHandle;
use crate::structured::FieldMutation;

use super::helpers::{rename_ident_in_item, resolve_target_mut, TargetItemMut};

pub(super) fn apply_field_mutation(ast: &mut syn::File, handle: &NodeHandle, symbol_id: &str, mutation: &FieldMutation) -> Result<bool> {
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
                    let filtered: syn::punctuated::Punctuated<syn::Field, syn::token::Comma> =
                        named.named.iter().cloned().filter(|f| f.ident.as_ref().map(|i| i != name).unwrap_or(true)).collect();
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
