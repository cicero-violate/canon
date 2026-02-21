use crate::state::{NodeHandle, NodeKind};

pub(super) enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}

pub(super) fn resolve_target_mut<'a>(ast: &'a mut syn::File, handle: &NodeHandle, symbol_id: &str) -> Option<TargetItemMut<'a>> {
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

pub(super) fn rename_ident_in_item(item: &mut syn::Item, target: &str, new_name: &str) -> bool {
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

pub(super) fn resolve_items_container_mut<'a>(ast: &'a mut syn::File, module_path: &[usize]) -> Option<&'a mut Vec<syn::Item>> {
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
