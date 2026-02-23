use crate::model::core_span::{span_to_offsets, span_to_range};


use crate::state::{NodeHandle, NodeKind};


use syn::spanned::Spanned;


pub enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}


pub fn get_item_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut syn::Item> {
    let (container_path, idx) = path.split_at(path.len().saturating_sub(1));
    let idx = *idx.first()?;
    let container = get_items_container_mut_by_path(items, container_path)?;
    container.get_mut(idx)
}


pub fn rename_ident_in_item(item: &mut syn::Item, target: &str, new_name: &str) -> bool {
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
