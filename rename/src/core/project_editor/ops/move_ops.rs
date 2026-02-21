use anyhow::Result;

use crate::state::NodeHandle;

use super::helpers::resolve_items_container_mut;

/// Intra-file move: extract item from its current position and append it into the
/// target inline mod named by `new_module_path` (e.g. "crate::foo::bar").
/// Returns Ok(false) if source and target module are the same (no-op).
/// Returns Ok(false) if the target module path does not exist as an inline mod in
/// this file — cross-file moves are handled at the mod.rs level.
pub(super) fn move_symbol_intra_file(ast: &mut syn::File, handle: &NodeHandle, new_module_path: &str) -> Result<bool> {
    if !handle.nested_path.is_empty() {
        anyhow::bail!("MoveSymbol not supported for items nested inside impl blocks");
    }
    let target_segments: Vec<&str> = new_module_path.trim_start_matches("crate::").split("::").filter(|s| !s.is_empty()).collect();

    let target_indices = find_mod_indices_by_name(&ast.items, &target_segments);
    let Some(target_indices) = target_indices else {
        return Ok(false);
    };

    if target_segments.is_empty() && handle.item_index < ast.items.len() {
        return Ok(false);
    }

    let item = ast
        .items
        .get(handle.item_index)
        .ok_or_else(|| anyhow::anyhow!("MoveSymbol: item_index {} out of bounds", handle.item_index))?
        .clone();

    ast.items.remove(handle.item_index);

    let adjusted: Vec<usize> = {
        let mut v = target_indices.clone();
        if let Some(first) = v.first_mut() {
            if *first > handle.item_index {
                *first -= 1;
            }
        }
        v
    };

    let container = resolve_items_container_mut(ast, &adjusted)
        .ok_or_else(|| anyhow::anyhow!("MoveSymbol: target mod path not found after extraction"))?;
    container.push(item);
    Ok(true)
}

/// Walk items by module name segments, returning the index path to the container vec.
/// Returns None if any segment is not found as an inline mod.
fn find_mod_indices_by_name(items: &[syn::Item], segments: &[&str]) -> Option<Vec<usize>> {
    if segments.is_empty() {
        return Some(vec![]);
    }
    let (head, tail) = segments.split_first()?;
    for (i, item) in items.iter().enumerate() {
        if let syn::Item::Mod(m) = item {
            if m.ident == head {
                if let Some((_, inner)) = &m.content {
                    let mut rest = find_mod_indices_by_name(inner, tail)?;
                    rest.insert(0, i);
                    return Some(rest);
                }
            }
        }
    }
    None
}
