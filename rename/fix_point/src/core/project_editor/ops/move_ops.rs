use anyhow::Result;


use crate::state::NodeHandle;


use super::helpers::{find_item_container_by_span, get_items_container_mut_by_path};


fn find_mod_indices_by_name(
    items: &[syn::Item],
    segments: &[&str],
) -> Option<Vec<usize>> {
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


pub fn move_symbol_intra_file(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_module_path: &str,
) -> Result<bool> {
    let target_segments: Vec<&str> = new_module_path
        .trim_start_matches("crate::")
        .split("::")
        .filter(|s| !s.is_empty())
        .collect();
    let target_path = find_mod_indices_by_name(&ast.items, &target_segments);
    let Some(target_path) = target_path else {
        return Ok(false);
    };
    let (src_path, src_idx) = find_item_container_by_span(
            &ast.items,
            content,
            handle.byte_range,
        )
        .ok_or_else(|| anyhow::anyhow!("MoveSymbol: item not found by span"))?;
    let item = {
        let src_items = get_items_container_mut_by_path(&mut ast.items, &src_path)
            .ok_or_else(|| anyhow::anyhow!("MoveSymbol: source container not found"))?;
        if src_idx >= src_items.len() {
            anyhow::bail!("MoveSymbol: item index out of bounds");
        }
        src_items.remove(src_idx)
    };
    let target_items = get_items_container_mut_by_path(&mut ast.items, &target_path)
        .ok_or_else(|| {
            anyhow::anyhow!("MoveSymbol: target mod path not found after extraction")
        })?;
    target_items.push(item);
    Ok(true)
}
