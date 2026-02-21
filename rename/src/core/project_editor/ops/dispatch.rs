use std::collections::HashMap;

use anyhow::Result;

use crate::state::{NodeHandle, NodeKind};
use crate::structured::NodeOp;

use super::field_mutations::apply_field_mutation;
use super::helpers::{resolve_items_container_mut};
use super::move_ops::move_symbol_intra_file;

pub(crate) fn apply_node_op(ast: &mut syn::File, handles: &HashMap<String, NodeHandle>, symbol_id: &str, op: &NodeOp) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => replace_node(ast, handle, new_node.clone()),
        NodeOp::InsertBefore { handle, new_node } => insert_node(ast, handle, new_node.clone(), true),
        NodeOp::InsertAfter { handle, new_node } => insert_node(ast, handle, new_node.clone(), false),
        NodeOp::DeleteNode { handle } => delete_node(ast, handle),
        NodeOp::ReorderItems { file: _, new_order } => reorder_items(ast, handles, new_order),
        NodeOp::MutateField { handle, mutation } => apply_field_mutation(ast, handle, symbol_id, mutation),
        NodeOp::MoveSymbol { handle, new_module_path, .. } => move_symbol_intra_file(ast, handle, new_module_path),
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
