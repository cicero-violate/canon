pub(crate) fn apply_node_op(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    symbol_id: &str,
    op: &NodeOp,
) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => {
            replace_node(ast, content, handle, new_node.clone())
        }
        NodeOp::InsertBefore { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), true)
        }
        NodeOp::InsertAfter { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), false)
        }
        NodeOp::DeleteNode { handle } => delete_node(ast, content, handle),
        NodeOp::ReorderItems { file: _, new_order } => {
            reorder_items(ast, content, handles, new_order)
        }
        NodeOp::MutateField { handle, mutation } => {
            apply_field_mutation(ast, content, handle, symbol_id, mutation)
        }
        NodeOp::MoveSymbol { .. } => Ok(false),
    }
}


pub(crate) fn apply_node_op(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    symbol_id: &str,
    op: &NodeOp,
) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => {
            replace_node(ast, content, handle, new_node.clone())
        }
        NodeOp::InsertBefore { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), true)
        }
        NodeOp::InsertAfter { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), false)
        }
        NodeOp::DeleteNode { handle } => delete_node(ast, content, handle),
        NodeOp::ReorderItems { file: _, new_order } => {
            reorder_items(ast, content, handles, new_order)
        }
        NodeOp::MutateField { handle, mutation } => {
            apply_field_mutation(ast, content, handle, symbol_id, mutation)
        }
        NodeOp::MoveSymbol { .. } => Ok(false),
    }
}


fn replace_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("replace node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("replace node: container not found"))?;
    items[idx] = new_node;
    Ok(true)
}


fn replace_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("replace node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("replace node: container not found"))?;
    items[idx] = new_node;
    Ok(true)
}


fn insert_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
    before: bool,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("insert node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("insert node: container not found"))?;
    let insert_at = if before { idx } else { idx.saturating_add(1) };
    if insert_at > items.len() {
        anyhow::bail!("insert index out of bounds");
    }
    items.insert(insert_at, new_node);
    Ok(true)
}


fn insert_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
    before: bool,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("insert node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("insert node: container not found"))?;
    let insert_at = if before { idx } else { idx.saturating_add(1) };
    if insert_at > items.len() {
        anyhow::bail!("insert index out of bounds");
    }
    items.insert(insert_at, new_node);
    Ok(true)
}


fn delete_node(ast: &mut syn::File, content: &str, handle: &NodeHandle) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("delete node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("delete node: container not found"))?;
    if idx >= items.len() {
        anyhow::bail!("delete index out of bounds");
    }
    items.remove(idx);
    Ok(true)
}


fn delete_node(ast: &mut syn::File, content: &str, handle: &NodeHandle) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("delete node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("delete node: container not found"))?;
    if idx >= items.len() {
        anyhow::bail!("delete index out of bounds");
    }
    items.remove(idx);
    Ok(true)
}


fn reorder_items(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    new_order: &[String],
) -> Result<bool> {
    let mut indices: Vec<usize> = Vec::new();
    let mut items_path: Option<Vec<usize>> = None;
    for symbol_id in new_order {
        let handle = handles
            .get(symbol_id)
            .ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.kind == NodeKind::ImplFn {
            anyhow::bail!("reorder not supported for impl items");
        }
        let (path, idx) = find_item_container_by_span(
                &ast.items,
                content,
                handle.byte_range,
            )
            .ok_or_else(|| anyhow::anyhow!("reorder: item not found by span"))?;
        let ptr = path.clone();
        if let Some(existing) = &items_path {
            if existing != &ptr {
                anyhow::bail!("reorder requires a single container scope");
            }
        } else {
            items_path = Some(ptr);
        }
        indices.push(idx);
    }
    let Some(items_path) = items_path else {
        return Ok(false);
    };
    let items = get_items_container_mut_by_path(&mut ast.items, &items_path)
        .ok_or_else(|| anyhow::anyhow!("reorder: container not found"))?;
    let mut taken = vec![false; items.len()];
    let mut reordered = Vec::with_capacity(items.len());
    for idx in indices {
        if idx >= items.len() || taken[idx] {
            continue;
        }
        reordered.push(items[idx].clone());
        taken[idx] = true;
    }
    for (idx, item) in items.iter().cloned().enumerate() {
        if !taken[idx] {
            reordered.push(item);
        }
    }
    *items = reordered;
    Ok(true)
}


fn reorder_items(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    new_order: &[String],
) -> Result<bool> {
    let mut indices: Vec<usize> = Vec::new();
    let mut items_path: Option<Vec<usize>> = None;
    for symbol_id in new_order {
        let handle = handles
            .get(symbol_id)
            .ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.kind == NodeKind::ImplFn {
            anyhow::bail!("reorder not supported for impl items");
        }
        let (path, idx) = find_item_container_by_span(
                &ast.items,
                content,
                handle.byte_range,
            )
            .ok_or_else(|| anyhow::anyhow!("reorder: item not found by span"))?;
        let ptr = path.clone();
        if let Some(existing) = &items_path {
            if existing != &ptr {
                anyhow::bail!("reorder requires a single container scope");
            }
        } else {
            items_path = Some(ptr);
        }
        indices.push(idx);
    }
    let Some(items_path) = items_path else {
        return Ok(false);
    };
    let items = get_items_container_mut_by_path(&mut ast.items, &items_path)
        .ok_or_else(|| anyhow::anyhow!("reorder: container not found"))?;
    let mut taken = vec![false; items.len()];
    let mut reordered = Vec::with_capacity(items.len());
    for idx in indices {
        if idx >= items.len() || taken[idx] {
            continue;
        }
        reordered.push(items[idx].clone());
        taken[idx] = true;
    }
    for (idx, item) in items.iter().cloned().enumerate() {
        if !taken[idx] {
            reordered.push(item);
        }
    }
    *items = reordered;
    Ok(true)
}


pub(crate) fn apply_node_op(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    symbol_id: &str,
    op: &NodeOp,
) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => {
            replace_node(ast, content, handle, new_node.clone())
        }
        NodeOp::InsertBefore { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), true)
        }
        NodeOp::InsertAfter { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), false)
        }
        NodeOp::DeleteNode { handle } => delete_node(ast, content, handle),
        NodeOp::ReorderItems { file: _, new_order } => {
            reorder_items(ast, content, handles, new_order)
        }
        NodeOp::MutateField { handle, mutation } => {
            apply_field_mutation(ast, content, handle, symbol_id, mutation)
        }
        NodeOp::MoveSymbol { .. } => Ok(false),
    }
}


pub(crate) fn apply_node_op(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    symbol_id: &str,
    op: &NodeOp,
) -> Result<bool> {
    match op {
        NodeOp::ReplaceNode { handle, new_node } => {
            replace_node(ast, content, handle, new_node.clone())
        }
        NodeOp::InsertBefore { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), true)
        }
        NodeOp::InsertAfter { handle, new_node } => {
            insert_node(ast, content, handle, new_node.clone(), false)
        }
        NodeOp::DeleteNode { handle } => delete_node(ast, content, handle),
        NodeOp::ReorderItems { file: _, new_order } => {
            reorder_items(ast, content, handles, new_order)
        }
        NodeOp::MutateField { handle, mutation } => {
            apply_field_mutation(ast, content, handle, symbol_id, mutation)
        }
        NodeOp::MoveSymbol { .. } => Ok(false),
    }
}


fn replace_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("replace node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("replace node: container not found"))?;
    items[idx] = new_node;
    Ok(true)
}


fn replace_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("replace node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("replace node: container not found"))?;
    items[idx] = new_node;
    Ok(true)
}


fn insert_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
    before: bool,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("insert node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("insert node: container not found"))?;
    let insert_at = if before { idx } else { idx.saturating_add(1) };
    if insert_at > items.len() {
        anyhow::bail!("insert index out of bounds");
    }
    items.insert(insert_at, new_node);
    Ok(true)
}


fn insert_node(
    ast: &mut syn::File,
    content: &str,
    handle: &NodeHandle,
    new_node: syn::Item,
    before: bool,
) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("insert node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("insert node: container not found"))?;
    let insert_at = if before { idx } else { idx.saturating_add(1) };
    if insert_at > items.len() {
        anyhow::bail!("insert index out of bounds");
    }
    items.insert(insert_at, new_node);
    Ok(true)
}


fn delete_node(ast: &mut syn::File, content: &str, handle: &NodeHandle) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("delete node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("delete node: container not found"))?;
    if idx >= items.len() {
        anyhow::bail!("delete index out of bounds");
    }
    items.remove(idx);
    Ok(true)
}


fn delete_node(ast: &mut syn::File, content: &str, handle: &NodeHandle) -> Result<bool> {
    let (path, idx) = find_item_container_by_span(&ast.items, content, handle.byte_range)
        .ok_or_else(|| anyhow::anyhow!("delete node: item not found by span"))?;
    let items = get_items_container_mut_by_path(&mut ast.items, &path)
        .ok_or_else(|| anyhow::anyhow!("delete node: container not found"))?;
    if idx >= items.len() {
        anyhow::bail!("delete index out of bounds");
    }
    items.remove(idx);
    Ok(true)
}


fn reorder_items(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    new_order: &[String],
) -> Result<bool> {
    let mut indices: Vec<usize> = Vec::new();
    let mut items_path: Option<Vec<usize>> = None;
    for symbol_id in new_order {
        let handle = handles
            .get(symbol_id)
            .ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.kind == NodeKind::ImplFn {
            anyhow::bail!("reorder not supported for impl items");
        }
        let (path, idx) = find_item_container_by_span(
                &ast.items,
                content,
                handle.byte_range,
            )
            .ok_or_else(|| anyhow::anyhow!("reorder: item not found by span"))?;
        let ptr = path.clone();
        if let Some(existing) = &items_path {
            if existing != &ptr {
                anyhow::bail!("reorder requires a single container scope");
            }
        } else {
            items_path = Some(ptr);
        }
        indices.push(idx);
    }
    let Some(items_path) = items_path else {
        return Ok(false);
    };
    let items = get_items_container_mut_by_path(&mut ast.items, &items_path)
        .ok_or_else(|| anyhow::anyhow!("reorder: container not found"))?;
    let mut taken = vec![false; items.len()];
    let mut reordered = Vec::with_capacity(items.len());
    for idx in indices {
        if idx >= items.len() || taken[idx] {
            continue;
        }
        reordered.push(items[idx].clone());
        taken[idx] = true;
    }
    for (idx, item) in items.iter().cloned().enumerate() {
        if !taken[idx] {
            reordered.push(item);
        }
    }
    *items = reordered;
    Ok(true)
}


fn reorder_items(
    ast: &mut syn::File,
    content: &str,
    handles: &HashMap<String, NodeHandle>,
    new_order: &[String],
) -> Result<bool> {
    let mut indices: Vec<usize> = Vec::new();
    let mut items_path: Option<Vec<usize>> = None;
    for symbol_id in new_order {
        let handle = handles
            .get(symbol_id)
            .ok_or_else(|| anyhow::anyhow!("missing handle for {}", symbol_id))?;
        if handle.kind == NodeKind::ImplFn {
            anyhow::bail!("reorder not supported for impl items");
        }
        let (path, idx) = find_item_container_by_span(
                &ast.items,
                content,
                handle.byte_range,
            )
            .ok_or_else(|| anyhow::anyhow!("reorder: item not found by span"))?;
        let ptr = path.clone();
        if let Some(existing) = &items_path {
            if existing != &ptr {
                anyhow::bail!("reorder requires a single container scope");
            }
        } else {
            items_path = Some(ptr);
        }
        indices.push(idx);
    }
    let Some(items_path) = items_path else {
        return Ok(false);
    };
    let items = get_items_container_mut_by_path(&mut ast.items, &items_path)
        .ok_or_else(|| anyhow::anyhow!("reorder: container not found"))?;
    let mut taken = vec![false; items.len()];
    let mut reordered = Vec::with_capacity(items.len());
    for idx in indices {
        if idx >= items.len() || taken[idx] {
            continue;
        }
        reordered.push(items[idx].clone());
        taken[idx] = true;
    }
    for (idx, item) in items.iter().cloned().enumerate() {
        if !taken[idx] {
            reordered.push(item);
        }
    }
    *items = reordered;
    Ok(true)
}
