enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}


enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}


fn resolve_target_mut<'a>(
    ast: &'a mut syn::File,
    content: &str,
    handle: &NodeHandle,
    _symbol_id: &str,
) -> Option<TargetItemMut<'a>> {
    if handle.kind == NodeKind::ImplFn {
        let (impl_path, fn_index) = find_impl_item_fn_by_span(
            &ast.items,
            content,
            handle.byte_range,
        )?;
        let impl_item = get_item_mut_by_path(&mut ast.items, &impl_path)?;
        let item_impl = match impl_item {
            syn::Item::Impl(item_impl) => item_impl,
            _ => return None,
        };
        let impl_fn = match item_impl.items.get_mut(fn_index)? {
            syn::ImplItem::Fn(impl_fn) => impl_fn,
            _ => return None,
        };
        return Some(TargetItemMut::ImplFn(impl_fn));
    }
    let (container_path, idx) = find_item_container_by_span(
        &ast.items,
        content,
        handle.byte_range,
    )?;
    let container = get_items_container_mut_by_path(&mut ast.items, &container_path)?;
    let item = container.get_mut(idx)?;
    Some(TargetItemMut::Top(item))
}


fn resolve_target_mut<'a>(
    ast: &'a mut syn::File,
    content: &str,
    handle: &NodeHandle,
    _symbol_id: &str,
) -> Option<TargetItemMut<'a>> {
    if handle.kind == NodeKind::ImplFn {
        let (impl_path, fn_index) = find_impl_item_fn_by_span(
            &ast.items,
            content,
            handle.byte_range,
        )?;
        let impl_item = get_item_mut_by_path(&mut ast.items, &impl_path)?;
        let item_impl = match impl_item {
            syn::Item::Impl(item_impl) => item_impl,
            _ => return None,
        };
        let impl_fn = match item_impl.items.get_mut(fn_index)? {
            syn::ImplItem::Fn(impl_fn) => impl_fn,
            _ => return None,
        };
        return Some(TargetItemMut::ImplFn(impl_fn));
    }
    let (container_path, idx) = find_item_container_by_span(
        &ast.items,
        content,
        handle.byte_range,
    )?;
    let container = get_items_container_mut_by_path(&mut ast.items, &container_path)?;
    let item = container.get_mut(idx)?;
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


fn find_item_container_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if item_byte_range(item, content) == target {
            return Some((Vec::new(), idx));
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, inner_idx)) = find_item_container_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, inner_idx));
                }
            }
        }
    }
    None
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


fn find_impl_item_fn_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if let syn::Item::Impl(item_impl) = item {
            for (fn_idx, impl_item) in item_impl.items.iter().enumerate() {
                if let syn::ImplItem::Fn(impl_fn) = impl_item {
                    let span = item_span_range(impl_fn.span(), content);
                    if span == target {
                        return Some((vec![idx], fn_idx));
                    }
                }
            }
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, fn_idx)) = find_impl_item_fn_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, fn_idx));
                }
            }
        }
    }
    None
}


fn find_item_container_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if item_byte_range(item, content) == target {
            return Some((Vec::new(), idx));
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, inner_idx)) = find_item_container_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, inner_idx));
                }
            }
        }
    }
    None
}


fn find_impl_item_fn_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if let syn::Item::Impl(item_impl) = item {
            for (fn_idx, impl_item) in item_impl.items.iter().enumerate() {
                if let syn::ImplItem::Fn(impl_fn) = impl_item {
                    let span = item_span_range(impl_fn.span(), content);
                    if span == target {
                        return Some((vec![idx], fn_idx));
                    }
                }
            }
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, fn_idx)) = find_impl_item_fn_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, fn_idx));
                }
            }
        }
    }
    None
}


fn get_items_container_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut Vec<syn::Item>> {
    if let Some((first, rest)) = path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return get_items_container_mut_by_path(inner, rest);
    }
    Some(items)
}


fn get_items_container_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut Vec<syn::Item>> {
    if let Some((first, rest)) = path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return get_items_container_mut_by_path(inner, rest);
    }
    Some(items)
}


fn get_item_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut syn::Item> {
    let (container_path, idx) = path.split_at(path.len().saturating_sub(1));
    let idx = *idx.first()?;
    let container = get_items_container_mut_by_path(items, container_path)?;
    container.get_mut(idx)
}


fn get_item_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut syn::Item> {
    let (container_path, idx) = path.split_at(path.len().saturating_sub(1));
    let idx = *idx.first()?;
    let container = get_items_container_mut_by_path(items, container_path)?;
    container.get_mut(idx)
}


fn item_byte_range(item: &syn::Item, content: &str) -> (usize, usize) {
    item_span_range(item.span(), content)
}


fn item_byte_range(item: &syn::Item, content: &str) -> (usize, usize) {
    item_span_range(item.span(), content)
}


fn item_span_range(span: proc_macro2::Span, content: &str) -> (usize, usize) {
    let range = span_to_range(span);
    span_to_offsets(content, &range.start, &range.end)
}


enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}


fn item_span_range(span: proc_macro2::Span, content: &str) -> (usize, usize) {
    let range = span_to_range(span);
    span_to_offsets(content, &range.start, &range.end)
}


enum TargetItemMut<'a> {
    Top(&'a mut syn::Item),
    ImplFn(&'a mut syn::ImplItemFn),
}


fn resolve_target_mut<'a>(
    ast: &'a mut syn::File,
    content: &str,
    handle: &NodeHandle,
    _symbol_id: &str,
) -> Option<TargetItemMut<'a>> {
    if handle.kind == NodeKind::ImplFn {
        let (impl_path, fn_index) = find_impl_item_fn_by_span(
            &ast.items,
            content,
            handle.byte_range,
        )?;
        let impl_item = get_item_mut_by_path(&mut ast.items, &impl_path)?;
        let item_impl = match impl_item {
            syn::Item::Impl(item_impl) => item_impl,
            _ => return None,
        };
        let impl_fn = match item_impl.items.get_mut(fn_index)? {
            syn::ImplItem::Fn(impl_fn) => impl_fn,
            _ => return None,
        };
        return Some(TargetItemMut::ImplFn(impl_fn));
    }
    let (container_path, idx) = find_item_container_by_span(
        &ast.items,
        content,
        handle.byte_range,
    )?;
    let container = get_items_container_mut_by_path(&mut ast.items, &container_path)?;
    let item = container.get_mut(idx)?;
    Some(TargetItemMut::Top(item))
}


fn resolve_target_mut<'a>(
    ast: &'a mut syn::File,
    content: &str,
    handle: &NodeHandle,
    _symbol_id: &str,
) -> Option<TargetItemMut<'a>> {
    if handle.kind == NodeKind::ImplFn {
        let (impl_path, fn_index) = find_impl_item_fn_by_span(
            &ast.items,
            content,
            handle.byte_range,
        )?;
        let impl_item = get_item_mut_by_path(&mut ast.items, &impl_path)?;
        let item_impl = match impl_item {
            syn::Item::Impl(item_impl) => item_impl,
            _ => return None,
        };
        let impl_fn = match item_impl.items.get_mut(fn_index)? {
            syn::ImplItem::Fn(impl_fn) => impl_fn,
            _ => return None,
        };
        return Some(TargetItemMut::ImplFn(impl_fn));
    }
    let (container_path, idx) = find_item_container_by_span(
        &ast.items,
        content,
        handle.byte_range,
    )?;
    let container = get_items_container_mut_by_path(&mut ast.items, &container_path)?;
    let item = container.get_mut(idx)?;
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


fn find_item_container_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if item_byte_range(item, content) == target {
            return Some((Vec::new(), idx));
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, inner_idx)) = find_item_container_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, inner_idx));
                }
            }
        }
    }
    None
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


fn find_impl_item_fn_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if let syn::Item::Impl(item_impl) = item {
            for (fn_idx, impl_item) in item_impl.items.iter().enumerate() {
                if let syn::ImplItem::Fn(impl_fn) = impl_item {
                    let span = item_span_range(impl_fn.span(), content);
                    if span == target {
                        return Some((vec![idx], fn_idx));
                    }
                }
            }
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, fn_idx)) = find_impl_item_fn_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, fn_idx));
                }
            }
        }
    }
    None
}


fn find_item_container_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if item_byte_range(item, content) == target {
            return Some((Vec::new(), idx));
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, inner_idx)) = find_item_container_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, inner_idx));
                }
            }
        }
    }
    None
}


fn find_impl_item_fn_by_span(
    items: &[syn::Item],
    content: &str,
    target: (usize, usize),
) -> Option<(Vec<usize>, usize)> {
    for (idx, item) in items.iter().enumerate() {
        if let syn::Item::Impl(item_impl) = item {
            for (fn_idx, impl_item) in item_impl.items.iter().enumerate() {
                if let syn::ImplItem::Fn(impl_fn) = impl_item {
                    let span = item_span_range(impl_fn.span(), content);
                    if span == target {
                        return Some((vec![idx], fn_idx));
                    }
                }
            }
        }
        if let syn::Item::Mod(m) = item {
            if let Some((_, inner)) = &m.content {
                if let Some((mut path, fn_idx)) = find_impl_item_fn_by_span(
                    inner,
                    content,
                    target,
                ) {
                    path.insert(0, idx);
                    return Some((path, fn_idx));
                }
            }
        }
    }
    None
}


fn get_items_container_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut Vec<syn::Item>> {
    if let Some((first, rest)) = path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return get_items_container_mut_by_path(inner, rest);
    }
    Some(items)
}


fn get_items_container_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut Vec<syn::Item>> {
    if let Some((first, rest)) = path.split_first() {
        let item = items.get_mut(*first)?;
        let item_mod = match item {
            syn::Item::Mod(item_mod) => item_mod,
            _ => return None,
        };
        let (_, inner) = item_mod.content.as_mut()?;
        return get_items_container_mut_by_path(inner, rest);
    }
    Some(items)
}


fn get_item_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut syn::Item> {
    let (container_path, idx) = path.split_at(path.len().saturating_sub(1));
    let idx = *idx.first()?;
    let container = get_items_container_mut_by_path(items, container_path)?;
    container.get_mut(idx)
}


fn get_item_mut_by_path<'a>(
    items: &'a mut Vec<syn::Item>,
    path: &[usize],
) -> Option<&'a mut syn::Item> {
    let (container_path, idx) = path.split_at(path.len().saturating_sub(1));
    let idx = *idx.first()?;
    let container = get_items_container_mut_by_path(items, container_path)?;
    container.get_mut(idx)
}


fn item_byte_range(item: &syn::Item, content: &str) -> (usize, usize) {
    item_span_range(item.span(), content)
}


fn item_byte_range(item: &syn::Item, content: &str) -> (usize, usize) {
    item_span_range(item.span(), content)
}


fn item_span_range(span: proc_macro2::Span, content: &str) -> (usize, usize) {
    let range = span_to_range(span);
    span_to_offsets(content, &range.start, &range.end)
}


fn item_span_range(span: proc_macro2::Span, content: &str) -> (usize, usize) {
    let range = span_to_range(span);
    span_to_offsets(content, &range.start, &range.end)
}
