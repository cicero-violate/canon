fn apply_cross_file_moves(
    registry: &mut NodeRegistry,
    changesets: &std::collections::HashMap<PathBuf, Vec<QueuedOp>>,
) -> Result<HashSet<PathBuf>> {
    let mut touched: HashSet<PathBuf> = HashSet::new();
    #[derive(Clone)]
    struct PendingMove {
        src_file: PathBuf,
        dst_file: PathBuf,
        dst_module_segments: Vec<String>,
        symbol_id: String,
        kind: crate::state::NodeKind,
        span: crate::model::types::SpanRange,
        byte_range: (usize, usize),
    }
    struct ResolvedMove {
        pending: PendingMove,
        item: syn::Item,
    }
    let mut pending: Vec<PendingMove> = Vec::new();
    for (src_file, ops) in changesets {
        for queued in ops {
            if let crate::structured::NodeOp::MoveSymbol {
                handle,
                new_module_path,
                ..
            } = &queued.op
            {
                let dst_file = match resolve_or_create_dst_file(
                    registry,
                    new_module_path,
                    src_file,
                ) {
                    Some(f) if f != *src_file => f,
                    _ => continue,
                };
                let segments: Vec<String> = new_module_path
                    .trim_start_matches("crate::")
                    .split("::")
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                pending
                    .push(PendingMove {
                        src_file: src_file.clone(),
                        dst_file,
                        dst_module_segments: segments,
                        symbol_id: queued.symbol_id.clone(),
                        kind: handle.kind,
                        span: handle.span.clone(),
                        byte_range: handle.byte_range,
                    });
            }
        }
    }
    if pending.is_empty() {
        return Ok(touched);
    }
    for mv in &pending {
        ensure_source_loaded(registry, &mv.src_file)?;
        ensure_source_loaded(registry, &mv.dst_file)?;
    }
    let mut seen: HashSet<(PathBuf, usize, usize)> = HashSet::new();
    for mv in &pending {
        seen.insert((mv.src_file.clone(), mv.byte_range.0, mv.byte_range.1));
    }
    let mut extra: Vec<PendingMove> = Vec::new();
    let mut src_ast_cache: std::collections::HashMap<PathBuf, syn::File> = std::collections::HashMap::new();
    for mv in &pending {
        if !matches!(
            mv.kind, crate ::state::NodeKind::Struct | crate ::state::NodeKind::Enum |
            crate ::state::NodeKind::Trait
        ) {
            continue;
        }
        let struct_name = mv
            .symbol_id
            .rsplit("::")
            .next()
            .unwrap_or(&mv.symbol_id)
            .to_string();
        let src_text = registry.sources.get(&mv.src_file).expect("source missing");
        let src_ast = src_ast_cache
            .entry(mv.src_file.clone())
            .or_insert_with(|| {
                syn::parse_file(src_text).unwrap_or_else(|_| syn::parse_quote!())
            });
        for item in &src_ast.items {
            let syn::Item::Impl(item_impl) = item else { continue };
            let self_name = impl_self_type_name(&item_impl.self_ty);
            if self_name.as_deref() != Some(&struct_name) {
                continue;
            }
            let span = crate::model::core_span::span_to_range(item.span());
            let (start, end) = crate::model::core_span::span_to_offsets(
                src_text,
                &span.start,
                &span.end,
            );
            if seen.insert((mv.src_file.clone(), start, end)) {
                extra
                    .push(PendingMove {
                        src_file: mv.src_file.clone(),
                        dst_file: mv.dst_file.clone(),
                        dst_module_segments: mv.dst_module_segments.clone(),
                        symbol_id: mv.symbol_id.clone(),
                        kind: crate::state::NodeKind::Impl,
                        span,
                        byte_range: (start, end),
                    });
            }
        }
    }
    pending.extend(extra);
    let mut resolved: Vec<ResolvedMove> = Vec::new();
    for mv in &pending {
        let src_text = registry.sources.get(&mv.src_file).expect("source missing");
        let (start, end) = mv.byte_range;
        if start >= end || end > src_text.len() {
            anyhow::bail!(
                "cross-file move: invalid span {}..{} for {}", start, end, mv.src_file
                .display()
            );
        }
        let snippet = src_text[start..end].to_string();
        let item = parse_single_item(&snippet)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "cross-file move: span does not parse as a single item in {}", mv
                    .src_file.display()
                )
            })?;
        validate_item_matches(&item, &mv.symbol_id)?;
        resolved
            .push(ResolvedMove {
                pending: mv.clone(),
                item,
            });
    }
    let mut removals_by_file: std::collections::HashMap<PathBuf, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for mv in &pending {
        removals_by_file.entry(mv.src_file.clone()).or_default().push(mv.byte_range);
    }
    for (file, mut ranges) in removals_by_file {
        let src_text = registry.sources.get(&file).expect("source missing");
        let mut text = src_text.as_str().to_string();
        ranges.sort_by(|a, b| b.0.cmp(&a.0));
        for (start, end) in ranges {
            if end <= text.len() && start < end {
                text.replace_range(start..end, "");
            }
        }
        registry.sources.insert(file.clone(), std::sync::Arc::new(text));
        touched.insert(file);
    }
    for mv in resolved {
        let dst_text = registry
            .sources
            .get(&mv.pending.dst_file)
            .expect("source missing");
        let mut text = dst_text.as_str().to_string();
        let snippet = crate::structured::ast_render::render_node({
            let mut item = mv.item;
            promote_private_to_pub_crate(&mut item);
            item
        });
        let insert_offset = find_insert_offset(&text, &mv.pending.dst_module_segments);
        let insert_text = normalize_snippet(&snippet, &text, insert_offset);
        text.insert_str(insert_offset, &insert_text);
        registry.sources.insert(mv.pending.dst_file.clone(), std::sync::Arc::new(text));
        touched.insert(mv.pending.dst_file.clone());
    }
    Ok(touched)
}


fn build_super_tree(tail: &syn::UseTree) -> syn::UseTree {
    let super_ident = syn::Ident::new("super", proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident: super_ident,
        colon2_token: syn::token::PathSep::default(),
        tree: Box::new(tail.clone()),
    })
}


fn collect_new_files(
    registry: &NodeRegistry,
    changesets: &std::collections::HashMap<PathBuf, Vec<QueuedOp>>,
) -> Vec<(PathBuf, String)> {
    let project_root = match find_project_root_sync(registry) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut result = Vec::new();
    for ops in changesets.values() {
        for queued in ops {
            if let crate::structured::NodeOp::MoveSymbol {
                handle: _,
                new_module_path,
                ..
            } = &queued.op
            {
                let norm_dst = normalize_symbol_id(new_module_path);
                let to_path = ModulePath::from_string(&norm_dst);
                let dst_file = match compute_new_file_path(&to_path, &project_root) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !dst_file.exists() && registry.asts.contains_key(&dst_file)
                    && seen.insert(dst_file.clone())
                {
                    result.push((dst_file, norm_dst));
                }
            }
        }
    }
    result
}


fn ensure_source_loaded(registry: &mut NodeRegistry, file: &PathBuf) -> Result<()> {
    if registry.sources.contains_key(file) {
        return Ok(());
    }
    if file.exists() {
        let content = std::fs::read_to_string(file)?;
        registry.sources.insert(file.clone(), Arc::new(content));
        return Ok(());
    }
    registry.sources.insert(file.clone(), Arc::new(String::new()));
    Ok(())
}


fn find_insert_offset(text: &str, dst_module_segments: &[String]) -> usize {
    if dst_module_segments.is_empty() {
        return text.len();
    }
    let Ok(ast) = syn::parse_file(text) else {
        return text.len();
    };
    let segs: Vec<&str> = dst_module_segments.iter().map(|s| s.as_str()).collect();
    if let Some((start, end)) = find_mod_span(&ast.items, &segs, text) {
        let slice = &text[start..end];
        if let Some(rel) = slice.rfind('}') {
            return start + rel;
        }
    }
    text.len()
}


fn find_mod_span(
    items: &[syn::Item],
    segments: &[&str],
    text: &str,
) -> Option<(usize, usize)> {
    let (head, tail) = segments.split_first()?;
    for item in items {
        let syn::Item::Mod(m) = item else { continue };
        if m.ident != head {
            continue;
        }
        if tail.is_empty() {
            let span = crate::model::core_span::span_to_range(m.span());
            let (start, end) = crate::model::core_span::span_to_offsets(
                text,
                &span.start,
                &span.end,
            );
            return Some((start, end));
        }
        let Some((_, inner)) = &m.content else { return None };
        return find_mod_span(inner, tail, text);
    }
    None
}


fn impl_self_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}


fn is_private(vis: &syn::Visibility) -> bool {
    match vis {
        syn::Visibility::Inherited => true,
        syn::Visibility::Restricted(r) => r.path.is_ident("self"),
        _ => false,
    }
}


fn normalize_snippet(snippet: &str, dst_text: &str, insert_offset: usize) -> String {
    let mut out = String::new();
    let before = dst_text.get(..insert_offset).unwrap_or("");
    let after = dst_text.get(insert_offset..).unwrap_or("");
    if !before.ends_with('\n') {
        out.push('\n');
    }
    let trimmed = snippet.trim_matches('\n');
    out.push_str(trimmed);
    out.push('\n');
    if !after.starts_with('\n') {
        out.push('\n');
    }
    out
}


fn parse_single_item(snippet: &str) -> Option<syn::Item> {
    if snippet.trim().is_empty() {
        return None;
    }
    let attempt = syn::parse_file(snippet)
        .or_else(|_| syn::parse_file(&format!("{snippet}\n")));
    let file = attempt.ok()?;
    if file.items.len() == 1 {
        return file.items.into_iter().next();
    }
    None
}


fn promote_private_to_pub_crate(item: &mut syn::Item) {
    struct Promoter {
        in_trait_impl: bool,
    }
    impl VisitMut for Promoter {
        fn visit_item_impl_mut(&mut self, node: &mut syn::ItemImpl) {
            let was = self.in_trait_impl;
            self.in_trait_impl = node.trait_.is_some();
            syn::visit_mut::visit_item_impl_mut(self, node);
            self.in_trait_impl = was;
        }
        fn visit_impl_item_fn_mut(&mut self, m: &mut syn::ImplItemFn) {
            if !self.in_trait_impl && is_private(&m.vis) {
                m.vis = pub_crate();
            }
            syn::visit_mut::visit_impl_item_fn_mut(self, m);
        }
        fn visit_field_mut(&mut self, f: &mut syn::Field) {
            if is_private(&f.vis) {
                f.vis = pub_crate();
            }
            syn::visit_mut::visit_field_mut(self, f);
        }
        fn visit_item_struct_mut(&mut self, s: &mut syn::ItemStruct) {
            if is_private(&s.vis) {
                s.vis = pub_crate();
            }
            syn::visit_mut::visit_item_struct_mut(self, s);
        }
        fn visit_item_enum_mut(&mut self, e: &mut syn::ItemEnum) {
            if is_private(&e.vis) {
                e.vis = pub_crate();
            }
            syn::visit_mut::visit_item_enum_mut(self, e);
        }
        fn visit_item_fn_mut(&mut self, f: &mut syn::ItemFn) {
            if is_private(&f.vis) {
                f.vis = pub_crate();
            }
            syn::visit_mut::visit_item_fn_mut(self, f);
        }
    }
    Promoter { in_trait_impl: false }.visit_item_mut(item);
}


fn pub_crate() -> syn::Visibility {
    syn::parse_quote!(pub (crate))
}


fn resolve_or_create_dst_file(
    registry: &mut NodeRegistry,
    new_module_path: &str,
    src_file: &PathBuf,
) -> Option<PathBuf> {
    let project_root = find_project_root_sync(registry)?;
    let norm_dst = normalize_symbol_id(new_module_path);
    let existing: Option<PathBuf> = registry
        .asts
        .keys()
        .filter(|f| *f != src_file)
        .find(|f| {
            normalize_symbol_id(&module_path_for_file(&project_root, f)) == norm_dst
        })
        .cloned();
    if let Some(f) = existing {
        return Some(f);
    }
    let to_path = ModulePath::from_string(&norm_dst);
    let dst_file = compute_new_file_path(&to_path, &project_root).ok()?;
    if dst_file.exists() {
        let content = std::fs::read_to_string(&dst_file).ok()?;
        let ast = syn::parse_file(&content).ok()?;
        registry.asts.insert(dst_file.clone(), ast);
        registry.sources.insert(dst_file.clone(), Arc::new(content));
        return Some(dst_file);
    }
    let blank: syn::File = syn::parse_quote!();
    registry.asts.insert(dst_file.clone(), blank);
    registry.sources.insert(dst_file.clone(), Arc::new(String::new()));
    Some(dst_file)
}


fn superize_tree(tree: syn::UseTree, src_crate_path: &str) -> syn::UseTree {
    let rendered = quote::quote!(use # tree;).to_string();
    if let Some(new_tree) = try_superize(&tree, src_crate_path, &[]) {
        new_tree
    } else {
        tree
    }
}


fn superize_use_if_from_src(item: syn::ItemUse, src_crate_path: &str) -> syn::ItemUse {
    let rewritten = superize_tree(item.tree.clone(), src_crate_path);
    syn::ItemUse {
        tree: rewritten,
        ..item
    }
}


fn try_superize(
    tree: &syn::UseTree,
    src_crate_path: &str,
    accumulated: &[String],
) -> Option<syn::UseTree> {
    match tree {
        syn::UseTree::Path(p) => {
            let mut acc = accumulated.to_vec();
            acc.push(p.ident.to_string());
            let acc_str = acc.join("::");
            if acc_str == src_crate_path {
                Some(build_super_tree(&p.tree))
            } else if src_crate_path.starts_with(&format!("{}::", acc_str)) {
                let inner = try_superize(&p.tree, src_crate_path, &acc)?;
                Some(
                    syn::UseTree::Path(syn::UsePath {
                        ident: p.ident.clone(),
                        colon2_token: p.colon2_token,
                        tree: Box::new(inner),
                    }),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}


fn validate_item_matches(item: &syn::Item, symbol_id: &str) -> Result<()> {
    let expected = symbol_id.rsplit("::").next().unwrap_or(symbol_id);
    let actual = match item {
        syn::Item::Struct(s) => Some(s.ident.to_string()),
        syn::Item::Enum(e) => Some(e.ident.to_string()),
        syn::Item::Trait(t) => Some(t.ident.to_string()),
        syn::Item::Fn(f) => Some(f.sig.ident.to_string()),
        syn::Item::Type(t) => Some(t.ident.to_string()),
        syn::Item::Const(c) => Some(c.ident.to_string()),
        syn::Item::Mod(m) => Some(m.ident.to_string()),
        _ => None,
    };
    if let Some(actual) = actual {
        if actual != expected {
            anyhow::bail!(
                "cross-file move: span item '{}' does not match expected '{}'", actual,
                expected
            );
        }
    }
    Ok(())
}
