fn remove_orphaned_uses(src_ast: &mut syn::File) {
    let body_tokens: String = src_ast
        .items
        .iter()
        .filter(|i| !matches!(i, syn::Item::Use(_)))
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    src_ast
        .items
        .retain(|item| {
            let syn::Item::Use(use_item) = item else { return true };
            let mut leaves = Vec::new();
            collect_use_leaves(&use_item.tree, &mut leaves);
            leaves.iter().any(|leaf| token_contains_word(&body_tokens, leaf))
                || leaves.iter().any(|leaf| leaf == "self")
        });
}


fn remove_orphaned_uses(src_ast: &mut syn::File) {
    let body_tokens: String = src_ast
        .items
        .iter()
        .filter(|i| !matches!(i, syn::Item::Use(_)))
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    src_ast
        .items
        .retain(|item| {
            let syn::Item::Use(use_item) = item else { return true };
            let mut leaves = Vec::new();
            collect_use_leaves(&use_item.tree, &mut leaves);
            leaves.iter().any(|leaf| token_contains_word(&body_tokens, leaf))
                || leaves.iter().any(|leaf| leaf == "self")
        });
}


fn collect_needed_uses(src_ast: &syn::File, item_tokens: &str) -> Vec<syn::ItemUse> {
    let mut result = Vec::new();
    for src_item in &src_ast.items {
        let syn::Item::Use(use_item) = src_item else { continue };
        let mut leaves: Vec<String> = Vec::new();
        collect_use_leaves(&use_item.tree, &mut leaves);
        if leaves.iter().any(|leaf| token_contains_word(item_tokens, leaf)) {
            result.push(use_item.clone());
        }
    }
    result
}


fn collect_needed_uses(src_ast: &syn::File, item_tokens: &str) -> Vec<syn::ItemUse> {
    let mut result = Vec::new();
    for src_item in &src_ast.items {
        let syn::Item::Use(use_item) = src_item else { continue };
        let mut leaves: Vec<String> = Vec::new();
        collect_use_leaves(&use_item.tree, &mut leaves);
        if leaves.iter().any(|leaf| token_contains_word(item_tokens, leaf)) {
            result.push(use_item.clone());
        }
    }
    result
}


fn collect_use_leaves(tree: &syn::UseTree, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Name(n) => out.push(n.ident.to_string()),
        syn::UseTree::Rename(r) => out.push(r.rename.to_string()),
        syn::UseTree::Glob(_) => out.push("*".to_string()),
        syn::UseTree::Path(p) => collect_use_leaves(&p.tree, out),
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_leaves(item, out);
            }
        }
    }
}


fn collect_use_leaves(tree: &syn::UseTree, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Name(n) => out.push(n.ident.to_string()),
        syn::UseTree::Rename(r) => out.push(r.rename.to_string()),
        syn::UseTree::Glob(_) => out.push("*".to_string()),
        syn::UseTree::Path(p) => collect_use_leaves(&p.tree, out),
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_leaves(item, out);
            }
        }
    }
}


fn token_contains_word(tokens: &str, ident: &str) -> bool {
    if ident == "*" {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = tokens[start..].find(ident) {
        let abs = start + pos;
        let before_ok = abs == 0
            || (!tokens.as_bytes()[abs - 1].is_ascii_alphanumeric()
                && tokens.as_bytes()[abs - 1] != b'_');
        let after = abs + ident.len();
        let after_ok = after >= tokens.len()
            || (!tokens.as_bytes()[after].is_ascii_alphanumeric()
                && tokens.as_bytes()[after] != b'_');
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}


fn absolutize_use(mut item: syn::ItemUse, src_module: &str) -> syn::ItemUse {
    item.tree = absolutize_tree(item.tree, src_module);
    item
}


fn token_contains_word(tokens: &str, ident: &str) -> bool {
    if ident == "*" {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = tokens[start..].find(ident) {
        let abs = start + pos;
        let before_ok = abs == 0
            || (!tokens.as_bytes()[abs - 1].is_ascii_alphanumeric()
                && tokens.as_bytes()[abs - 1] != b'_');
        let after = abs + ident.len();
        let after_ok = after >= tokens.len()
            || (!tokens.as_bytes()[after].is_ascii_alphanumeric()
                && tokens.as_bytes()[after] != b'_');
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}


fn absolutize_tree(tree: syn::UseTree, src_module: &str) -> syn::UseTree {
    match tree {
        syn::UseTree::Path(mut path) => {
            let seg = path.ident.to_string();
            if seg == "super" || seg == "self" {
                let resolved = if seg == "self" {
                    src_module.to_string()
                } else {
                    let mut parts: Vec<&str> = src_module.split("::").collect();
                    parts.pop();
                    parts.join("::")
                };
                let segments: Vec<&str> = resolved.split("::").collect();
                let inner = absolutize_tree(*path.tree, src_module);
                return build_use_path(&segments, inner);
            }
            path.tree = Box::new(absolutize_tree(*path.tree, src_module));
            syn::UseTree::Path(path)
        }
        syn::UseTree::Group(mut group) => {
            group.items = group
                .items
                .into_iter()
                .map(|t| absolutize_tree(t, src_module))
                .collect();
            syn::UseTree::Group(group)
        }
        other => other,
    }
}


fn absolutize_use(mut item: syn::ItemUse, src_module: &str) -> syn::ItemUse {
    item.tree = absolutize_tree(item.tree, src_module);
    item
}


fn absolutize_tree(tree: syn::UseTree, src_module: &str) -> syn::UseTree {
    match tree {
        syn::UseTree::Path(mut path) => {
            let seg = path.ident.to_string();
            if seg == "super" || seg == "self" {
                let resolved = if seg == "self" {
                    src_module.to_string()
                } else {
                    let mut parts: Vec<&str> = src_module.split("::").collect();
                    parts.pop();
                    parts.join("::")
                };
                let segments: Vec<&str> = resolved.split("::").collect();
                let inner = absolutize_tree(*path.tree, src_module);
                return build_use_path(&segments, inner);
            }
            path.tree = Box::new(absolutize_tree(*path.tree, src_module));
            syn::UseTree::Path(path)
        }
        syn::UseTree::Group(mut group) => {
            group.items = group
                .items
                .into_iter()
                .map(|t| absolutize_tree(t, src_module))
                .collect();
            syn::UseTree::Group(group)
        }
        other => other,
    }
}


fn build_use_path(segments: &[&str], inner: syn::UseTree) -> syn::UseTree {
    if segments.is_empty() {
        return inner;
    }
    let (head, tail) = segments.split_first().unwrap();
    let ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident,
        colon2_token: Default::default(),
        tree: Box::new(build_use_path(tail, inner)),
    })
}


fn build_use_path(segments: &[&str], inner: syn::UseTree) -> syn::UseTree {
    if segments.is_empty() {
        return inner;
    }
    let (head, tail) = segments.split_first().unwrap();
    let ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident,
        colon2_token: Default::default(),
        tree: Box::new(build_use_path(tail, inner)),
    })
}


fn remove_orphaned_uses(src_ast: &mut syn::File) {
    let body_tokens: String = src_ast
        .items
        .iter()
        .filter(|i| !matches!(i, syn::Item::Use(_)))
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    src_ast
        .items
        .retain(|item| {
            let syn::Item::Use(use_item) = item else { return true };
            let mut leaves = Vec::new();
            collect_use_leaves(&use_item.tree, &mut leaves);
            leaves.iter().any(|leaf| token_contains_word(&body_tokens, leaf))
                || leaves.iter().any(|leaf| leaf == "self")
        });
}


fn remove_orphaned_uses(src_ast: &mut syn::File) {
    let body_tokens: String = src_ast
        .items
        .iter()
        .filter(|i| !matches!(i, syn::Item::Use(_)))
        .map(|i| i.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    src_ast
        .items
        .retain(|item| {
            let syn::Item::Use(use_item) = item else { return true };
            let mut leaves = Vec::new();
            collect_use_leaves(&use_item.tree, &mut leaves);
            leaves.iter().any(|leaf| token_contains_word(&body_tokens, leaf))
                || leaves.iter().any(|leaf| leaf == "self")
        });
}


fn collect_needed_uses(src_ast: &syn::File, item_tokens: &str) -> Vec<syn::ItemUse> {
    let mut result = Vec::new();
    for src_item in &src_ast.items {
        let syn::Item::Use(use_item) = src_item else { continue };
        let mut leaves: Vec<String> = Vec::new();
        collect_use_leaves(&use_item.tree, &mut leaves);
        if leaves.iter().any(|leaf| token_contains_word(item_tokens, leaf)) {
            result.push(use_item.clone());
        }
    }
    result
}


fn collect_needed_uses(src_ast: &syn::File, item_tokens: &str) -> Vec<syn::ItemUse> {
    let mut result = Vec::new();
    for src_item in &src_ast.items {
        let syn::Item::Use(use_item) = src_item else { continue };
        let mut leaves: Vec<String> = Vec::new();
        collect_use_leaves(&use_item.tree, &mut leaves);
        if leaves.iter().any(|leaf| token_contains_word(item_tokens, leaf)) {
            result.push(use_item.clone());
        }
    }
    result
}


fn collect_use_leaves(tree: &syn::UseTree, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Name(n) => out.push(n.ident.to_string()),
        syn::UseTree::Rename(r) => out.push(r.rename.to_string()),
        syn::UseTree::Glob(_) => out.push("*".to_string()),
        syn::UseTree::Path(p) => collect_use_leaves(&p.tree, out),
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_leaves(item, out);
            }
        }
    }
}


fn collect_use_leaves(tree: &syn::UseTree, out: &mut Vec<String>) {
    match tree {
        syn::UseTree::Name(n) => out.push(n.ident.to_string()),
        syn::UseTree::Rename(r) => out.push(r.rename.to_string()),
        syn::UseTree::Glob(_) => out.push("*".to_string()),
        syn::UseTree::Path(p) => collect_use_leaves(&p.tree, out),
        syn::UseTree::Group(g) => {
            for item in &g.items {
                collect_use_leaves(item, out);
            }
        }
    }
}


fn token_contains_word(tokens: &str, ident: &str) -> bool {
    if ident == "*" {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = tokens[start..].find(ident) {
        let abs = start + pos;
        let before_ok = abs == 0
            || (!tokens.as_bytes()[abs - 1].is_ascii_alphanumeric()
                && tokens.as_bytes()[abs - 1] != b'_');
        let after = abs + ident.len();
        let after_ok = after >= tokens.len()
            || (!tokens.as_bytes()[after].is_ascii_alphanumeric()
                && tokens.as_bytes()[after] != b'_');
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}


fn absolutize_use(mut item: syn::ItemUse, src_module: &str) -> syn::ItemUse {
    item.tree = absolutize_tree(item.tree, src_module);
    item
}


fn token_contains_word(tokens: &str, ident: &str) -> bool {
    if ident == "*" {
        return false;
    }
    let mut start = 0;
    while let Some(pos) = tokens[start..].find(ident) {
        let abs = start + pos;
        let before_ok = abs == 0
            || (!tokens.as_bytes()[abs - 1].is_ascii_alphanumeric()
                && tokens.as_bytes()[abs - 1] != b'_');
        let after = abs + ident.len();
        let after_ok = after >= tokens.len()
            || (!tokens.as_bytes()[after].is_ascii_alphanumeric()
                && tokens.as_bytes()[after] != b'_');
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}


fn absolutize_tree(tree: syn::UseTree, src_module: &str) -> syn::UseTree {
    match tree {
        syn::UseTree::Path(mut path) => {
            let seg = path.ident.to_string();
            if seg == "super" || seg == "self" {
                let resolved = if seg == "self" {
                    src_module.to_string()
                } else {
                    let mut parts: Vec<&str> = src_module.split("::").collect();
                    parts.pop();
                    parts.join("::")
                };
                let segments: Vec<&str> = resolved.split("::").collect();
                let inner = absolutize_tree(*path.tree, src_module);
                return build_use_path(&segments, inner);
            }
            path.tree = Box::new(absolutize_tree(*path.tree, src_module));
            syn::UseTree::Path(path)
        }
        syn::UseTree::Group(mut group) => {
            group.items = group
                .items
                .into_iter()
                .map(|t| absolutize_tree(t, src_module))
                .collect();
            syn::UseTree::Group(group)
        }
        other => other,
    }
}


fn absolutize_use(mut item: syn::ItemUse, src_module: &str) -> syn::ItemUse {
    item.tree = absolutize_tree(item.tree, src_module);
    item
}


fn absolutize_tree(tree: syn::UseTree, src_module: &str) -> syn::UseTree {
    match tree {
        syn::UseTree::Path(mut path) => {
            let seg = path.ident.to_string();
            if seg == "super" || seg == "self" {
                let resolved = if seg == "self" {
                    src_module.to_string()
                } else {
                    let mut parts: Vec<&str> = src_module.split("::").collect();
                    parts.pop();
                    parts.join("::")
                };
                let segments: Vec<&str> = resolved.split("::").collect();
                let inner = absolutize_tree(*path.tree, src_module);
                return build_use_path(&segments, inner);
            }
            path.tree = Box::new(absolutize_tree(*path.tree, src_module));
            syn::UseTree::Path(path)
        }
        syn::UseTree::Group(mut group) => {
            group.items = group
                .items
                .into_iter()
                .map(|t| absolutize_tree(t, src_module))
                .collect();
            syn::UseTree::Group(group)
        }
        other => other,
    }
}


fn build_use_path(segments: &[&str], inner: syn::UseTree) -> syn::UseTree {
    if segments.is_empty() {
        return inner;
    }
    let (head, tail) = segments.split_first().unwrap();
    let ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident,
        colon2_token: Default::default(),
        tree: Box::new(build_use_path(tail, inner)),
    })
}


fn build_use_path(segments: &[&str], inner: syn::UseTree) -> syn::UseTree {
    if segments.is_empty() {
        return inner;
    }
    let (head, tail) = segments.split_first().unwrap();
    let ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident,
        colon2_token: Default::default(),
        tree: Box::new(build_use_path(tail, inner)),
    })
}
