use std::collections::HashMap;
pub(crate) fn build_use_map(ast: &syn::File, module_path: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for item in &ast.items {
        let syn::Item::Use(u) = item else { continue };
        let mut prefix = Vec::new();
        if u.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        use_tree_to_map(&u.tree, &mut prefix, module_path, &mut map);
    }
    map
}

fn use_tree_to_map(tree: &syn::UseTree, prefix: &mut Vec<String>, module_path: &str, map: &mut HashMap<String, String>) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            use_tree_to_map(&p.tree, prefix, module_path, map);
            prefix.pop();
        }
        syn::UseTree::Name(name) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(name.ident.to_string());
            map.insert(name.ident.to_string(), full.join("::"));
        }
        syn::UseTree::Rename(rename) => {
            let mut full = normalize_use_prefix(prefix, module_path);
            full.push(rename.ident.to_string());
            map.insert(rename.rename.to_string(), full.join("::"));
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                use_tree_to_map(item, prefix, module_path, map);
            }
        }
        syn::UseTree::Glob(_) => {}
    }
}

pub(crate) fn normalize_use_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    if prefix.first().map(|s| s.as_str()) == Some("crate") {
        return prefix.to_vec();
    }
    if prefix.first().map(|s| s.as_str()) == Some("self") || prefix.first().map(|s| s.as_str()) == Some("super") {
        return resolve_relative_prefix(prefix, module_path);
    }
    let mut out: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    out.extend(prefix.iter().cloned());
    out
}

fn resolve_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    let mut module_parts: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    let mut idx = 0usize;
    while idx < prefix.len() && prefix[idx] == "super" {
        if module_parts.len() > 1 {
            module_parts.pop();
        }
        idx += 1;
    }
    if idx < prefix.len() && prefix[idx] == "self" {
        idx += 1;
    }
    let mut out = module_parts;
    out.extend(prefix[idx..].iter().cloned());
    out
}

pub(crate) fn path_to_string(path: &syn::Path, module_path: &str) -> String {
    let segments: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
    if segments.first().map(|s| s.as_str()) == Some("crate") {
        segments.join("::")
    } else if segments.first().map(|s| s.as_str()) == Some("self") || segments.first().map(|s| s.as_str()) == Some("super") {
        let rel = resolve_relative_prefix(&segments, module_path);
        rel.join("::")
    } else {
        format!("{}::{}", module_path, segments.join("::"))
    }
}

pub(crate) fn type_path_string(ty: &syn::Type, module_path: &str) -> String {
    if let syn::Type::Path(tp) = ty {
        path_to_string(&tp.path, module_path)
    } else {
        format!("{}::{}", module_path, "Self")
    }
}
