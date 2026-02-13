use super::super::edges::UseEntry;

pub(crate) fn module_segments_from_key(key: &str) -> Vec<String> {
    if key.is_empty() || key == "crate" {
        Vec::new()
    } else {
        key.split("::").map(|s| s.to_string()).collect()
    }
}

pub(crate) fn render_use_item(item: &syn::ItemUse) -> String {
    let body = use_tree_to_string(&item.tree);
    if item.leading_colon.is_some() {
        format!("::{body}")
    } else {
        body
    }
}

fn use_tree_to_string(tree: &syn::UseTree) -> String {
    match tree {
        syn::UseTree::Path(path) => {
            let rest = use_tree_to_string(&path.tree);
            if rest.is_empty() {
                path.ident.to_string()
            } else {
                format!("{}::{}", path.ident, rest)
            }
        }
        syn::UseTree::Name(name) => name.ident.to_string(),
        syn::UseTree::Rename(rename) => format!("{} as {}", rename.ident, rename.rename),
        syn::UseTree::Glob(_) => "*".to_owned(),
        syn::UseTree::Group(group) => {
            let parts = group
                .items
                .iter()
                .map(use_tree_to_string)
                .collect::<Vec<_>>();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

pub(crate) fn flatten_use_tree(
    prefix: Vec<String>,
    tree: &syn::UseTree,
    leading_colon: bool,
    acc: &mut Vec<UseEntry>,
) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut next = prefix;
            next.push(path.ident.to_string());
            flatten_use_tree(next, &path.tree, leading_colon, acc);
        }
        syn::UseTree::Name(name) => {
            let mut segments = prefix;
            segments.push(name.ident.to_string());
            acc.push(UseEntry {
                segments,
                alias: None,
                is_glob: false,
                leading_colon,
            });
        }
        syn::UseTree::Rename(rename) => {
            let mut segments = prefix;
            segments.push(rename.ident.to_string());
            acc.push(UseEntry {
                segments,
                alias: Some(rename.rename.to_string()),
                is_glob: false,
                leading_colon,
            });
        }
        syn::UseTree::Glob(_) => {
            acc.push(UseEntry {
                segments: prefix,
                alias: None,
                is_glob: true,
                leading_colon,
            });
        }
        syn::UseTree::Group(group) => {
            for item in &group.items {
                flatten_use_tree(prefix.clone(), item, leading_colon, acc);
            }
        }
    }
}

pub(crate) fn resolve_use_entry(entry: &UseEntry, module_key: &str) -> Option<(String, String)> {
    let mut segments = entry.segments.clone();
    let mut base = if entry.leading_colon {
        Vec::new()
    } else {
        module_segments_from_key(module_key)
    };
    if let Some(first) = segments.first() {
        match first.as_str() {
            "crate" => {
                base.clear();
                segments.remove(0);
            }
            "self" => {
                base = module_segments_from_key(module_key);
                segments.remove(0);
            }
            "super" => {
                base = module_segments_from_key(module_key);
                while let Some(seg) = segments.first() {
                    if seg == "super" {
                        segments.remove(0);
                        if !base.is_empty() {
                            base.pop();
                        }
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    base.extend(segments);
    if entry.is_glob {
        let module_name = if base.is_empty() {
            module_key.to_owned()
        } else {
            base.join("::")
        };
        if module_name == module_key {
            return None;
        }
        return Some((module_name, "*".to_owned()));
    }
    if base.is_empty() {
        return None;
    }
    let item_name = base.pop()?;
    if item_name == "self" {
        return None;
    }
    let module_name = if base.is_empty() {
        "crate".to_owned()
    } else {
        base.join("::")
    };
    if module_name == module_key {
        return None;
    }
    let imported = if let Some(alias) = &entry.alias {
        format!("{item_name} as {alias}")
    } else {
        item_name
    };
    Some((module_name, imported))
}
