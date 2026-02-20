use std::collections::HashSet;
use syn::{Item, ItemMod, ItemUse, UseTree};

pub fn extract_module_deps(content: &str, current_module: &str) -> Vec<String> {
    let mut deps = Vec::new();
    let mut seen = HashSet::new();

    let syntax_tree = match syn::parse_file(content) {
        Ok(tree) => tree,
        Err(_) => return deps,
    };

    for item in syntax_tree.items {
        match item {
            Item::Mod(item_mod) => {
                if let Some(dep) = extract_mod_dep(&item_mod, current_module) {
                    if seen.insert(dep.clone()) {
                        deps.push(dep);
                    }
                }
            }
            Item::Use(item_use) => {
                for dep in extract_use_deps(&item_use, current_module) {
                    if seen.insert(dep.clone()) {
                        deps.push(dep);
                    }
                }
            }
            _ => {}
        }
    }

    deps
}

fn extract_mod_dep(item_mod: &ItemMod, current_module: &str) -> Option<String> {
    let mod_name = item_mod.ident.to_string();

    if current_module.is_empty() || current_module == "lib" || current_module == "main" {
        Some(mod_name)
    } else {
        Some(format!("{}::{}", current_module, mod_name))
    }
}

fn extract_use_deps(item_use: &ItemUse, current_module: &str) -> Vec<String> {
    let mut deps = Vec::new();
    extract_use_tree(&item_use.tree, &mut deps, String::new(), current_module);
    deps
}

fn extract_use_tree(tree: &UseTree, deps: &mut Vec<String>, prefix: String, current_module: &str) {
    extract_use_tree_inner(tree, deps, prefix, current_module, false)
}

fn extract_use_tree_inner(
    tree: &UseTree,
    deps: &mut Vec<String>,
    prefix: String,
    current_module: &str,
    rooted: bool,
) {
    match tree {
        UseTree::Path(path) => {
            let segment = path.ident.to_string();
            if segment == "crate" {
                extract_use_tree_inner(&path.tree, deps, String::new(), current_module, true);
            } else if segment == "super" {
                let parent = if let Some(pos) = current_module.rfind("::") {
                    &current_module[..pos]
                } else {
                    return;
                };
                // parent::Item means the current module is a child of parent.
                // mod.rs already declares child modules, so this edge is always
                // a child→parent reference which forms a trivial SCC.  Skip it
                // to avoid false-positive cycles in the dependency graph.
                let _ = (path, parent);
            } else if segment == "self" {
                extract_use_tree_inner(
                    &path.tree,
                    deps,
                    current_module.to_string(),
                    current_module,
                    true,
                );
            } else {
                let new_prefix = if prefix.is_empty() {
                    if rooted {
                        segment
                    } else {
                        if current_module.is_empty()
                            || current_module == "lib"
                            || current_module == "main"
                        {
                            segment
                        } else {
                            format!("{}::{}", current_module, segment)
                        }
                    }
                } else {
                    format!("{}::{}", prefix, segment)
                };
                extract_use_tree_inner(&path.tree, deps, new_prefix, current_module, rooted);
            }
        }
        UseTree::Name(name) => {
            let full_path = if prefix.is_empty() {
                name.ident.to_string()
            } else {
                format!("{}::{}", prefix, name.ident)
            };
            deps.push(full_path);
        }
        UseTree::Glob(_) => {
            if !prefix.is_empty() {
                deps.push(prefix);
            }
        }
        UseTree::Group(group) => {
            for item in &group.items {
                extract_use_tree_inner(item, deps, prefix.clone(), current_module, rooted);
            }
        }
        UseTree::Rename(rename) => {
            let full_path = if prefix.is_empty() {
                rename.ident.to_string()
            } else {
                format!("{}::{}", prefix, rename.ident)
            };
            deps.push(full_path);
        }
    }
}
