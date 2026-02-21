use anyhow::Result;


use std::collections::HashMap;


use std::path::Path;


use syn::visit_mut::VisitMut;


use super::config::StructuredEditOptions;


use super::orchestrator::StructuredPass;


use crate::alias::ImportNode;


use crate::resolve::ResolverContext;


use crate::resolve::Resolver;


struct UseAstRewriter<'a> {
    updates: &'a HashMap<String, String>,
    changed: bool,
    resolver: &'a ResolverContext,
}


pub struct UsePathRewritePass {
    path_updates: HashMap<String, String>,
    _alias_nodes: Vec<ImportNode>,
    config: StructuredEditOptions,
    resolver: ResolverContext,
}


fn extract_segment_replacement(
    path: &str,
    new_path: &str,
    segment_index: usize,
) -> String {
    let old_parts: Vec<&str> = path.split("::").collect();
    let new_parts: Vec<&str> = new_path.split("::").collect();
    if segment_index >= new_parts.len() {
        return old_parts.get(segment_index).map(|s| s.to_string()).unwrap_or_default();
    }
    new_parts[segment_index].to_string()
}


fn find_replacement_path(
    path: &str,
    updates: &HashMap<String, String>,
) -> Option<String> {
    if let Some(new_path) = updates.get(path) {
        return Some(new_path.clone());
    }
    let parts: Vec<&str> = path.split("::").collect();
    for i in (1..=parts.len()).rev() {
        let prefix = parts[..i].join("::");
        if let Some(new_prefix) = updates.get(&prefix) {
            if i < parts.len() {
                let suffix = parts[i..].join("::");
                return Some(format!("{}::{}", new_prefix, suffix));
            } else {
                return Some(new_prefix.clone());
            }
        }
    }
    None
}


fn rewrite_use_tree_mut(
    tree: &mut syn::UseTree,
    updates: &HashMap<String, String>,
    changed: &mut bool,
    current_path: &mut Vec<String>,
    resolver_ctx: &ResolverContext,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let segment = p.ident.to_string();
            current_path.push(segment.clone());
            let resolver = Resolver::new(
                &resolver_ctx.module_path,
                resolver_ctx.alias_graph.as_ref(),
                resolver_ctx.symbol_table.as_ref(),
            );
            let path_str = current_path.join("::");
            if let Some(canonical) = resolver.resolve_path_segments(current_path) {
                if let Some(new_path) = find_replacement_path(&canonical, updates) {
                    let replacement = extract_segment_replacement(
                        &canonical,
                        &new_path,
                        current_path.len() - 1,
                    );
                    if replacement != segment {
                        p.ident = syn::Ident::new(&replacement, p.ident.span());
                        *changed = true;
                        current_path.pop();
                        current_path.push(replacement);
                    }
                }
            } else if let Some(new_path) = find_replacement_path(&path_str, updates) {
                let replacement = extract_segment_replacement(
                    &path_str,
                    &new_path,
                    current_path.len() - 1,
                );
                if replacement != segment {
                    p.ident = syn::Ident::new(&replacement, p.ident.span());
                    *changed = true;
                    current_path.pop();
                    current_path.push(replacement);
                }
            }
            rewrite_use_tree_mut(
                &mut p.tree,
                updates,
                changed,
                current_path,
                resolver_ctx,
            );
            current_path.pop();
        }
        syn::UseTree::Name(n) => {
            let segment = n.ident.to_string();
            current_path.push(segment.clone());
            let resolver = Resolver::new(
                &resolver_ctx.module_path,
                resolver_ctx.alias_graph.as_ref(),
                resolver_ctx.symbol_table.as_ref(),
            );
            let path_str = current_path.join("::");
            if let Some(canonical) = resolver.resolve_path_segments(current_path) {
                if let Some(new_path) = find_replacement_path(&canonical, updates) {
                    let replacement = extract_segment_replacement(
                        &canonical,
                        &new_path,
                        current_path.len() - 1,
                    );
                    if replacement != segment {
                        n.ident = syn::Ident::new(&replacement, n.ident.span());
                        *changed = true;
                    }
                }
            } else if let Some(new_path) = find_replacement_path(&path_str, updates) {
                let replacement = extract_segment_replacement(
                    &path_str,
                    &new_path,
                    current_path.len() - 1,
                );
                if replacement != segment {
                    n.ident = syn::Ident::new(&replacement, n.ident.span());
                    *changed = true;
                }
            }
            current_path.pop();
        }
        syn::UseTree::Rename(r) => {
            let segment = r.ident.to_string();
            current_path.push(segment.clone());
            let resolver = Resolver::new(
                &resolver_ctx.module_path,
                resolver_ctx.alias_graph.as_ref(),
                resolver_ctx.symbol_table.as_ref(),
            );
            let path_str = current_path.join("::");
            if let Some(canonical) = resolver.resolve_path_segments(current_path) {
                if let Some(new_path) = find_replacement_path(&canonical, updates) {
                    let replacement = extract_segment_replacement(
                        &canonical,
                        &new_path,
                        current_path.len() - 1,
                    );
                    if replacement != segment {
                        r.ident = syn::Ident::new(&replacement, r.ident.span());
                        r.rename = syn::Ident::new(&replacement, r.rename.span());
                        *changed = true;
                    }
                }
            } else if let Some(new_path) = find_replacement_path(&path_str, updates) {
                let replacement = extract_segment_replacement(
                    &path_str,
                    &new_path,
                    current_path.len() - 1,
                );
                if replacement != segment {
                    r.ident = syn::Ident::new(&replacement, r.ident.span());
                    r.rename = syn::Ident::new(&replacement, r.rename.span());
                    *changed = true;
                }
            }
            current_path.pop();
        }
        syn::UseTree::Glob(_) => {}
        syn::UseTree::Group(g) => {
            for item in &mut g.items {
                rewrite_use_tree_mut(item, updates, changed, current_path, resolver_ctx);
            }
        }
    }
}
