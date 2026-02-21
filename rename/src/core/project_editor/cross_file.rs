use super::use_imports::{absolutize_use, collect_needed_uses, remove_orphaned_uses, token_contains_word};
use syn::visit_mut::VisitMut;
use super::utils::find_project_root_sync;
use super::QueuedOp;
use crate::core::paths::module_path_for_file;
use crate::core::symbol_id::normalize_symbol_id;
use crate::module_path::{compute_new_file_path, ModulePath};
use crate::state::NodeRegistry;
use anyhow::Result;
use quote::ToTokens;
use std::collections::HashSet;
use std::path::PathBuf;

/// Cross-file move pass: for each MoveSymbol op, if source and destination file differ,
/// extract the item from the source AST and append it to the destination AST.
pub(super) fn apply_cross_file_moves(registry: &mut NodeRegistry, changesets: &std::collections::HashMap<PathBuf, Vec<QueuedOp>>) -> Result<HashSet<PathBuf>> {
    let mut touched: HashSet<PathBuf> = HashSet::new();
    struct PendingMove {
        src_file: PathBuf,
        item_index: usize,
        dst_file: PathBuf,
        dst_module_segments: Vec<String>,
    }
    let mut pending: Vec<PendingMove> = Vec::new();
    for (src_file, ops) in changesets {
        for queued in ops {
            if let crate::structured::NodeOp::MoveSymbol { handle, new_module_path, .. } = &queued.op {
                let dst_file = match resolve_or_create_dst_file(registry, new_module_path, src_file) {
                    Some(f) if f != *src_file => f,
                    _ => continue,
                };
                let segments: Vec<String> = new_module_path
                    .trim_start_matches("crate::")
                    .split("::")
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();
                pending.push(PendingMove { src_file: src_file.clone(), item_index: handle.item_index, dst_file, dst_module_segments: segments });
            }
        }
    }
    let mut extra: Vec<PendingMove> = Vec::new();
    for mv in &pending {
        let struct_name = {
            let src_ast = match registry.asts.get(&mv.src_file) {
                Some(a) => a,
                None => continue,
            };
            let item = match src_ast.items.get(mv.item_index) {
                Some(i) => i,
                None => continue,
            };
            match item {
                syn::Item::Struct(s) => s.ident.to_string(),
                syn::Item::Enum(e) => e.ident.to_string(),
                syn::Item::Trait(t) => t.ident.to_string(),
                _ => continue,
            }
        };
        let src_ast = match registry.asts.get(&mv.src_file) {
            Some(a) => a,
            None => continue,
        };
        for (idx, item) in src_ast.items.iter().enumerate() {
            if idx == mv.item_index {
                continue;
            }
            if let syn::Item::Impl(item_impl) = item {
                let self_name = impl_self_type_name(&item_impl.self_ty);
                if self_name.as_deref() == Some(&struct_name) {
                    let already = pending.iter().any(|p| p.src_file == mv.src_file && p.item_index == idx);
                    if !already {
                        extra.push(PendingMove { src_file: mv.src_file.clone(), item_index: idx, dst_file: mv.dst_file.clone(), dst_module_segments: mv.dst_module_segments.clone() });
                    }
                }
            }
        }
    }
    pending.extend(extra);
    pending.sort_by(|a, b| b.item_index.cmp(&a.item_index));
    for mv in pending {
        let mut item = {
            let src_ast = registry
                .asts
                .get_mut(&mv.src_file)
                .ok_or_else(|| anyhow::anyhow!("missing source AST for {}", mv.src_file.display()))?;
            if mv.item_index >= src_ast.items.len() {
                anyhow::bail!("cross-file move: item_index {} out of bounds", mv.item_index);
            }
        src_ast.items.remove(mv.item_index)
        };
        promote_private_to_pub_crate(&mut item);
        // Collect self-type names from impl blocks before item is consumed,
        // used by Gap 7b to inject `use super::Type` in the dst file.
        let impl_self_names: Vec<String> = match &item {
            syn::Item::Impl(impl_item) => {
                impl_self_type_name(&impl_item.self_ty).into_iter().collect()
            }
            _ => vec![],
        };
        let needed_uses: Vec<syn::ItemUse> = {
            let src_ast = registry
                .asts
                .get(&mv.src_file)
                .ok_or_else(|| anyhow::anyhow!("missing source AST for {}", mv.src_file.display()))?;
            let item_tokens = item.to_token_stream().to_string();
            collect_needed_uses(src_ast, &item_tokens)
        };
        let needed_uses: Vec<syn::ItemUse> = {
            let project_root = find_project_root_sync(registry).unwrap_or_else(|| PathBuf::from("."));
            let src_module = module_path_for_file(&project_root, &mv.src_file);
            needed_uses.into_iter().map(|u| absolutize_use(u, &src_module)).collect()
        };
        // Gap 7: rewrite absolute crate:: imports inside the moved item that
        // point back at the source module → use super:: when dst is a direct
        // child of src (e.g. crate::occurrence::Foo → super::Foo).
        // For deeper moves, the absolute crate:: path is kept as-is.
        let needed_uses: Vec<syn::ItemUse> = {
            let project_root = find_project_root_sync(registry).unwrap_or_else(|| PathBuf::from("."));
            let src_module = module_path_for_file(&project_root, &mv.src_file);
            let dst_module = mv.dst_module_segments.join("::");
            let dst_is_direct_child = {
                let dst_full = format!("crate::{}", dst_module);
                let src_full = format!("crate::{}", src_module.trim_start_matches("crate::"));
                dst_full.starts_with(&format!("{}::", src_full))
                    && dst_full[src_full.len() + 2..].split("::").count() == 1
            };
            if dst_is_direct_child {
                let src_crate = format!("crate::{}", src_module.trim_start_matches("crate::"));
                needed_uses
                    .into_iter()
                    .map(|u| superize_use_if_from_src(u, &src_crate))
                    .collect()
            } else {
                needed_uses
            }
        };
        touched.insert(mv.src_file.clone());
        {
            let src_ast = registry
                .asts
                .get_mut(&mv.src_file)
                .ok_or_else(|| anyhow::anyhow!("missing source AST for {}", mv.src_file.display()))?;
            remove_orphaned_uses(src_ast);
        }
        {
            let symbol_name = match &item {
                syn::Item::Struct(s) => Some(s.ident.to_string()),
                syn::Item::Enum(e) => Some(e.ident.to_string()),
                syn::Item::Trait(t) => Some(t.ident.to_string()),
                _ => None,
            };
            if let Some(name) = symbol_name {
                let src_ast = registry
                    .asts
                    .get(&mv.src_file)
                    .ok_or_else(|| anyhow::anyhow!("missing source AST for {}", mv.src_file.display()))?;
                let body_tokens: String = src_ast
                    .items
                    .iter()
                    .filter(|i| !matches!(i, syn::Item::Use(_)))
                    .map(|i| i.to_token_stream().to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                if token_contains_word(&body_tokens, &name) {
                    let mut use_path = mv.dst_module_segments.join("::");
                    if !use_path.starts_with("crate") {
                        use_path = format!("crate::{}", use_path);
                    }
                    let use_str = format!("use {}::{};", use_path, name);
                    if let Ok(parsed) = syn::parse_str::<syn::ItemUse>(&use_str) {
                        let already = src_ast.items.iter().any(|i| {
                            if let syn::Item::Use(u) = i {
                                u.to_token_stream().to_string() == parsed.to_token_stream().to_string()
                            } else {
                                false
                            }
                        });
                        if !already {
                            let src_ast = registry.asts.get_mut(&mv.src_file).ok_or_else(|| anyhow::anyhow!("missing source AST"))?;
                            src_ast.items.insert(0, syn::Item::Use(parsed));
                        }
                    }
                }
            }
        }
        // Gap 5b: inject `use dst::Name;` for co-moved free functions still
        // referenced in the source file (e.g. is_ignored_dir).
        {
            // First: immutable analysis phase
            let (body_tokens, exported_names) = {
                let src_ast = registry
                    .asts
                    .get(&mv.src_file)
                    .ok_or_else(|| anyhow::anyhow!("missing source AST for {}", mv.src_file.display()))?;

                let body_tokens: String = src_ast
                    .items
                    .iter()
                    .filter(|i| !matches!(i, syn::Item::Use(_)))
                    .map(|i| i.to_token_stream().to_string())
                    .collect::<Vec<_>>()
                    .join(" ");

                let mut exported_names: Vec<String> = Vec::new();
                if let syn::Item::Impl(impl_item) = &item {
                    for impl_item_inner in &impl_item.items {
                        if let syn::ImplItem::Fn(f) = impl_item_inner {
                            exported_names.push(f.sig.ident.to_string());
                        }
                    }
                }
                if let syn::Item::Fn(f) = &item {
                    exported_names.push(f.sig.ident.to_string());
                }

                (body_tokens, exported_names)
            };

            // Second: mutation phase
            let mut use_path = mv.dst_module_segments.join("::");
            if !use_path.starts_with("crate") {
                use_path = format!("crate::{}", use_path);
            }

            for name in exported_names {
                if token_contains_word(&body_tokens, &name) {
                    let use_str = format!("use {}::{};", use_path, name);
                    if let Ok(parsed) = syn::parse_str::<syn::ItemUse>(&use_str) {
                        let src_ast = registry
                            .asts
                            .get_mut(&mv.src_file)
                            .ok_or_else(|| anyhow::anyhow!("missing source AST"))?;

                        let already = src_ast.items.iter().any(|i| {
                            if let syn::Item::Use(u) = i {
                                u.to_token_stream().to_string() == parsed.to_token_stream().to_string()
                            } else { false }
                        });

                        if !already {
                            src_ast.items.insert(0, syn::Item::Use(parsed));
                        }
                    }
                }
            }
        }

        // Compute relation before mutable borrow
        let project_root = find_project_root_sync(registry).unwrap_or_else(|| PathBuf::from("."));
        let src_module = module_path_for_file(&project_root, &mv.src_file);
        let dst_full = format!("crate::{}", mv.dst_module_segments.join("::"));
        let src_full = format!("crate::{}", src_module.trim_start_matches("crate::"));
        let is_direct_child = dst_full.starts_with(&format!("{}::", src_full))
            && dst_full[src_full.len() + 2..].split("::").count() == 1;

        {
            let dst_ast = registry
                .asts
                .get_mut(&mv.dst_file)
                .ok_or_else(|| anyhow::anyhow!("missing dest AST for {}", mv.dst_file.display()))?;

            let segs: Vec<&str> = mv.dst_module_segments.iter().map(|s| s.as_str()).collect();
            match find_mod_container_mut(dst_ast, &segs) {
                Some(container) => container.push(item),
                None => dst_ast.items.push(item),
            }

            if is_direct_child {
               for self_name in &impl_self_names {
                    // Don't inject `use super::X` if X is already defined
                    // in the dst file (i.e. the struct itself was also moved there).
                    let already_defined = dst_ast.items.iter().any(|i| match i {
                        syn::Item::Struct(s) => s.ident == self_name.as_str(),
                        syn::Item::Enum(e) => e.ident == self_name.as_str(),
                        _ => false,
                    });
                    if already_defined { continue; }
                   let use_str = format!("use super::{};", self_name);
                    if let Ok(parsed) = syn::parse_str::<syn::ItemUse>(&use_str) {
                        let already = dst_ast.items.iter().any(|i| {
                            if let syn::Item::Use(u) = i {
                                u.to_token_stream().to_string() == parsed.to_token_stream().to_string()
                            } else { false }
                        });
                        if !already {
                            dst_ast.items.insert(0, syn::Item::Use(parsed));
                        }
                    }
                }
            }

            let existing: HashSet<String> = dst_ast
                .items
                .iter()
                .filter_map(|i| {
                    if let syn::Item::Use(u) = i {
                        Some(u.to_token_stream().to_string())
                    } else {
                        None
                    }
                })
                .collect();

            for use_item in needed_uses {
                let rendered = use_item.to_token_stream().to_string();
                if !existing.contains(&rendered) {
                    dst_ast.items.insert(0, syn::Item::Use(use_item));
                }
            }
        }

        touched.insert(mv.dst_file);
    }
    Ok(touched)
}

fn impl_self_type_name(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

fn find_mod_container_mut<'a>(ast: &'a mut syn::File, segments: &[&str]) -> Option<&'a mut Vec<syn::Item>> {
    fn recurse<'a>(items: &'a mut Vec<syn::Item>, segments: &[&str]) -> Option<&'a mut Vec<syn::Item>> {
        if segments.is_empty() {
            return Some(items);
        }
        let (head, tail) = segments.split_first()?;
        for item in items.iter_mut() {
            if let syn::Item::Mod(m) = item {
                if m.ident == head {
                    if let Some((_, inner)) = m.content.as_mut() {
                        return recurse(inner, tail);
                    }
                }
            }
        }
        None
    }
    recurse(&mut ast.items, segments)
}

fn resolve_dst_file(registry: &NodeRegistry, new_module_path: &str, src_file: &PathBuf) -> Option<PathBuf> {
    let project_root = registry.asts.keys().next().and_then(|f| {
        let mut cur = f.parent()?.to_path_buf();
        loop {
            if cur.join("Cargo.toml").exists() {
                return Some(cur);
            }
            if !cur.pop() {
                return None;
            }
        }
    })?;
    let norm_dst = normalize_symbol_id(new_module_path);
    for file in registry.asts.keys() {
        if file == src_file {
            continue;
        }
        let module = normalize_symbol_id(&module_path_for_file(&project_root, file));
        if module == norm_dst {
            return Some(file.clone());
        }
    }
    None
}

fn resolve_or_create_dst_file(registry: &mut NodeRegistry, new_module_path: &str, src_file: &PathBuf) -> Option<PathBuf> {
    let project_root = find_project_root_sync(registry)?;
    let norm_dst = normalize_symbol_id(new_module_path);
    let existing: Option<PathBuf> = registry
        .asts
        .keys()
        .filter(|f| *f != src_file)
        .find(|f| normalize_symbol_id(&module_path_for_file(&project_root, f)) == norm_dst)
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
        return Some(dst_file);
    }
    let blank: syn::File = syn::parse_quote!();
    registry.asts.insert(dst_file.clone(), blank);
    Some(dst_file)
}

/// After apply(), scan changesets to find which dst files were newly synthesized
/// (i.e. not present on disk). Returns (path, module_id) pairs for commit() to write.
pub(super) fn collect_new_files(registry: &NodeRegistry, changesets: &std::collections::HashMap<PathBuf, Vec<QueuedOp>>) -> Vec<(PathBuf, String)> {
    let project_root = match find_project_root_sync(registry) {
        Some(r) => r,
        None => return Vec::new(),
    };
    let mut seen: HashSet<PathBuf> = HashSet::new();
    let mut result = Vec::new();
    for ops in changesets.values() {
        for queued in ops {
            if let crate::structured::NodeOp::MoveSymbol { handle: _, new_module_path, .. } = &queued.op {
                let norm_dst = normalize_symbol_id(new_module_path);
                let to_path = ModulePath::from_string(&norm_dst);
                let dst_file = match compute_new_file_path(&to_path, &project_root) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !dst_file.exists() && registry.asts.contains_key(&dst_file) && seen.insert(dst_file.clone()) {
                    result.push((dst_file, norm_dst));
                }
            }
        }
    }
    result
}

/// Gap 6: Promote `Inherited` / `pub(self)` visibility → `pub(crate)` on cross-module move.
///
/// Inherent impl methods (`impl Type { fn foo() }`) are promoted.
/// Trait impl methods (`impl Trait for Type { fn foo() }`) are NOT touched —
/// their visibility is dictated by the trait contract.
fn promote_private_to_pub_crate(item: &mut syn::Item) {
    struct Promoter { in_trait_impl: bool }

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

fn is_private(vis: &syn::Visibility) -> bool {
    match vis {
        syn::Visibility::Inherited => true,
        syn::Visibility::Restricted(r) => {
            // pub(self) is the same as private
            r.path.is_ident("self")
        }
        _ => false,
    }
}

fn pub_crate() -> syn::Visibility {
    syn::parse_quote!(pub(crate))
}

/// Gap 7: if a collected use item resolves into `src_crate_path`, rewrite it
/// to use `super::` — valid when dst is a direct child of src.
fn superize_use_if_from_src(item: syn::ItemUse, src_crate_path: &str) -> syn::ItemUse {
    let rewritten = superize_tree(item.tree.clone(), src_crate_path);
    syn::ItemUse { tree: rewritten, ..item }
}

fn superize_tree(tree: syn::UseTree, src_crate_path: &str) -> syn::UseTree {
    // Reconstruct the full path this tree starts with and check if it
    // begins with src_crate_path. If so, strip that prefix and prepend super::.
    let rendered = quote::quote!(use #tree;).to_string();
    // Walk: if the tree is a Path chain starting with "crate" that matches
    // src_crate_path, rewrite by building super:: prefix.
    if let Some(new_tree) = try_superize(&tree, src_crate_path, &[]) {
        new_tree
    } else {
        tree
    }
}

/// Recursively walk a UseTree. `accumulated` tracks segments seen so far.
/// When the accumulated path equals src_crate_path, replace with super:: tail.
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
                // The path up to here matches src. Replace with super:: + tail.
                Some(build_super_tree(&p.tree))
            } else if src_crate_path.starts_with(&format!("{}::", acc_str)) {
                // Still a prefix — keep descending.
                let inner = try_superize(&p.tree, src_crate_path, &acc)?;
                Some(syn::UseTree::Path(syn::UsePath {
                    ident: p.ident.clone(),
                    colon2_token: p.colon2_token,
                    tree: Box::new(inner),
                }))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Wrap a UseTree tail with `super::`.
fn build_super_tree(tail: &syn::UseTree) -> syn::UseTree {
    let super_ident = syn::Ident::new("super", proc_macro2::Span::call_site());
    syn::UseTree::Path(syn::UsePath {
        ident: super_ident,
        colon2_token: syn::token::PathSep::default(),
        tree: Box::new(tail.clone()),
    })
}
