use proc_macro2::Span;
use std::path::Path;

// use crate::alias::{ImportNode, UseKind, VisibilityScope};
use crate::alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};

use super::super::paths::module_child_path;
use super::super::symbol_id::normalize_symbol_id;
use super::super::use_map::{normalize_use_prefix, path_to_string, type_path_string};
use crate::model::core_span::span_to_range;
use crate::model::types::SymbolRecord;

pub(super) struct SymbolCollector<'a> {
    file: &'a Path,
    symbols: Vec<SymbolRecord>,
    alias_graph: &'a mut crate::alias::AliasGraph,
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}

impl<'a> SymbolCollector<'a> {
    pub(super) fn new(file: &'a Path, alias_graph: &'a mut crate::alias::AliasGraph) -> Self {
        Self { file, symbols: Vec::new(), alias_graph }
    }

    pub(super) fn into_symbols(self) -> Vec<SymbolRecord> {
        self.symbols
    }

    /// Iterative top-level entry point — drives a heap stack to avoid OS stack overflow
    /// from deeply nested inline `mod` blocks.
    pub(super) fn walk(&mut self, ast: &syn::File, root_module_path: &str) {
        // Stack entries: (items slice, module_path, current_impl_ctx)
        let mut stack: Vec<(Vec<syn::Item>, String, Option<ImplContext>)> = Vec::new();
        stack.push((ast.items.clone(), root_module_path.to_string(), None));

        while let Some((items, module_path, impl_ctx)) = stack.pop() {
            for item in &items {
                self.process_item(item, &module_path, &impl_ctx, &mut stack);
            }
        }
    }

    fn process_item(&mut self, item: &syn::Item, module_path: &str, impl_ctx: &Option<ImplContext>, stack: &mut Vec<(Vec<syn::Item>, String, Option<ImplContext>)>) {
        match item {
            syn::Item::Mod(i) => {
                let mod_id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                let file_path = self.file.to_string_lossy().to_string();
                let is_inline = i.content.is_some();
                self.symbols.push(SymbolRecord {
                    id: normalize_symbol_id(&mod_id),
                    kind: "module".to_string(),
                    name: i.ident.to_string(),
                    module: module_path.to_string(),
                    file: file_path.clone(),
                    declaration_file: Some(file_path.clone()),
                    definition_file: if is_inline { Some(file_path) } else { None },
                    span: span_to_range(i.ident.span()),
                    alias: None,
                    doc_comments: docs,
                    attributes: attrs,
                });
                // Push inline mod contents onto heap stack instead of recursing
                if let Some((_, inline_items)) = &i.content {
                    stack.push((inline_items.clone(), mod_id, None));
                }
            }
            syn::Item::Struct(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id.clone(), "struct", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
                for field in &i.fields {
                    if let Some(ident) = &field.ident {
                        let (fd, fa) = Self::extract_docs_and_attrs(&field.attrs);
                        self.add_symbol_in(format!("{}::{}", id, ident), "field", &ident.to_string(), ident.span(), fd, fa, module_path);
                    }
                }
            }
            syn::Item::Enum(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id.clone(), "enum", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
                for variant in &i.variants {
                    let (vd, va) = Self::extract_docs_and_attrs(&variant.attrs);
                    let vid = format!("{}::{}", id, variant.ident);
                    self.add_symbol_in(vid.clone(), "variant", &variant.ident.to_string(), variant.ident.span(), vd, va, module_path);
                    for field in &variant.fields {
                        if let Some(ident) = &field.ident {
                            let (fd, fa) = Self::extract_docs_and_attrs(&field.attrs);
                            self.add_symbol_in(format!("{}::{}", vid, ident), "field", &ident.to_string(), ident.span(), fd, fa, module_path);
                        }
                    }
                }
            }
            syn::Item::Trait(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id.clone(), "trait", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
                for ti in &i.items {
                    if let syn::TraitItem::Fn(method) = ti {
                        let (md, ma) = Self::extract_docs_and_attrs(&method.attrs);
                        let mid = format!("{}::{}", id, method.sig.ident);
                        self.add_symbol_in(mid, "trait_method", &method.sig.ident.to_string(), method.sig.ident.span(), md, ma, module_path);
                    }
                }
            }
            syn::Item::Fn(i) => {
                let id = module_child_path(module_path, i.sig.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id, "function", &i.sig.ident.to_string(), i.sig.ident.span(), docs, attrs, module_path);
            }
            syn::Item::Type(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id, "type", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
            }
            syn::Item::Const(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id, "const", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
            }
            syn::Item::Static(i) => {
                let id = module_child_path(module_path, i.ident.to_string());
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                self.add_symbol_in(id, "static", &i.ident.to_string(), i.ident.span(), docs, attrs, module_path);
            }
            syn::Item::Macro(i) => {
                if let Some(ident) = &i.ident {
                    let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                    let id = module_child_path(module_path, ident.to_string());
                    self.add_symbol_in(id, "macro", &ident.to_string(), ident.span(), docs, attrs, module_path);
                }
            }
            syn::Item::Use(i) => {
                let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
                let visibility = match &i.vis {
                    syn::Visibility::Public(_) => VisibilityScope::Public,
                    syn::Visibility::Restricted(r) => VisibilityScope::Restricted(r.path.segments.iter().map(|s| s.ident.to_string()).collect::<Vec<_>>().join("::")),
                    syn::Visibility::Inherited => VisibilityScope::Private,
                };
                self.collect_use_tree(module_path, &i.tree, String::new(), &docs, &attrs, visibility);
            }
            syn::Item::Impl(i) => {
                let struct_path = type_path_string(&i.self_ty, module_path);
                let trait_path = i.trait_.as_ref().map(|(_, path, _)| path_to_string(path, module_path));
                let ctx = ImplContext { struct_path, trait_path };
                // Push impl items as a batch with the impl context
                let impl_items: Vec<syn::Item> = i
                    .items
                    .iter()
                    .filter_map(|ii| {
                        if let syn::ImplItem::Fn(f) = ii {
                            Some(syn::Item::Fn(syn::ItemFn { attrs: f.attrs.clone(), vis: f.vis.clone(), sig: f.sig.clone(), block: Box::new(f.block.clone()) }))
                        } else {
                            None
                        }
                    })
                    .collect();
                // Process impl methods directly (avoid extra stack frame overhead)
                for ii in &i.items {
                    if let syn::ImplItem::Fn(f) = ii {
                        let name = f.sig.ident.to_string();
                        let id = if let Some(tp) = &ctx.trait_path { format!("{} as {}::{}", ctx.struct_path, tp, name) } else { format!("{}::{}", ctx.struct_path, name) };
                        let (docs, attrs) = Self::extract_docs_and_attrs(&f.attrs);
                        self.add_symbol_in(id, "method", &f.sig.ident.to_string(), f.sig.ident.span(), docs, attrs, module_path);
                    }
                }
                let _ = impl_items; // not pushed; methods handled above
            }
            _ => {}
        }
    }

    fn add_symbol_in(&mut self, id: String, kind: &str, name: &str, span: Span, docs: Vec<String>, attrs: Vec<String>, module_path: &str) {
        let id = normalize_symbol_id(&id);
        let file_path = self.file.to_string_lossy().to_string();
        self.symbols.push(SymbolRecord {
            id,
            kind: kind.to_string(),
            name: name.to_string(),
            module: module_path.to_string(),
            file: file_path.clone(),
            declaration_file: None,
            definition_file: Some(file_path),
            span: span_to_range(span),
            alias: None,
            doc_comments: docs,
            attributes: attrs,
        });
    }

    fn extract_docs_and_attrs(attrs: &[syn::Attribute]) -> (Vec<String>, Vec<String>) {
        let mut docs = Vec::new();
        let mut attributes = Vec::new();
        for attr in attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(expr_lit) = &nv.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            docs.push(lit_str.value());
                        }
                    }
                }
            } else {
                attributes.push(quote::quote!(#attr).to_string());
            }
        }
        (docs, attributes)
    }

    /// Iterative use-tree walker — no recursion on `Path` arm.
    fn collect_use_tree(&mut self, module_path: &str, root: &syn::UseTree, root_prefix: String, docs: &[String], attrs: &[String], visibility: VisibilityScope) {
        // Stack entries: (tree_ref, prefix)
        let mut stack: Vec<(&syn::UseTree, String)> = vec![(root, root_prefix)];
        while let Some((tree, prefix)) = stack.pop() {
            match tree {
                syn::UseTree::Path(path) => {
                    let new_prefix = if prefix.is_empty() { path.ident.to_string() } else { format!("{}::{}", prefix, path.ident) };
                    stack.push((&path.tree, new_prefix));
                }
                syn::UseTree::Name(name) => {
                    let mut parts = split_prefix(&prefix);
                    parts.push(name.ident.to_string());
                    let source_path = normalize_use_prefix(&parts, module_path).join("::");
                    let local_name = name.ident.to_string();
                    let id = format!("{}::use::{}", module_path, local_name);
                    self.add_symbol_in(id.clone(), "use", &local_name, name.ident.span(), docs.to_vec(), attrs.to_vec(), module_path);
                    let kind = if matches!(visibility, VisibilityScope::Public) { UseKind::ReExport } else { UseKind::Simple };
                    self.alias_graph.add_use_node(ImportNode {
                        id,
                        module_path: module_path.to_string(),
                        source_path,
                        local_name,
                        original_name: None,
                        kind,
                        visibility: visibility.clone(),
                        file: self.file.to_string_lossy().to_string(),
                    });
                }
                syn::UseTree::Rename(rename) => {
                    let mut parts = split_prefix(&prefix);
                    parts.push(rename.ident.to_string());
                    let source_path = normalize_use_prefix(&parts, module_path).join("::");
                    let local_name = rename.rename.to_string();
                    let original_name = rename.ident.to_string();
                    let id = format!("{}::use::{}", module_path, local_name);
                    self.add_symbol_in(id.clone(), "use", &local_name, rename.rename.span(), docs.to_vec(), attrs.to_vec(), module_path);
                    let kind = if matches!(visibility, VisibilityScope::Public) { UseKind::ReExportAliased } else { UseKind::Aliased };
                    self.alias_graph.add_use_node(ImportNode {
                        id,
                        module_path: module_path.to_string(),
                        source_path,
                        local_name,
                        original_name: Some(original_name),
                        kind,
                        visibility: visibility.clone(),
                        file: self.file.to_string_lossy().to_string(),
                    });
                }
                syn::UseTree::Glob(_) => {
                    let parts = split_prefix(&prefix);
                    let source_path = normalize_use_prefix(&parts, module_path).join("::");
                    let id = format!("{}::use::*::{}", module_path, source_path.replace("::", "_"));
                    self.alias_graph.add_use_node(ImportNode {
                        id,
                        module_path: module_path.to_string(),
                        source_path,
                        local_name: "*".to_string(),
                        original_name: None,
                        kind: UseKind::Glob,
                        visibility: visibility.clone(),
                        file: self.file.to_string_lossy().to_string(),
                    });
                }
                syn::UseTree::Group(group) => {
                    for item in &group.items {
                        stack.push((item, prefix.clone()));
                    }
                }
            }
        }
    }
}

fn split_prefix(prefix: &str) -> Vec<String> {
    if prefix.is_empty() {
        Vec::new()
    } else {
        prefix.split("::").map(|s| s.to_string()).collect()
    }
}
