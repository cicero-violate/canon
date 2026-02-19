use proc_macro2::Span;
use std::path::Path;
use syn::visit::{self, Visit};

use crate::rename::alias::{ImportNode, UseKind, VisibilityScope};

use super::super::paths::module_child_path;
use super::super::span::span_to_range;
use super::super::types::SymbolRecord;
use super::super::use_map::{path_to_string, type_path_string};

pub(super) struct SymbolCollector<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbols: Vec<SymbolRecord>,
    current_impl: Option<ImplContext>,
    alias_graph: &'a mut crate::rename::alias::AliasGraph,
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}

impl<'a> SymbolCollector<'a> {
    pub(super) fn new(
        module_path: &'a str,
        file: &'a Path,
        alias_graph: &'a mut crate::rename::alias::AliasGraph,
    ) -> Self {
        Self {
            module_path,
            file,
            symbols: Vec::new(),
            current_impl: None,
            alias_graph,
        }
    }

    pub(super) fn into_symbols(self) -> Vec<SymbolRecord> {
        self.symbols
    }

    fn add_symbol(
        &mut self,
        id: String,
        kind: &str,
        name: &str,
        span: Span,
        docs: Vec<String>,
        attrs: Vec<String>,
    ) {
        let file_path = self.file.to_string_lossy().to_string();
        self.symbols.push(SymbolRecord {
            id,
            kind: kind.to_string(),
            name: name.to_string(),
            module: self.module_path.to_string(),
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

    fn collect_use_tree_root(
        &mut self,
        tree: &syn::UseTree,
        docs: &[String],
        attrs: &[String],
        visibility: VisibilityScope,
    ) {
        self.collect_use_tree_with_prefix(tree, String::new(), docs, attrs, visibility);
    }

    fn collect_use_tree_with_prefix(
        &mut self,
        tree: &syn::UseTree,
        prefix: String,
        docs: &[String],
        attrs: &[String],
        visibility: VisibilityScope,
    ) {
        match tree {
            syn::UseTree::Path(path) => {
                let new_prefix = if prefix.is_empty() {
                    path.ident.to_string()
                } else {
                    format!("{}::{}", prefix, path.ident)
                };
                self.collect_use_tree_with_prefix(&path.tree, new_prefix, docs, attrs, visibility);
            }
            syn::UseTree::Name(name) => {
                // Simple use: use foo::Bar;
                let source_path = if prefix.is_empty() {
                    name.ident.to_string()
                } else {
                    format!("{}::{}", prefix, name.ident)
                };
                let local_name = name.ident.to_string();
                let id = format!("{}::use::{}", self.module_path, local_name);

                self.add_symbol(
                    id.clone(),
                    "use",
                    &local_name,
                    name.ident.span(),
                    docs.to_vec(),
                    attrs.to_vec(),
                );

                // Determine use kind based on visibility
                let kind = if matches!(visibility, VisibilityScope::Public) {
                    UseKind::ReExport
                } else {
                    UseKind::Simple
                };

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name,
                    original_name: None,
                    kind,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Rename(rename) => {
                // Aliased use: use foo::Bar as Baz;
                let source_path = if prefix.is_empty() {
                    rename.ident.to_string()
                } else {
                    format!("{}::{}", prefix, rename.ident)
                };
                let local_name = rename.rename.to_string();
                let original_name = rename.ident.to_string();
                let id = format!("{}::use::{}", self.module_path, local_name);

                self.add_symbol(
                    id.clone(),
                    "use",
                    &local_name,
                    rename.rename.span(),
                    docs.to_vec(),
                    attrs.to_vec(),
                );

                let kind = if matches!(visibility, VisibilityScope::Public) {
                    UseKind::ReExportAliased
                } else {
                    UseKind::Aliased
                };

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name,
                    original_name: Some(original_name),
                    kind,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Glob(_glob) => {
                // Glob import: use foo::*;
                let source_path = prefix;
                let id = format!(
                    "{}::use::*::{}",
                    self.module_path,
                    source_path.replace("::", "_")
                );

                let use_node = ImportNode {
                    id,
                    module_path: self.module_path.to_string(),
                    source_path,
                    local_name: "*".to_string(),
                    original_name: None,
                    kind: UseKind::Glob,
                    visibility: visibility.clone(),
                    file: self.file.to_string_lossy().to_string(),
                };
                self.alias_graph.add_use_node(use_node);
            }
            syn::UseTree::Group(group) => {
                // Group: use foo::{Bar, Baz};
                for item in &group.items {
                    self.collect_use_tree_with_prefix(
                        item,
                        prefix.clone(),
                        docs,
                        attrs,
                        visibility.clone(),
                    );
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for SymbolCollector<'_> {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let mod_id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        let file_path = self.file.to_string_lossy().to_string();
        let is_inline = i.content.is_some();
        self.symbols.push(SymbolRecord {
            id: mod_id,
            kind: "module".to_string(),
            name: i.ident.to_string(),
            module: self.module_path.to_string(),
            file: file_path.clone(),
            declaration_file: Some(file_path.clone()),
            definition_file: if is_inline { Some(file_path.clone()) } else { None },
            span: span_to_range(i.ident.span()),
            alias: None,
            doc_comments: docs,
            attributes: attrs,
        });
        visit::visit_item_mod(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "struct",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for field in &i.fields {
            if let Some(ident) = &field.ident {
                let (field_docs, field_attrs) = Self::extract_docs_and_attrs(&field.attrs);
                let fid = format!("{}::{}", id, ident);
                self.add_symbol(
                    fid,
                    "field",
                    &ident.to_string(),
                    ident.span(),
                    field_docs,
                    field_attrs,
                );
            }
        }
        visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "enum",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for variant in &i.variants {
            let (variant_docs, variant_attrs) = Self::extract_docs_and_attrs(&variant.attrs);
            let vid = format!("{}::{}", id, variant.ident);
            self.add_symbol(
                vid.clone(),
                "variant",
                &variant.ident.to_string(),
                variant.ident.span(),
                variant_docs,
                variant_attrs,
            );
            for field in &variant.fields {
                if let Some(ident) = &field.ident {
                    let (field_docs, field_attrs) = Self::extract_docs_and_attrs(&field.attrs);
                    let fid = format!("{}::{}", vid, ident);
                    self.add_symbol(
                        fid,
                        "field",
                        &ident.to_string(),
                        ident.span(),
                        field_docs,
                        field_attrs,
                    );
                }
            }
        }
        visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id.clone(),
            "trait",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        for item in &i.items {
            if let syn::TraitItem::Fn(method) = item {
                let (method_docs, method_attrs) = Self::extract_docs_and_attrs(&method.attrs);
                let mid = format!("{}::{}", id, method.sig.ident);
                self.add_symbol(
                    mid,
                    "trait_method",
                    &method.sig.ident.to_string(),
                    method.sig.ident.span(),
                    method_docs,
                    method_attrs,
                );
            }
        }
        visit::visit_item_trait(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        let id = module_child_path(self.module_path, i.sig.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "function",
            &i.sig.ident.to_string(),
            i.sig.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_fn(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "type",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "const",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_const(self, i);
    }

    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        let id = module_child_path(self.module_path, i.ident.to_string());
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        self.add_symbol(
            id,
            "static",
            &i.ident.to_string(),
            i.ident.span(),
            docs,
            attrs,
        );
        visit::visit_item_static(self, i);
    }

    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        if let Some(ident) = &i.ident {
            let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
            let id = module_child_path(self.module_path, ident.to_string());
            self.add_symbol(id, "macro", &ident.to_string(), ident.span(), docs, attrs);
        }
        visit::visit_item_macro(self, i);
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
        let visibility = match &i.vis {
            syn::Visibility::Public(_) => VisibilityScope::Public,
            syn::Visibility::Restricted(restricted) => VisibilityScope::Restricted(
                restricted
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            syn::Visibility::Inherited => VisibilityScope::Private,
        };
        self.collect_use_tree_root(&i.tree, &docs, &attrs, visibility);
        visit::visit_item_use(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let prev_impl = self.current_impl.clone();
        let struct_path = type_path_string(&i.self_ty, self.module_path);
        let trait_path = i
            .trait_
            .as_ref()
            .map(|(_, path, _)| path_to_string(path, self.module_path));

        self.current_impl = Some(ImplContext {
            struct_path,
            trait_path,
        });
        visit::visit_item_impl(self, i);
        self.current_impl = prev_impl;
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if let Some(ctx) = &self.current_impl {
            let name = i.sig.ident.to_string();
            let id = if let Some(trait_path) = &ctx.trait_path {
                format!("{} as {}::{}", ctx.struct_path, trait_path, name)
            } else {
                format!("{}::{}", ctx.struct_path, name)
            };
            let (docs, attrs) = Self::extract_docs_and_attrs(&i.attrs);
            self.add_symbol(
                id,
                "method",
                &i.sig.ident.to_string(),
                i.sig.ident.span(),
                docs,
                attrs,
            );
        }
        visit::visit_impl_item_fn(self, i);
    }
}
