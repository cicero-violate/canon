use crate::core::paths::{module_child_path, module_path_for_file};
use crate::core::symbol_id::normalize_symbol_id;
use crate::core::use_map::{path_to_string, type_path_string};
use crate::model::core_span::{span_to_offsets, span_to_range};
use crate::model::types::SpanRange;
use crate::state::{NodeKind, NodeRegistry};
use crate::structured::node_handle;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use syn::spanned::Spanned;

pub(super) struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}

pub(super) type SpanLookup = std::collections::HashMap<PathBuf, std::collections::HashMap<String, SpanOverride>>;

pub(super) struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
    source: Arc<String>,
    span_lookup: Option<&'a SpanLookup>,
    canonical_file: PathBuf,
}

impl<'a> NodeRegistryBuilder<'a> {
    pub(super) fn new(
        project_root: &'a Path,
        file: &'a Path,
        registry: &'a mut NodeRegistry,
        source: Arc<String>,
        span_lookup: Option<&'a SpanLookup>,
    ) -> Self {
        let canonical_file = std::fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
        Self {
            project_root,
            file,
            registry,
            module_path: module_path_for_file(project_root, file),
            item_index: 0,
            parent_path: Vec::new(),
            current_impl: None,
            source,
            span_lookup,
            canonical_file,
        }
    }

    fn register(&mut self, ident: &syn::Ident, kind: NodeKind, span: SpanRange) {
        let id = module_child_path(&self.module_path, ident.to_string());
        self.register_with_id(id, kind, span);
    }

    fn register_with_id(&mut self, id: String, kind: NodeKind, span: SpanRange) {
        let norm_id = normalize_symbol_id(&id);
        let (span, byte_range) = match self
            .span_lookup
            .and_then(|lookup| lookup.get(&self.canonical_file))
            .and_then(|by_symbol| by_symbol.get(&norm_id))
        {
            Some(override_span) => {
                let range = override_span.span.clone();
                let bytes = override_span.byte_range.unwrap_or_else(|| span_to_offsets(&self.source, &range.start, &range.end));
                (range, bytes)
            }
            None => {
                let bytes = span_to_offsets(&self.source, &span.start, &span.end);
                (span, bytes)
            }
        };
        let handle = node_handle(self.file, self.item_index, self.parent_path.clone(), kind, span, byte_range, self.source.clone());
        self.registry.insert_handle(norm_id, handle);
    }

    fn register_use_tree(&mut self, tree: &syn::UseTree, span: &SpanRange) {
        match tree {
            syn::UseTree::Name(name) => {
                let id = format!("{}::use::{}", self.module_path, name.ident);
                self.register_with_id(id, NodeKind::Use, span.clone());
            }
            syn::UseTree::Rename(rename) => {
                let id = format!("{}::use::{}", self.module_path, rename.rename);
                self.register_with_id(id, NodeKind::Use, span.clone());
            }
            syn::UseTree::Path(path) => self.register_use_tree(&path.tree, span),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.register_use_tree(item, span);
                }
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}

impl<'ast> syn::visit::Visit<'ast> for NodeRegistryBuilder<'_> {
    fn visit_file(&mut self, i: &'ast syn::File) {
        self.module_path = module_path_for_file(self.project_root, self.file);
        for (idx, item) in i.items.iter().enumerate() {
            self.item_index = idx;
            self.visit_item(item);
        }
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.register(&i.sig.ident, NodeKind::Fn, span_to_range(i.span()));
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.register(&i.ident, NodeKind::Struct, span_to_range(i.span()));
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.register(&i.ident, NodeKind::Enum, span_to_range(i.span()));
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.register(&i.ident, NodeKind::Trait, span_to_range(i.span()));
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.register(&i.ident, NodeKind::Type, span_to_range(i.span()));
        syn::visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.register(&i.ident, NodeKind::Const, span_to_range(i.span()));
        syn::visit::visit_item_const(self, i);
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.register(&i.ident, NodeKind::Mod, span_to_range(i.span()));
        if let Some((_brace, items)) = &i.content {
            let prev = self.module_path.clone();
            self.module_path = module_child_path(&prev, i.ident.to_string());
            let mod_index = self.item_index;
            self.parent_path.push(mod_index);
            for (idx, item) in items.iter().enumerate() {
                self.item_index = idx;
                self.visit_item(item);
            }
            self.parent_path.pop();
            self.module_path = prev;
        }
    }

    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        let span = span_to_range(i.span());
        self.register_use_tree(&i.tree, &span);
        syn::visit::visit_item_use(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let impl_index = self.item_index;
        let prev_impl = self.current_impl.clone();
        let struct_path = type_path_string(&i.self_ty, &self.module_path);
        let trait_path = i.trait_.as_ref().map(|(_, path, _)| path_to_string(path, &self.module_path));
        self.current_impl = Some(ImplContext { struct_path, trait_path });
        self.parent_path.push(impl_index);
        syn::visit::visit_item_impl(self, i);
        self.parent_path.pop();
        self.current_impl = prev_impl;
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        if let Some(ctx) = &self.current_impl {
            let name = i.sig.ident.to_string();
            let id = if let Some(trait_path) = &ctx.trait_path { format!("{} as {}::{}", ctx.struct_path, trait_path, name) } else { format!("{}::{}", ctx.struct_path, name) };
            self.register_with_id(id, NodeKind::ImplFn, span_to_range(i.span()));
        } else {
            self.register(&i.sig.ident, NodeKind::ImplFn, span_to_range(i.span()));
        }
        syn::visit::visit_impl_item_fn(self, i);
    }
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}
