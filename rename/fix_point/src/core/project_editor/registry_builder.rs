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


#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


pub struct NodeRegistryBuilder<'a> {
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
        let canonical_file = std::fs::canonicalize(file)
            .unwrap_or_else(|_| file.to_path_buf());
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
                let bytes = override_span
                    .byte_range
                    .unwrap_or_else(|| span_to_offsets(
                        &self.source,
                        &range.start,
                        &range.end,
                    ));
                (range, bytes)
            }
            None => {
                let bytes = span_to_offsets(&self.source, &span.start, &span.end);
                (span, bytes)
            }
        };
        let handle = node_handle(
            self.file,
            self.item_index,
            self.parent_path.clone(),
            kind,
            span,
            byte_range,
            self.source.clone(),
        );
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


pub type SpanLookup = std::collections::HashMap<
    PathBuf,
    std::collections::HashMap<String, SpanOverride>,
>;


pub struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}
