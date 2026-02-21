use crate::core::paths::{module_child_path, module_path_for_file};
use crate::core::symbol_id::normalize_symbol_id;
use crate::core::use_map::{path_to_string, type_path_string};
use crate::state::{NodeKind, NodeRegistry};
use crate::structured::node_handle;
use std::path::Path;

pub(super) struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
}

impl<'a> NodeRegistryBuilder<'a> {
    pub(super) fn new(project_root: &'a Path, file: &'a Path, registry: &'a mut NodeRegistry) -> Self {
        Self { project_root, file, registry, module_path: module_path_for_file(project_root, file), item_index: 0, parent_path: Vec::new(), current_impl: None }
    }

    fn register(&mut self, ident: &syn::Ident, kind: NodeKind) {
        let id = module_child_path(&self.module_path, ident.to_string());
        self.register_with_id(id, kind);
    }

    fn register_with_id(&mut self, id: String, kind: NodeKind) {
        let handle = node_handle(self.file, self.item_index, self.parent_path.clone(), kind);
        self.registry.insert_handle(normalize_symbol_id(&id), handle);
    }

    fn register_use_tree(&mut self, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Name(name) => {
                let id = format!("{}::use::{}", self.module_path, name.ident);
                self.register_with_id(id, NodeKind::Use);
            }
            syn::UseTree::Rename(rename) => {
                let id = format!("{}::use::{}", self.module_path, rename.rename);
                self.register_with_id(id, NodeKind::Use);
            }
            syn::UseTree::Path(path) => self.register_use_tree(&path.tree),
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.register_use_tree(item);
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
        self.register(&i.sig.ident, NodeKind::Fn);
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.register(&i.ident, NodeKind::Struct);
        syn::visit::visit_item_struct(self, i);
    }

    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.register(&i.ident, NodeKind::Enum);
        syn::visit::visit_item_enum(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.register(&i.ident, NodeKind::Trait);
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.register(&i.ident, NodeKind::Type);
        syn::visit::visit_item_type(self, i);
    }

    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.register(&i.ident, NodeKind::Const);
        syn::visit::visit_item_const(self, i);
    }

    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.register(&i.ident, NodeKind::Mod);
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
        self.register_use_tree(&i.tree);
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
            self.register_with_id(id, NodeKind::ImplFn);
        } else {
            self.register(&i.sig.ident, NodeKind::ImplFn);
        }
        syn::visit::visit_impl_item_fn(self, i);
    }
}

#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}
