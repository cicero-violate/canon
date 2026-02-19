use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use syn::visit::Visit;

use crate::fs;
use crate::rename::core::oracle::StructuralEditOracle;
use crate::rustc_integration::frontends::rustc::RustcFrontend;
use crate::rustc_integration::multi_capture::capture_project;
use crate::rustc_integration::project::CargoProject;
use crate::rename::core::symbol_id::normalize_symbol_id;
use crate::rename::structured::{FieldMutation, NodeOp, node_handle};
use crate::rename::core::paths::{module_child_path, module_path_for_file};
use crate::rename::core::use_map::{path_to_string, type_path_string};
use crate::state::{NodeKind, NodeRegistry};

#[derive(Debug, Clone)]
pub struct EditConflict {
    pub symbol_id: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct ChangeReport {
    pub touched_files: Vec<PathBuf>,
    pub conflicts: Vec<EditConflict>,
}

pub struct ProjectEditor {
    pub registry: NodeRegistry,
    pub changesets: HashMap<PathBuf, Vec<QueuedOp>>,
    pub oracle: Box<dyn StructuralEditOracle>,
}

#[derive(Clone)]
pub struct QueuedOp {
    pub symbol_id: String,
    pub op: NodeOp,
}

impl ProjectEditor {
    pub fn load(project: &Path, oracle: Box<dyn StructuralEditOracle>) -> Result<Self> {
        let files = fs::collect_rs_files(project)?;
        let mut registry = NodeRegistry::new();

        for file in files {
            let content = std::fs::read_to_string(&file)?;
            let ast = syn::parse_file(&content)
                .with_context(|| format!("Failed to parse {}", file.display()))?;
            let mut builder = NodeRegistryBuilder::new(project, &file, &mut registry);
            builder.visit_file(&ast);
            registry.insert_ast(file, ast);
        }

        Ok(Self {
            registry,
            changesets: HashMap::new(),
            oracle,
        })
    }

    pub fn load_with_rustc(project: &Path) -> Result<Self> {
        let cargo = CargoProject::from_entry(project)?;
        let frontend = RustcFrontend::new();
        let artifacts = capture_project(&frontend, &cargo, &[]).with_context(|| {
            format!("rustc capture failed for {}", project.display())
        })?;
        let oracle = Box::new(cargo.with_snapshot(artifacts.snapshot));
        Self::load(project, oracle)
    }

    pub fn queue(&mut self, symbol_id: &str, op: NodeOp) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = match &op {
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. } => Some(handle),
            NodeOp::ReorderItems { .. } => None,
        };

        if let Some(handle) = handle {
            let exists = self.registry.handles.get(&norm);
            if exists.is_none() {
                self.registry.insert_handle(norm.clone(), handle.clone());
            }
        }

        let file = match &op {
            NodeOp::ReplaceNode { handle, .. }
            | NodeOp::InsertBefore { handle, .. }
            | NodeOp::InsertAfter { handle, .. }
            | NodeOp::DeleteNode { handle }
            | NodeOp::MutateField { handle, .. } => handle.file.clone(),
            NodeOp::ReorderItems { file, .. } => file.clone(),
        };

        self.changesets
            .entry(file)
            .or_default()
            .push(QueuedOp {
                symbol_id: norm,
                op,
            });
        Ok(())
    }

    pub fn queue_by_id(&mut self, symbol_id: &str, mutation: FieldMutation) -> Result<()> {
        let norm = normalize_symbol_id(symbol_id);
        let handle = self
            .registry
            .handles
            .get(&norm)
            .cloned()
            .with_context(|| format!("no handle found for {symbol_id}"))?;
        let op = NodeOp::MutateField { handle, mutation };
        self.queue(&norm, op)
    }

    pub fn apply(&mut self) -> Result<ChangeReport> {
        let mut touched_files = Vec::new();
        let handle_snapshot = self.registry.handles.clone();
        for (file, ops) in &self.changesets {
            let ast = self
                .registry
                .asts
                .get_mut(file)
                .with_context(|| format!("missing AST for {}", file.display()))?;
            for queued in ops {
                apply_node_op(ast, &handle_snapshot, &queued.symbol_id, &queued.op)
                    .with_context(|| format!("failed to apply {}", queued.symbol_id))?;
            }
            touched_files.push(file.clone());
        }
        let conflicts = self.validate()?;
        Ok(ChangeReport {
            touched_files,
            conflicts,
        })
    }

    pub fn validate(&self) -> Result<Vec<EditConflict>> {
        let mut conflicts = Vec::new();
        for (symbol_id, _handle) in &self.registry.handles {
            if self.oracle.is_macro_generated(symbol_id) {
                conflicts.push(EditConflict {
                    symbol_id: symbol_id.clone(),
                    reason: "symbol generated by macro".to_string(),
                });
            }
        }
        for ops in self.changesets.values() {
            for queued in ops {
                if let NodeOp::MutateField { mutation, .. } = &queued.op {
                    if matches!(mutation, FieldMutation::ReplaceSignature(_)) {
                        let impacts = self.oracle.impact_of(&queued.symbol_id);
                        if !impacts.is_empty() {
                            conflicts.push(EditConflict {
                                symbol_id: queued.symbol_id.clone(),
                                reason: "signature change may require updating call sites"
                                    .to_string(),
                            });
                        }
                    }
                }
            }
        }
        Ok(conflicts)
    }

    pub fn commit(&self) -> Result<Vec<PathBuf>> {
        let mut written = Vec::new();
        for file in self.changesets.keys() {
            let ast = self
                .registry
                .asts
                .get(file)
                .with_context(|| format!("missing AST for {}", file.display()))?;
            let rendered = crate::rename::structured::render_file(ast);
            std::fs::write(file, rendered)?;
            written.push(file.clone());
        }
        Ok(written)
    }

    pub fn preview(&self) -> Result<String> {
        Ok(format!("{} files touched", self.changesets.len()))
    }
}

mod ops;

use ops::apply_node_op;

struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
}

impl<'a> NodeRegistryBuilder<'a> {
    fn new(project_root: &'a Path, file: &'a Path, registry: &'a mut NodeRegistry) -> Self {
        Self {
            project_root,
            file,
            registry,
            module_path: module_path_for_file(project_root, file),
            item_index: 0,
            parent_path: Vec::new(),
            current_impl: None,
        }
    }

    fn register(&mut self, ident: &syn::Ident, kind: NodeKind) {
        let id = module_child_path(&self.module_path, ident.to_string());
        self.register_with_id(id, kind);
    }

    fn register_with_id(&mut self, id: String, kind: NodeKind) {
        let handle = node_handle(
            self.file,
            self.item_index,
            self.parent_path.clone(),
            kind,
        );
        self.registry
            .insert_handle(normalize_symbol_id(&id), handle);
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
        let trait_path = i
            .trait_
            .as_ref()
            .map(|(_, path, _)| path_to_string(path, &self.module_path));
        self.current_impl = Some(ImplContext {
            struct_path,
            trait_path,
        });
        self.parent_path.push(impl_index);
        syn::visit::visit_item_impl(self, i);
        self.parent_path.pop();
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
