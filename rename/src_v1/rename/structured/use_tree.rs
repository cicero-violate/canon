use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use syn::visit_mut::VisitMut;

use crate::rename::alias::ImportNode;
use crate::rename::core::find_replacement_path;
use crate::rename::rewrite::{RewriteBufferSet, SourceTextEdit};

use super::config::StructuredEditConfig;
use super::orchestrator::StructuredPass;

pub struct UseTreePass {
    path_updates: HashMap<String, String>,
    _alias_nodes: Vec<ImportNode>, // retained for API parity
    config: StructuredEditConfig,
}

impl UseTreePass {
    pub fn new(
        path_updates: HashMap<String, String>,
        alias_nodes: Vec<ImportNode>,
        config: StructuredEditConfig,
    ) -> Self {
        Self {
            path_updates,
            _alias_nodes: alias_nodes,
            config,
        }
    }
}

impl StructuredPass for UseTreePass {
    fn name(&self) -> &'static str {
        "use_tree"
    }

    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<bool> {
        if !self.config.use_statements_enabled() {
            return Ok(false);
        }

        if self.path_updates.is_empty() {
            return Ok(false);
        }

        let mut new_ast = ast.clone();

        let mut rewriter = UseAstRewriter {
            updates: &self.path_updates,
            changed: false,
        };

        rewriter.visit_file_mut(&mut new_ast);

        if !rewriter.changed {
            return Ok(false);
        }

        let rendered = prettyplease::unparse(&new_ast);

        buffers.queue_edits(
            file,
            content,
            [SourceTextEdit {
                start: 0,
                end: content.len(),
                text: rendered,
            }],
        )?;

        Ok(true)
    }

    fn is_enabled(&self) -> bool {
        self.config.use_statements_enabled()
    }
}

struct UseAstRewriter<'a> {
    updates: &'a HashMap<String, String>,
    changed: bool,
}

impl<'a> VisitMut for UseAstRewriter<'a> {
    fn visit_item_use_mut(&mut self, node: &mut syn::ItemUse) {
        rewrite_use_tree_mut(&mut node.tree, self.updates, &mut self.changed);
        syn::visit_mut::visit_item_use_mut(self, node);
    }
}

fn rewrite_use_tree_mut(
    tree: &mut syn::UseTree,
    updates: &HashMap<String, String>,
    changed: &mut bool,
) {
    match tree {
        syn::UseTree::Path(p) => {
            let full = p.ident.to_string();

            if let Some(new_ident) = updates.get(&full) {
                p.ident = syn::Ident::new(new_ident, p.ident.span());
                *changed = true;
            }

            rewrite_use_tree_mut(&mut p.tree, updates, changed);
        }
        syn::UseTree::Name(n) => {
            let ident = n.ident.to_string();

            if let Some(new_ident) = updates.get(&ident) {
                n.ident = syn::Ident::new(new_ident, n.ident.span());
                *changed = true;
            }
        }
        syn::UseTree::Rename(r) => {
            let ident = r.ident.to_string();

            if let Some(new_ident) = updates.get(&ident) {
                r.ident = syn::Ident::new(new_ident, r.ident.span());
                *changed = true;
            }
        }
        syn::UseTree::Glob(_) => {}
        syn::UseTree::Group(g) => {
            for item in &mut g.items {
                rewrite_use_tree_mut(item, updates, changed);
            }
        }
    }
}
