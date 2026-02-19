use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use super::config::StructuredEditConfig;
use super::doc_attr::DocAttrPass;
use super::use_tree::UseTreePass;
use crate::rename::alias::ImportNode;
use crate::rename::rewrite::RewriteBufferSet;

pub trait StructuredPass {
    fn name(&self) -> &'static str;
    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<bool>;
    fn is_enabled(&self) -> bool {
        true
    }
}

pub struct PassOrchestrator {
    passes: Vec<Box<dyn StructuredPass>>,
}

impl PassOrchestrator {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn StructuredPass>) {
        self.passes.push(pass);
    }

    pub fn run_passes(
        &mut self,
        file: &Path,
        content: &str,
        ast: &syn::File,
        buffers: &mut RewriteBufferSet,
    ) -> Result<Vec<&'static str>> {
        let mut changed = Vec::new();
        for pass in &mut self.passes {
            if !pass.is_enabled() {
                continue;
            }
            if pass.execute(file, content, ast, buffers)? {
                changed.push(pass.name());
            }
        }
        Ok(changed)
    }

    pub fn enabled_count(&self) -> usize {
        self.passes.iter().filter(|p| p.is_enabled()).count()
    }
}

impl Default for PassOrchestrator {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_rename_orchestrator(
    mapping: &HashMap<String, String>,
    path_updates: &HashMap<String, String>,
    alias_nodes: Vec<ImportNode>,
    config: StructuredEditConfig,
) -> PassOrchestrator {
    let mut orchestrator = PassOrchestrator::new();

    orchestrator.add_pass(Box::new(DocAttrPass::new(mapping.clone(), config.clone())));

    orchestrator.add_pass(Box::new(UseTreePass::new(
        path_updates.clone(),
        alias_nodes,
        config,
    )));

    orchestrator
}
