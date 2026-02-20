use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use super::config::StructuredEditOptions;
use super::doc_attr::DocAttrPass;
use super::use_tree::UsePathRewritePass;
use crate::rename::alias::ImportNode;
pub trait StructuredPass {
    fn name(&self) -> &'static str;
    fn execute(
        &mut self,
        file: &Path,
        content: &str,
        ast: &mut syn::File,
    ) -> Result<bool>;
    fn is_enabled(&self) -> bool {
        true
    }
}
pub struct StructuredPassRunner {
    passes: Vec<Box<dyn StructuredPass>>,
}
impl StructuredPassRunner {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }
    pub fn add_pass(&mut self, pass: Box<dyn StructuredPass>) {
        self.passes.push(pass);
    }
    pub fn execute_passes(
        &mut self,
        file: &Path,
        content: &str,
        ast: &mut syn::File,
    ) -> Result<Vec<&'static str>> {
        let mut changed = Vec::new();
        for pass in &mut self.passes {
            if !pass.is_enabled() {
                continue;
            }
            if pass.execute(file, content, ast)? {
                changed.push(pass.name());
            }
        }
        Ok(changed)
    }
    pub fn count_enabled_passes(&self) -> usize {
        self.passes.iter().filter(|p| p.is_enabled()).count()
    }
}
impl Default for StructuredPassRunner {
    fn default() -> Self {
        Self::new()
    }
}
pub fn create_rename_orchestrator(
    mapping: &HashMap<String, String>,
    path_updates: &HashMap<String, String>,
    alias_nodes: Vec<ImportNode>,
    config: StructuredEditOptions,
) -> StructuredPassRunner {
    let mut orchestrator = StructuredPassRunner::new();
    orchestrator.add_pass(Box::new(DocAttrPass::new(mapping.clone(), config.clone())));
    orchestrator
        .add_pass(
            Box::new(UsePathRewritePass::new(path_updates.clone(), alias_nodes, config)),
        );
    orchestrator
}
