use super::config::StructuredEditOptions;


use super::doc_attr::DocAttrPass;


use super::use_tree::UsePathRewritePass;


use crate::alias::ImportNode;


use crate::resolve::ResolverContext;


use algorithms::graph::topological_sort::topological_sort;


use anyhow::Result;


use std::collections::HashMap;


use std::path::Path;


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
    /// Names of passes this pass must run after. Default: no dependencies.
    fn dependencies(&self) -> &'static [&'static str] {
        &[]
    }
}


pub struct StructuredPassRunner {
    passes: Vec<Box<dyn StructuredPass>>,
}
