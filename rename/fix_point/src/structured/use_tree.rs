use anyhow::Result;


use std::collections::HashMap;


use std::path::Path;


use syn::visit_mut::VisitMut;


use super::config::StructuredEditOptions;


use super::orchestrator::StructuredPass;


use crate::alias::ImportNode;


use crate::resolve::ResolverContext;


use crate::resolve::Resolver;


struct UseAstRewriter<'a> {
    updates: &'a HashMap<String, String>,
    changed: bool,
    resolver: &'a ResolverContext,
}


pub struct UsePathRewritePass {
    path_updates: HashMap<String, String>,
    _alias_nodes: Vec<ImportNode>,
    config: StructuredEditOptions,
    resolver: ResolverContext,
}
