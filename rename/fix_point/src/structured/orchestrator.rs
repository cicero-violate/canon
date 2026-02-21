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


pub fn create_rename_orchestrator(
    mapping: &HashMap<String, String>,
    path_updates: &HashMap<String, String>,
    alias_nodes: Vec<ImportNode>,
    config: StructuredEditOptions,
    resolver: ResolverContext,
) -> StructuredPassRunner {
    let mut orchestrator = StructuredPassRunner::new();
    orchestrator.add_pass(Box::new(DocAttrPass::new(mapping.clone(), config.clone())));
    orchestrator
        .add_pass(
            Box::new(
                UsePathRewritePass::new(
                    path_updates.clone(),
                    alias_nodes,
                    config,
                    resolver,
                ),
            ),
        );
    orchestrator
}
