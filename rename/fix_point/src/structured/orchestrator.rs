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
        let n = self.passes.len();
        let name_to_idx: HashMap<&str, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name(), i))
            .collect();
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, pass) in self.passes.iter().enumerate() {
            for &dep in pass.dependencies() {
                if let Some(&j) = name_to_idx.get(dep) {
                    adj[j].push(i);
                }
            }
        }
        let order = topological_sort(&adj);
        for idx in order {
            let pass = &mut self.passes[idx];
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
        let n = self.passes.len();
        let name_to_idx: HashMap<&str, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name(), i))
            .collect();
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, pass) in self.passes.iter().enumerate() {
            for &dep in pass.dependencies() {
                if let Some(&j) = name_to_idx.get(dep) {
                    adj[j].push(i);
                }
            }
        }
        let order = topological_sort(&adj);
        for idx in order {
            let pass = &mut self.passes[idx];
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
        let n = self.passes.len();
        let name_to_idx: HashMap<&str, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name(), i))
            .collect();
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, pass) in self.passes.iter().enumerate() {
            for &dep in pass.dependencies() {
                if let Some(&j) = name_to_idx.get(dep) {
                    adj[j].push(i);
                }
            }
        }
        let order = topological_sort(&adj);
        for idx in order {
            let pass = &mut self.passes[idx];
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
        let n = self.passes.len();
        let name_to_idx: HashMap<&str, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.name(), i))
            .collect();
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        for (i, pass) in self.passes.iter().enumerate() {
            for &dep in pass.dependencies() {
                if let Some(&j) = name_to_idx.get(dep) {
                    adj[j].push(i);
                }
            }
        }
        let order = topological_sort(&adj);
        for idx in order {
            let pass = &mut self.passes[idx];
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
