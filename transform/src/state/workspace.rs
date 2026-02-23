#[derive(Debug, Clone)]
pub struct GraphWorkspace {
    hash: u64,
}

impl GraphWorkspace {
    pub fn hash(&self) -> u64 {
        self.hash
    }
}

#[derive(Debug, Clone)]
pub struct WorkspaceBuilder {
    hash: u64,
}

impl WorkspaceBuilder {
    pub fn new(hash: u64) -> Self {
        Self { hash }
    }

    pub fn finalize(self) -> GraphWorkspace {
        GraphWorkspace { hash: self.hash }
    }
}
