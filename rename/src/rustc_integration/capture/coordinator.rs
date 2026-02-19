//! Coordinates running compiler frontends across projects within a workspace.
use std::error::Error;
use std::path::{Path, PathBuf};
use super::dedup::Deduplicator;
use super::session::CaptureSession;
use crate::rustc_integration::{ExtractionResult, FrontendExtractor};
/// High level orchestrator that walks a workspace and runs a frontend for each crate.
pub struct CaptureCoordinator {
    dedup: Deduplicator,
    session: CaptureSession,
}
impl CaptureCoordinator {
    /// Creates a new coordinator with fresh session/dedup state.
    pub fn new() -> Self {
        Self {
            dedup: Deduplicator::new(),
            session: CaptureSession::default(),
        }
    }
    /// Walks the workspace, invoking the provided frontend on every discovered crate.
    pub fn capture_project_workspace<F>(
        &mut self,
        workspace_root: &Path,
        frontend: &mut F,
    ) -> Result<Vec<ExtractionResult>, Box<dyn Error>>
    where
        F: FrontendExtractor<Config = PathBuf>,
        F::Error: Error + 'static,
    {
        let crates = self.discover_crates(workspace_root)?;
        let mut results = Vec::with_capacity(crates.len());
        for crate_path in crates {
            let mut result = frontend.extract(crate_path)?;
            result = self.dedup.deduplicate(result);
            self.session.record(&result);
            results.push(result);
        }
        Ok(results)
    }
    /// Discovers candidate crates inside the workspace.
    fn discover_crates(
        &self,
        workspace_root: &Path,
    ) -> Result<Vec<PathBuf>, std::io::Error> {
        Ok(vec![workspace_root.to_path_buf()])
    }
}
