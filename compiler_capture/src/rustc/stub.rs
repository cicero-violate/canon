use super::RustcFrontendError;
use crate::compiler_capture::graph::GraphDelta;

/// Stubbed frontend compiled when `rustc_frontend` feature is disabled.
#[derive(Debug, Clone, Default)]
pub struct RustcFrontend;

impl RustcFrontend {
    /// Creates a stub frontend.
    pub fn new() -> Self {
        Self
    }

    /// No-op when the real frontend is unavailable.
    pub fn with_crate_type(self, _crate_type: impl Into<String>) -> Self {
        self
    }

    /// No-op when the real frontend is unavailable.
    pub fn with_edition(self, _edition: impl Into<String>) -> Self {
        self
    }

    /// No-op when the real frontend is unavailable.
    pub fn with_rust_version(self, _rust_version: impl Into<String>) -> Self {
        self
    }

    /// No-op when the real frontend is unavailable.
    pub fn with_target_name(self, _target_name: impl Into<String>) -> Self {
        self
    }

    pub fn with_workspace_root(self, _root: impl Into<std::path::PathBuf>) -> Self {
        self
    }

    pub fn with_package_info(self, _name: impl Into<String>, _version: impl Into<String>) -> Self {
        self
    }

    pub fn with_package_features<I, S>(self, _features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self
    }

    pub fn with_cfg_flags<I, S>(self, _flags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self
    }

    /// Always returns [`RustcFrontendError::Unavailable`].
    pub fn capture_deltas<P: AsRef<std::path::Path>>(&self, _entry: P, _args: &[String], _env_vars: &[(String, String)]) -> Result<Vec<GraphDelta>, RustcFrontendError> {
        Err(RustcFrontendError::Unavailable)
    }
}
