#![cfg(feature = "rustc_frontend")]

/// Frontend metadata propagated to captured nodes.
#[derive(Clone)]
pub(super) struct FrontendMetadata {
    pub edition: String,
    pub rust_version: Option<String>,
    pub crate_type: String,
    pub target_triple: String,
    pub target_name: Option<String>,
    pub workspace_root: Option<String>,
    pub package_name: Option<String>,
    pub package_version: Option<String>,
    pub package_features: Vec<String>,
    pub cfg_flags: Vec<String>,
}
