use crate::structured::StructuredEditOptions;
use std::collections::HashSet;
/// B2: Structured files tracking for pipeline coordination
///
/// # D2: Enhanced with per-pass file tracking
///
/// Tracks which files were touched by each structured editing pass:
/// - doc_files: Files modified by doc comment literal rewrites
/// - attr_files: Files modified by attribute literal rewrites
/// - use_files: Files modified by use statement synthesis
///
/// This enables:
/// - Detailed diagnostics showing what changed where
/// - Dry-run previews with pass-specific breakdowns
/// - Debugging structured editing pipelines
#[derive(Default)]
pub struct StructuredEditTracker {
    files: HashSet<String>,
    pub(crate) doc_files: HashSet<String>,
    pub(crate) attr_files: HashSet<String>,
    pub(crate) use_files: HashSet<String>,
}
impl StructuredEditTracker {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn mark_doc_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.doc_files.insert(file);
    }
    pub fn mark_attr_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.attr_files.insert(file);
    }
    pub fn mark_use_edit(&mut self, file: String) {
        self.files.insert(file.clone());
        self.use_files.insert(file);
    }
    pub fn mark_generic(&mut self, file: String) {
        self.files.insert(file);
    }
    pub fn all_files(&self) -> &HashSet<String> {
        &self.files
    }
    pub fn into_set(self) -> HashSet<String> {
        self.files
    }
    /// D2: Get files touched by doc literal rewrites
    pub fn doc_files(&self) -> &HashSet<String> {
        &self.doc_files
    }
    /// D2: Get files touched by attr literal rewrites
    pub fn attr_files(&self) -> &HashSet<String> {
        &self.attr_files
    }
    /// D2: Get files touched by use statement rewrites
    pub fn use_files(&self) -> &HashSet<String> {
        &self.use_files
    }
    pub fn summary(&self, config: &StructuredEditOptions) -> String {
        let mut parts = Vec::new();
        if config.doc_literals_enabled() && !self.doc_files.is_empty() {
            parts.push(format!("docs:{}", self.doc_files.len()));
        }
        if config.attr_literals_enabled() && !self.attr_files.is_empty() {
            parts.push(format!("attrs:{}", self.attr_files.len()));
        }
        if config.use_statements_enabled() && !self.use_files.is_empty() {
            parts.push(format!("uses:{}", self.use_files.len()));
        }
        if parts.is_empty() {
            format!("{} files via structured edits", self.files.len())
        } else {
            format!("{} files via structured edits ({})", self.files.len(), parts.join(", "))
        }
    }
}
