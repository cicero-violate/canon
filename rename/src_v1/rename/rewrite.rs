//! Rewrite buffer infrastructure for structured editing layers

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Represents a single text edit operation
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTextEdit {
    /// Start byte offset in the original text
    pub start: usize,
    /// End byte offset in the original text (exclusive)
    pub end: usize,
    /// Replacement text to insert
    pub text: String,
}

/// A conflict-aware buffer for accumulating edits to a single file
///
/// # A1: Conflict-Aware RewriteBuffer
///
/// Features:
/// - Validates all edit spans are within bounds
/// - Deduplicates identical edits automatically
/// - Reports conflicts when overlapping edits would produce ambiguous results
/// - Provides helpers: replace, insert, insert_before, insert_after, insert_tokens
pub struct RewriteBuffer {
    path: PathBuf,
    original: String,
    edits: Vec<SourceTextEdit>,
}

#[cfg(test)]
mod d1_d2_tests {
    use super::*;

    #[test]
    fn test_flush_result_tracking() {
        let mut set = RewriteBufferSet::new();
        let path1 = PathBuf::from("file1.rs");
        let path2 = PathBuf::from("file2.rs");

        // Add edits to multiple files
        let buf1 = set.ensure_buffer(&path1, "content1");
        buf1.insert(0, "// ").unwrap();
        buf1.insert(8, "!").unwrap();

        let buf2 = set.ensure_buffer(&path2, "content2");
        buf2.insert(0, "use std;").unwrap();

        // Verify state before flush
        assert_eq!(set.total_edit_count(), 3);
        assert_eq!(set.dirty_files().len(), 2);

        // Note: We can't actually flush to disk in tests without temp files,
        // but we can verify the tracking logic
        assert!(set.is_dirty());
        assert!(set.is_file_dirty(&path1));
        assert!(set.is_file_dirty(&path2));
    }
}

impl RewriteBuffer {
    /// Create a new buffer for a file with given original content
    pub fn new(path: PathBuf, original: String) -> Self {
        Self {
            path,
            original,
            edits: Vec::new(),
        }
    }

    /// Queue an edit, validating bounds and checking for conflicts
    ///
    /// Returns:
    /// - Ok(()) if edit is valid and doesn't conflict
    /// - Ok(()) if edit is a duplicate (silently ignored)
    /// - Err if edit is out of bounds or conflicts with existing edits
    pub fn queue_edit(&mut self, edit: SourceTextEdit) -> Result<()> {
        self.validate_bounds(&edit)?;
        if self.is_duplicate(&edit) {
            return Ok(());
        }
        self.detect_conflict(&edit)?;
        self.edits.push(edit);
        Ok(())
    }

    /// Replace a span [start, end) in the buffer with new text
    ///
    /// # A1 Helper: replace
    ///
    /// Primary edit operation. All other helpers delegate to this.
    pub fn replace(&mut self, start: usize, end: usize, text: impl Into<String>) -> Result<()> {
        self.queue_edit(SourceTextEdit {
            start,
            end,
            text: text.into(),
        })
    }

    /// Insert text at an exact offset (zero-width replacement)
    ///
    /// # A1 Helper: insert
    pub fn insert(&mut self, offset: usize, text: impl Into<String>) -> Result<()> {
        self.replace(offset, offset, text)
    }

    /// Insert text immediately before a span (semantic alias for insert)
    ///
    /// # A1 Helper: insert_before
    ///
    /// Useful for prepending to an existing node without replacing it.
    pub fn insert_before(&mut self, anchor_start: usize, text: impl Into<String>) -> Result<()> {
        self.insert(anchor_start, text)
    }

    /// Insert text immediately after a span
    ///
    /// # A1 Helper: insert_after
    ///
    /// Useful for appending after an existing node without replacing it.
    pub fn insert_after(&mut self, anchor_end: usize, text: impl Into<String>) -> Result<()> {
        self.insert(anchor_end, text)
    }

    /// Insert a rendered AST fragment at an offset
    ///
    /// # A1 Helper: insert_tokens
    ///
    /// Renders any `syn` or `quote::ToTokens` node to text and inserts it.
    /// Enables structured AST-level edits without manual string construction.
    pub fn insert_tokens<T: quote::ToTokens>(&mut self, offset: usize, tokens: T) -> Result<()> {
        let mut stream = proc_macro2::TokenStream::new();
        tokens.to_tokens(&mut stream);
        self.insert(offset, stream.to_string())
    }

    /// Check if this buffer has any pending edits
    pub fn is_dirty(&self) -> bool {
        !self.edits.is_empty()
    }

    /// Get the original content before any edits
    pub fn original(&self) -> &str {
        &self.original
    }

    /// Get the file path this buffer is editing
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get number of pending edits
    pub fn edit_count(&self) -> usize {
        self.edits.len()
    }

    /// Get all pending edits (for diagnostics/debugging)
    pub fn edits(&self) -> &[SourceTextEdit] {
        &self.edits
    }

    /// Clear all pending edits without applying them
    pub fn clear_edits(&mut self) {
        self.edits.clear();
    }

    /// Internal: commit rendered output as new baseline and clear edits
    fn commit(&mut self, next: String) {
        self.original = next;
        self.edits.clear();
    }

    /// Internal: render all edits by applying them in reverse order
    fn render(&self) -> String {
        if self.edits.is_empty() {
            return self.original.clone();
        }
        let mut output = self.original.clone();
        let mut edits = self.edits.clone();
        edits.sort_by(|a, b| b.start.cmp(&a.start).then_with(|| b.end.cmp(&a.end)));
        for edit in edits {
            if edit.start > edit.end || edit.end > output.len() {
                continue;
            }
            output.replace_range(edit.start..edit.end, &edit.text);
        }
        output
    }

    /// Validate edit bounds against original file size
    fn validate_bounds(&self, edit: &SourceTextEdit) -> Result<()> {
        if edit.start > edit.end {
            bail!(
                "Invalid edit for {}: start {} > end {}",
                self.path.display(),
                edit.start,
                edit.end
            );
        }
        if edit.end > self.original.len() {
            bail!(
                "Edit for {} ends past file ({} > {})",
                self.path.display(),
                edit.end,
                self.original.len()
            );
        }
        Ok(())
    }

    /// Check if edit is identical to an existing queued edit
    fn is_duplicate(&self, edit: &SourceTextEdit) -> bool {
        self.edits.iter().any(|existing| existing == edit)
    }

    /// Detect if new edit conflicts with any existing edit
    fn detect_conflict(&self, new_edit: &SourceTextEdit) -> Result<()> {
        for existing in &self.edits {
            if edits_conflict(existing, new_edit) {
                bail!(
                    "Conflicting edits queued for {}: {:?} vs {:?}",
                    self.path.display(),
                    existing,
                    new_edit
                );
            }
        }
        Ok(())
    }
}

/// A manager for coordinating edits across multiple files
///
/// # A2: Buffer Manager API
///
/// Features:
/// - Get mutable handles to individual buffers
/// - Query dirty state per file or globally
/// - List all touched files
/// - Coordinate edits and diagnostics across passes
#[derive(Default)]
pub struct RewriteBufferSet {
    buffers: HashMap<PathBuf, RewriteBuffer>,
}

impl RewriteBufferSet {
    /// Create a new empty buffer set
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Get or create a buffer for the given file
    ///
    /// # A2 API: ensure_buffer
    ///
    /// Returns a mutable handle so callers can queue edits directly.
    pub fn ensure_buffer<'a>(&'a mut self, path: &Path, original: &str) -> &'a mut RewriteBuffer {
        self.buffers
            .entry(path.to_path_buf())
            .or_insert_with(|| RewriteBuffer::new(path.to_path_buf(), original.to_string()))
    }

    /// Queue multiple edits to a file at once
    pub fn queue_edits(
        &mut self,
        path: &Path,
        original: &str,
        edits: impl IntoIterator<Item = SourceTextEdit>,
    ) -> Result<()> {
        let buffer = self.ensure_buffer(path, original);
        for edit in edits {
            buffer.queue_edit(edit)?;
        }
        Ok(())
    }

    /// Get a buffer if it exists (without creating)
    ///
    /// # A2 API: get_buffer
    pub fn get_buffer(&self, path: &Path) -> Option<&RewriteBuffer> {
        self.buffers.get(path)
    }

    /// Get a mutable buffer if it exists (without creating)
    ///
    /// # A2 API: get_buffer_mut
    pub fn get_buffer_mut(&mut self, path: &Path) -> Option<&mut RewriteBuffer> {
        self.buffers.get_mut(path)
    }

    /// Check if any buffer has pending edits
    ///
    /// # A2 API: is_dirty
    pub fn is_dirty(&self) -> bool {
        self.buffers.values().any(|b| b.is_dirty())
    }

    /// Check if a specific file has pending edits
    ///
    /// # A2 API: is_file_dirty
    pub fn is_file_dirty(&self, path: &Path) -> bool {
        self.buffers
            .get(path)
            .map(|b| b.is_dirty())
            .unwrap_or(false)
    }

    /// List all files with pending edits
    ///
    /// # A2 API: dirty_files
    pub fn dirty_files(&self) -> Vec<&Path> {
        self.buffers
            .iter()
            .filter(|(_, b)| b.is_dirty())
            .map(|(p, _)| p.as_path())
            .collect()
    }

    /// List all files tracked by this buffer set
    ///
    /// # A2 API: all_files
    pub fn all_files(&self) -> Vec<&Path> {
        self.buffers.keys().map(|p| p.as_path()).collect()
    }

    /// Get total number of pending edits across all buffers
    ///
    /// # A2 API: total_edit_count
    pub fn total_edit_count(&self) -> usize {
        self.buffers.values().map(|b| b.edit_count()).sum()
    }

    /// Clear all pending edits without flushing
    ///
    /// # A2 API: clear_all
    pub fn clear_all(&mut self) {
        for buffer in self.buffers.values_mut() {
            buffer.clear_edits();
        }
    }

    /// Render all dirty buffers to disk and return list of touched files
    ///
    /// # A2 API: flush
    ///
    /// Only writes files that actually changed after applying edits.
    pub fn flush(&mut self) -> Result<Vec<PathBuf>> {
        let mut touched = Vec::new();
        for buffer in self.buffers.values_mut() {
            if !buffer.is_dirty() {
                continue;
            }
            let updated = buffer.render();
            if updated != buffer.original() {
                std::fs::write(buffer.path(), &updated)?;
                touched.push(buffer.path().to_path_buf());
            }
            buffer.commit(updated);
        }
        Ok(touched)
    }
}

/// Detect if two edits conflict (overlap in a way that makes order ambiguous)
fn edits_conflict(existing: &SourceTextEdit, new_edit: &SourceTextEdit) -> bool {
    let existing_is_insert = existing.start == existing.end;
    let new_is_insert = new_edit.start == new_edit.end;
    match (existing_is_insert, new_is_insert) {
        (true, true) => existing.start == new_edit.start && existing.text != new_edit.text,
        (true, false) => {
            let pos = existing.start;
            new_edit.start < pos && pos < new_edit.end
        }
        (false, true) => {
            let pos = new_edit.start;
            existing.start < pos && pos < existing.end
        }
        (false, false) => {
            let start = existing.start.max(new_edit.start);
            let end = existing.end.min(new_edit.end);
            start < end
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detects_conflicts_and_allows_duplicates() {
        let mut buf = RewriteBuffer::new(PathBuf::from("demo.rs"), "abcdef".into());
        buf.replace(0, 3, "foo").unwrap();
        // Duplicate is ignored
        buf.replace(0, 3, "foo").unwrap();
        assert_eq!(buf.edits.len(), 1);

        let err = buf.replace(1, 4, "bar").unwrap_err();
        assert!(
            err.to_string().contains("Conflicting edits"),
            "expected conflict, got {err}"
        );
    }

    #[test]
    fn insert_helpers_work() {
        let mut buf = RewriteBuffer::new(PathBuf::from("demo.rs"), "abc".into());
        buf.insert_before(0, "use ").unwrap();
        buf.insert_after(3, "!").unwrap();
        let rendered = buf.render();
        assert_eq!(rendered, "use abc!");
    }

    #[test]
    fn detects_insert_vs_replace_conflict() {
        let mut buf = RewriteBuffer::new(PathBuf::from("demo.rs"), "abcdef".into());
        buf.insert_after(2, "_").unwrap();
        let err = buf.replace(0, 4, "x").unwrap_err();
        assert!(
            err.to_string().contains("Conflicting edits"),
            "expected conflict, got {err}"
        );
    }

    #[test]
    fn buffer_manager_api_queries() {
        let mut set = RewriteBufferSet::new();
        let path1 = PathBuf::from("file1.rs");
        let path2 = PathBuf::from("file2.rs");

        // Initially no dirty files
        assert!(!set.is_dirty());
        assert_eq!(set.dirty_files().len(), 0);
        assert_eq!(set.total_edit_count(), 0);

        // Add edit to first file
        let buf1 = set.ensure_buffer(&path1, "content1");
        buf1.insert(0, "prefix_").unwrap();

        // Query state
        assert!(set.is_dirty());
        assert!(set.is_file_dirty(&path1));
        assert!(!set.is_file_dirty(&path2));
        assert_eq!(set.dirty_files().len(), 1);
        assert_eq!(set.total_edit_count(), 1);

        // Add buffer without edits
        set.ensure_buffer(&path2, "content2");
        assert_eq!(set.all_files().len(), 2);
        assert_eq!(set.dirty_files().len(), 1);

        // Clear all
        set.clear_all();
        assert!(!set.is_dirty());
        assert_eq!(set.total_edit_count(), 0);
    }

    #[test]
    fn buffer_info_accessors() {
        let mut buf = RewriteBuffer::new(PathBuf::from("test.rs"), "original".into());
        assert_eq!(buf.edit_count(), 0);
        assert!(!buf.is_dirty());

        buf.insert(0, "pre_").unwrap();
        assert_eq!(buf.edit_count(), 1);
        assert!(buf.is_dirty());
        assert_eq!(buf.edits().len(), 1);

        buf.insert(8, "_post").unwrap();
        assert_eq!(buf.edit_count(), 2);

        buf.clear_edits();
        assert_eq!(buf.edit_count(), 0);
        assert!(!buf.is_dirty());
    }
}
