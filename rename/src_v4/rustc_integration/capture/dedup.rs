//! Deduplication utilities that collapse duplicate captured items.

use crate::rustc_integration::{CapturedItem, ExtractionResult};

/// Collapses duplicate captured items before normalization.
#[derive(Debug, Default)]
pub struct Deduplicator;

impl Deduplicator {
    /// Creates an empty deduplicator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Deduplicates the captured items in-place.
    pub fn deduplicate(&mut self, mut result: ExtractionResult) -> ExtractionResult {
        result.items = collapse_duplicates(result.items);
        result
    }
}

fn collapse_duplicates(items: Vec<CapturedItem>) -> Vec<CapturedItem> {
    // TODO: implement actual deduplication keyed by DefId/paths.
    items
}
