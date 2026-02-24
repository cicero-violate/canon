//! Deduplication utilities that collapse duplicate captured items.

use crate::{CapturedItem, ExtractionResult};

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
    use std::collections::HashSet;
    let mut seen: HashSet<String> = HashSet::new();
    items.into_iter().filter(|item| {
        let key = dedup_key(item);
        seen.insert(key)
    }).collect()
}

fn dedup_key(item: &CapturedItem) -> String {
    match item {
        CapturedItem::Function(f)  => format!("function::{}", f.path),
        CapturedItem::Struct(s)    => format!("struct::{}", s.path),
        CapturedItem::Enum(e)      => format!("enum::{}", e.path),
        CapturedItem::Trait(t)     => format!("trait::{}", t.path),
        CapturedItem::Impl(i)      => format!("impl::{}", i.path),
        CapturedItem::Module(m)    => format!("module::{}", m.path),
        CapturedItem::TypeAlias(a) => format!("type_alias::{}", a.path),
        CapturedItem::Const(c)     => format!("const::{}", c.path),
        CapturedItem::Static(s)    => format!("static::{}", s.path),
    }
}
