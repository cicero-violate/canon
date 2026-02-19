//! semantic: domain=tooling
//! Rename and refactoring core.

pub mod alias;
pub mod cli;
pub mod collect;
pub mod format;
pub mod mod_decls;
pub mod paths;
pub mod preview;
pub mod rename;
pub mod span;
pub mod structured;
pub mod use_map;
pub mod use_paths;
pub mod types;

pub use cli::{run_names, run_rename};
pub use collect::{collect_names, emit_names};
pub use rename::{apply_rename, apply_rename_with_map};
pub use span::span_to_range;
pub use structured::StructuredEditTracker;
pub use types::{
    AliasGraphReport, LineColumn, SpanRange, SymbolIndex, SymbolIndexReport, SymbolOccurrence,
    SymbolRecord,
};
pub(crate) use span::span_to_offsets;
