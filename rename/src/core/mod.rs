//! semantic: domain=tooling
//! Rename and refactoring core.
pub mod alias;
pub mod cli;
pub mod collect;
pub mod format;
pub mod mod_decls;
pub mod oracle;
pub mod paths;
pub mod preview;
pub mod project_editor;
pub mod rename;
pub mod structured;
pub mod symbol_id;
pub mod use_map;
pub mod use_paths;
pub use cli::{run_names, run_rename};
pub use collect::{collect_names, emit_names};
pub use oracle::{NullOracle, StructuralEditOracle};
pub use project_editor::{ChangeReport, EditConflict, ProjectEditor};
pub use rename::{apply_rename, apply_rename_with_map};
pub use structured::EditSessionTracker;
pub use symbol_id::normalize_symbol_id;
pub use crate::model::core_span::span_to_range;
pub use crate::model::types::{
    AliasGraphReport, LineColumn, SpanRange, SymbolIndex, SymbolIndexReport,
    SymbolOccurrence, SymbolRecord,
};
