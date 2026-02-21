//! semantic: domain=tooling
//! Rename and refactoring infrastructure
pub mod alias;
pub mod api;
pub mod attributes;
pub mod core;
pub mod macros;
pub mod module_path;
pub mod occurrence;
pub mod pattern;
pub mod scope;
pub mod structured;
pub use crate::alias;
pub use crate::api;
pub use crate::attributes;
pub use crate::core;
pub use crate::macros;
pub use crate::module_path;
pub use crate::occurrence;
pub use crate::pattern;
pub use crate::scope;
pub use crate::structured;
pub use core::{
    apply_rename, apply_rename_with_map, collect_names, emit_names, run_names,
    run_rename, LineColumn, SpanRange, EditSessionTracker, SymbolIndexReport,
    SymbolOccurrence, SymbolRecord,
};
pub use alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};
pub use api::{
    execute_mutation_json, execute_query_json, execute_upsert_json, MutationRequest,
    MutationResult, QueryRequest, QueryResult, UpsertRequest, UpsertResult,
};
pub use structured::{
    apply_ast_rewrites, ast_render, create_rename_orchestrator, AstEdit, DocAttrPass,
    StructuredEditOptions, StructuredPass, StructuredPassRunner, UsePathRewritePass,
};
