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
pub mod rewrite;
pub mod scope;
pub mod structured;

// Re-export main rename functions
pub use core::{
    apply_rename, apply_rename_with_map, collect_names, emit_names, run_names, run_rename,
    FlushResult, LineColumn, NamesReport, OccurrenceEntry, SpanRange, StructuredEditTracker,
    SymbolEntry,
};

pub use alias::{AliasGraph, UseKind, UseNode, VisibilityScope};

pub use api::{
    execute_mutation_json, execute_query_json, execute_upsert_json, MutationRequest,
    MutationResult, QueryRequest, QueryResult, UpsertRequest, UpsertResult,
};

// C1: Re-export structured pass infrastructure
pub use structured::{
    apply_ast_rewrites, ast_render, create_rename_orchestrator, AstEdit, DocAttrPass,
    PassOrchestrator, StructuredEditConfig, StructuredPass, UseTreePass,
};
