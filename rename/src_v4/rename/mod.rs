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

// Re-export main rename functions
pub use core::{
    LineColumn, SpanRange, StructuredEditTracker, SymbolIndexReport, SymbolOccurrence,
    SymbolRecord, apply_rename, apply_rename_with_map, collect_names, emit_names, run_names,
    run_rename,
};

pub use alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};

pub use api::{
    MutationRequest, MutationResult, QueryRequest, QueryResult, UpsertRequest, UpsertResult,
    execute_mutation_json, execute_query_json, execute_upsert_json,
};

// C1: Re-export structured pass infrastructure
pub use structured::{
    AstEdit, DocAttrPass, StructuredPassRunner, StructuredEditOptions, StructuredPass, UsePathRewritePass,
    apply_ast_rewrites, ast_render, create_rename_orchestrator,
};
