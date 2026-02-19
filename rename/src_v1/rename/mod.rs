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
    LineColumn, RewriteSummary, SpanRange, StructuredEditTracker, SymbolIndexReport,
    SymbolOccurrence, SymbolRecord, apply_rename, apply_rename_with_map, collect_names, emit_names,
    run_names, run_rename,
};

pub use alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};

pub use api::{
    MutationRequest, MutationResult, QueryRequest, QueryResult, UpsertRequest, UpsertResult,
    execute_mutation_json, execute_query_json, execute_upsert_json,
};

// C1: Re-export structured pass infrastructure
pub use structured::{
    AstEdit, DocAttrPass, PassOrchestrator, StructuredEditConfig, StructuredPass, UseTreePass,
    apply_ast_rewrites, ast_render, create_rename_orchestrator,
};
