//! Layout graph definitions (LAY-002).
//!
//! These types describe how semantic Canon IR nodes are split from their
//! filesystem layout. Future work will teach the ingestor/materializer to
//! consume these structures instead of mixing semantics + layout data.

use crate::ir::{
    CallEdge, EnumId, EnumNode, Function, FunctionId, ImplBlock, ImplId, Module, ModuleEdge,
    ModuleId, Struct, StructId, SystemGraph, TickGraph, Trait, TraitId, Word,
};

/// Container that pairs the semantic and layout graphs used during
/// ingestion/materialization.
#[derive(Debug, Clone)]
pub struct LayoutMap {
    pub semantic: SemanticGraph,
    pub layout: LayoutGraph,
}

/// A purely semantic projection of Canon IR.
#[derive(Debug, Clone, Default)]
pub struct SemanticGraph {
    pub modules: Vec<Module>,
    pub structs: Vec<Struct>,
    pub enums: Vec<EnumNode>,
    pub traits: Vec<Trait>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    pub module_edges: Vec<ModuleEdge>,
    pub call_edges: Vec<CallEdge>,
    pub tick_graphs: Vec<TickGraph>,
    pub system_graphs: Vec<SystemGraph>,
}

/// Filesystem layout for the semantic nodes.
#[derive(Debug, Clone, Default)]
pub struct LayoutGraph {
    pub modules: Vec<LayoutModule>,
    pub routing: Vec<LayoutAssignment>,
}

/// Modules describe their explicit file nodes and imports.
#[derive(Debug, Clone)]
pub struct LayoutModule {
    pub id: ModuleId,
    pub name: Word,
    pub files: Vec<LayoutFile>,
    pub imports: Vec<LayoutImport>,
}

#[derive(Debug, Clone)]
pub struct LayoutFile {
    pub id: String,
    pub path: String,
    pub use_block: Vec<String>,
}

/// Maps a semantic node to the file that should render it.
#[derive(Debug, Clone)]
pub struct LayoutAssignment {
    pub node: LayoutNode,
    pub file_id: String,
    pub rationale: String,
}

#[derive(Debug, Clone)]
pub struct LayoutImport {
    pub from: ModuleId,
    pub to: ModuleId,
    pub symbols: Vec<String>,
}

/// Identifiers that can be routed to a file.
#[derive(Debug, Clone)]
pub enum LayoutNode {
    Struct(StructId),
    Enum(EnumId),
    Trait(TraitId),
    Impl(ImplId),
    Function(FunctionId),
}

/// Strategies describe how to derive a `LayoutGraph` for a given semantic view.
pub trait LayoutStrategy {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str {
        ""
    }
    fn plan(&self, semantic: &SemanticGraph) -> LayoutGraph;
}

/// Represents helpers that can strip layout metadata from semantic nodes.
pub trait LayoutCleaner {
    fn drop_layout_fields(&self, map: LayoutMap) -> LayoutMap;
}
