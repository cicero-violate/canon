use std::collections::HashMap;


use std::path::PathBuf;


use std::sync::Arc;


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeHandle {
    /// File containing the node.
    pub file: PathBuf,
    /// Index within syn::File::items.
    pub item_index: usize,
    /// Nested path for items inside impls or modules.
    pub nested_path: Vec<usize>,
    /// Kind of node being referenced.
    pub kind: NodeKind,
    /// Span range for the node (line/column).
    pub span: crate::model::types::SpanRange,
    /// Byte offsets for the node within its source file.
    pub byte_range: (usize, usize),
    /// Source text snapshot used to compute the span/byte offsets.
    pub source: Arc<String>,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeKind {
    Fn,
    Struct,
    Enum,
    Trait,
    Impl,
    ImplFn,
    Use,
    Mod,
    Type,
    Const,
}


#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}
