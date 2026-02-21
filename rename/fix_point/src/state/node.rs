impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert_handle(&mut self, symbol_id: impl Into<String>, handle: NodeHandle) {
        self.handles.insert(symbol_id.into(), handle);
    }
    pub fn insert_ast(&mut self, file: impl Into<PathBuf>, ast: syn::File) {
        self.asts.insert(file.into(), ast);
    }
    pub fn insert_source(&mut self, file: impl Into<PathBuf>, source: Arc<String>) {
        self.sources.insert(file.into(), source);
    }
}


impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert_handle(&mut self, symbol_id: impl Into<String>, handle: NodeHandle) {
        self.handles.insert(symbol_id.into(), handle);
    }
    pub fn insert_ast(&mut self, file: impl Into<PathBuf>, ast: syn::File) {
        self.asts.insert(file.into(), ast);
    }
    pub fn insert_source(&mut self, file: impl Into<PathBuf>, source: Arc<String>) {
        self.sources.insert(file.into(), source);
    }
}


impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert_handle(&mut self, symbol_id: impl Into<String>, handle: NodeHandle) {
        self.handles.insert(symbol_id.into(), handle);
    }
    pub fn insert_ast(&mut self, file: impl Into<PathBuf>, ast: syn::File) {
        self.asts.insert(file.into(), ast);
    }
    pub fn insert_source(&mut self, file: impl Into<PathBuf>, source: Arc<String>) {
        self.sources.insert(file.into(), source);
    }
}


impl NodeRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert_handle(&mut self, symbol_id: impl Into<String>, handle: NodeHandle) {
        self.handles.insert(symbol_id.into(), handle);
    }
    pub fn insert_ast(&mut self, file: impl Into<PathBuf>, ast: syn::File) {
        self.asts.insert(file.into(), ast);
    }
    pub fn insert_source(&mut self, file: impl Into<PathBuf>, source: Arc<String>) {
        self.sources.insert(file.into(), source);
    }
}


/// Identifies a concrete AST node inside a file.
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


/// Identifies a concrete AST node inside a file.
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


/// Identifies a concrete AST node inside a file.
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


/// Identifies a concrete AST node inside a file.
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


/// Supported structural node categories.
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


/// Supported structural node categories.
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


/// Supported structural node categories.
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


/// Supported structural node categories.
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


/// Registry connecting symbol IDs to live AST handles.
#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}


/// Registry connecting symbol IDs to live AST handles.
#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}


/// Registry connecting symbol IDs to live AST handles.
#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}


/// Registry connecting symbol IDs to live AST handles.
#[derive(Default)]
pub struct NodeRegistry {
    /// symbol_id -> node handle
    pub handles: HashMap<String, NodeHandle>,
    /// file -> parsed AST
    pub asts: HashMap<PathBuf, syn::File>,
    /// file -> source text snapshot
    pub sources: HashMap<PathBuf, Arc<String>>,
}
