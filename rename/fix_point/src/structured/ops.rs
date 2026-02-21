pub fn node_handle(
    file: impl Into<PathBuf>,
    item_index: usize,
    nested_path: Vec<usize>,
    kind: NodeKind,
    span: SpanRange,
    byte_range: (usize, usize),
    source: Arc<String>,
) -> NodeHandle {
    NodeHandle {
        file: file.into(),
        item_index,
        nested_path,
        kind,
        span,
        byte_range,
        source,
    }
}


pub fn node_handle(
    file: impl Into<PathBuf>,
    item_index: usize,
    nested_path: Vec<usize>,
    kind: NodeKind,
    span: SpanRange,
    byte_range: (usize, usize),
    source: Arc<String>,
) -> NodeHandle {
    NodeHandle {
        file: file.into(),
        item_index,
        nested_path,
        kind,
        span,
        byte_range,
        source,
    }
}


pub fn node_handle(
    file: impl Into<PathBuf>,
    item_index: usize,
    nested_path: Vec<usize>,
    kind: NodeKind,
    span: SpanRange,
    byte_range: (usize, usize),
    source: Arc<String>,
) -> NodeHandle {
    NodeHandle {
        file: file.into(),
        item_index,
        nested_path,
        kind,
        span,
        byte_range,
        source,
    }
}


pub fn node_handle(
    file: impl Into<PathBuf>,
    item_index: usize,
    nested_path: Vec<usize>,
    kind: NodeKind,
    span: SpanRange,
    byte_range: (usize, usize),
    source: Arc<String>,
) -> NodeHandle {
    NodeHandle {
        file: file.into(),
        item_index,
        nested_path,
        kind,
        span,
        byte_range,
        source,
    }
}


/// Structural node operations executed against live ASTs.
#[derive(Clone)]
pub enum NodeOp {
    ReplaceNode { handle: NodeHandle, new_node: syn::Item },
    InsertBefore { handle: NodeHandle, new_node: syn::Item },
    InsertAfter { handle: NodeHandle, new_node: syn::Item },
    DeleteNode { handle: NodeHandle },
    MutateField { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems { file: PathBuf, new_order: Vec<String> },
    MoveSymbol {
        handle: NodeHandle,
        symbol_id: String,
        new_module_path: String,
        new_crate: Option<String>,
    },
}


/// Structural node operations executed against live ASTs.
#[derive(Clone)]
pub enum NodeOp {
    ReplaceNode { handle: NodeHandle, new_node: syn::Item },
    InsertBefore { handle: NodeHandle, new_node: syn::Item },
    InsertAfter { handle: NodeHandle, new_node: syn::Item },
    DeleteNode { handle: NodeHandle },
    MutateField { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems { file: PathBuf, new_order: Vec<String> },
    MoveSymbol {
        handle: NodeHandle,
        symbol_id: String,
        new_module_path: String,
        new_crate: Option<String>,
    },
}


/// Structural node operations executed against live ASTs.
#[derive(Clone)]
pub enum NodeOp {
    ReplaceNode { handle: NodeHandle, new_node: syn::Item },
    InsertBefore { handle: NodeHandle, new_node: syn::Item },
    InsertAfter { handle: NodeHandle, new_node: syn::Item },
    DeleteNode { handle: NodeHandle },
    MutateField { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems { file: PathBuf, new_order: Vec<String> },
    MoveSymbol {
        handle: NodeHandle,
        symbol_id: String,
        new_module_path: String,
        new_crate: Option<String>,
    },
}


/// Structural node operations executed against live ASTs.
#[derive(Clone)]
pub enum NodeOp {
    ReplaceNode { handle: NodeHandle, new_node: syn::Item },
    InsertBefore { handle: NodeHandle, new_node: syn::Item },
    InsertAfter { handle: NodeHandle, new_node: syn::Item },
    DeleteNode { handle: NodeHandle },
    MutateField { handle: NodeHandle, mutation: FieldMutation },
    ReorderItems { file: PathBuf, new_order: Vec<String> },
    MoveSymbol {
        handle: NodeHandle,
        symbol_id: String,
        new_module_path: String,
        new_crate: Option<String>,
    },
}


/// Field-level mutation options for structural edits.
#[derive(Clone)]
pub enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}


/// Field-level mutation options for structural edits.
#[derive(Clone)]
pub enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}


/// Field-level mutation options for structural edits.
#[derive(Clone)]
pub enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}


/// Field-level mutation options for structural edits.
#[derive(Clone)]
pub enum FieldMutation {
    RenameIdent(String),
    ChangeVisibility(syn::Visibility),
    AddAttribute(syn::Attribute),
    RemoveAttribute(String),
    ReplaceSignature(syn::Signature),
    AddStructField(syn::Field),
    RemoveStructField(String),
    AddVariant(syn::Variant),
    RemoveVariant(String),
}
