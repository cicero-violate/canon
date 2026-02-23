use std::path::PathBuf;


use std::sync::Arc;


use crate::state::{NodeHandle, NodeKind};


use crate::model::types::SpanRange;


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
