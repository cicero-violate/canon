use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rename::alias::{ImportNode, VisibilityLeakAnalysis};

#[derive(Serialize)]
pub struct SymbolIndexReport {
    pub version: i64,
    pub symbols: Vec<SymbolRecord>,
    pub occurrences: Vec<SymbolOccurrence>,
    pub alias_graph: AliasGraphReport,
    pub visibility_analysis: Option<VisibilityLeakAnalysis>,
}

#[derive(Serialize)]
pub struct AliasGraphReport {
    pub use_nodes: Vec<ImportNode>,
    pub edge_count: usize,
    pub total_imports: usize,
    pub total_reexports: usize,
    pub glob_imports: usize,
}

#[derive(Serialize, Clone)]
pub struct SymbolRecord {
    pub id: String,
    pub kind: String,
    pub name: String,
    pub module: String,
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub declaration_file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub definition_file: Option<String>,
    pub span: SpanRange,
    pub alias: Option<String>,
    pub doc_comments: Vec<String>,
    pub attributes: Vec<String>,
}

#[derive(Serialize)]
pub struct SymbolOccurrence {
    pub id: String,
    pub file: String,
    pub kind: String,
    pub span: SpanRange,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SpanRange {
    pub start: LineColumn,
    pub end: LineColumn,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LineColumn {
    pub line: i64,
    pub column: i64,
}

#[derive(Default)]
pub struct SymbolIndex {
    pub symbols: HashMap<String, SymbolRecord>,
}

#[derive(Clone, Serialize)]
pub(crate) struct SymbolEdit {
    pub(crate) id: String,
    pub(crate) file: String,
    pub(crate) kind: String,
    pub(crate) start: LineColumn,
    pub(crate) end: LineColumn,
    pub(crate) new_name: String,
}

#[derive(Clone, Serialize)]
pub(crate) struct FileRename {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) is_directory_move: bool,
    pub(crate) old_module_id: String,
    pub(crate) new_module_id: String,
}
