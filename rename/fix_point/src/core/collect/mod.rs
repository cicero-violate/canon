mod collector;

use anyhow::{Context, Result};


use std::collections::{HashMap, HashSet};


use std::path::Path;


use crate::alias::{AliasGraph, ImportNode, UseKind, VisibilityScope};


use crate::fs;


use crate::occurrence::OccurrenceVisitor;


use super::paths::module_path_for_file;


use super::symbol_id::normalize_symbol_id;


use crate::model::types::{
    AliasGraphReport, LineColumn, SpanRange, SymbolIndex, SymbolIndexReport, SymbolRecord,
};


use super::use_map::build_use_map;


use collector::ItemCollector;


fn stub_range() -> SpanRange {
    SpanRange {
        start: LineColumn { line: 1, column: 1 },
        end: LineColumn { line: 1, column: 1 },
    }
}
