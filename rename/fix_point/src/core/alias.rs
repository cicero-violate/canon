use anyhow::{Context, Result};


use std::collections::HashMap;


use std::path::Path;


use syn::visit::{self, Visit};


use crate::fs;


use super::paths::module_path_for_file;


use crate::model::core_span::span_to_range;


use crate::model::types::{SymbolEdit, SymbolIndex};


use super::use_map::normalize_use_prefix;


struct AliasUsageVisitor<'a> {
    alias_name: String,
    new_alias: String,
    target_id: String,
    file: &'a Path,
    edits: &'a mut Vec<SymbolEdit>,
}
