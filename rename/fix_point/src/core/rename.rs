use super::alias::collect_and_rename_aliases;


use super::collect::{add_file_module_symbol, collect_symbols};


use super::format::format_files;


use super::mod_decls::update_mod_declarations;


use super::paths::{module_path_for_file, plan_file_renames};


use super::preview::write_preview;


use super::structured::EditSessionTracker;


use crate::model::types::{SpanRange, SymbolEdit, SymbolIndex, SymbolOccurrence};


use super::use_map::build_use_map;


use super::use_paths::update_use_paths;


use crate::alias::AliasGraph;


use crate::fs;


use crate::structured::{
    rewrite_doc_and_attr_literals, structured_edit_config, StructuredAttributeResult,
};


use anyhow::{bail, Context, Result};


use proc_macro2::Span;


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use syn::visit::Visit;


use syn::visit_mut::VisitMut;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


struct SpanRangeRenamer {
    map: HashMap<SpanRangeKey, String>,
    changed: bool,
}


pub fn apply_rename(
    project: &Path,
    map_path: &Path,
    dry_run: bool,
    out_path: Option<&Path>,
) -> Result<()> {
    let mapping: HashMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(map_path)?,
    )?;
    apply_rename_with_map(project, &mapping, dry_run, out_path)
}


fn is_valid_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}
