use crate::core::collect::{add_file_module_symbol, collect_symbols};


use crate::core::paths::module_path_for_file;


use crate::core::symbol_id::normalize_symbol_id;


use crate::model::types::SymbolIndex;


use crate::state::NodeRegistry;


use anyhow::Result;


use std::collections::HashSet;


use std::path::{Path, PathBuf};


pub fn build_symbol_index(
    project_root: &Path,
    registry: &NodeRegistry,
) -> Result<SymbolIndex> {
    let mut symbol_table = SymbolIndex::default();
    let mut symbols = Vec::new();
    let mut symbol_set = HashSet::new();
    for (file, ast) in &registry.asts {
        let module_path = normalize_symbol_id(&module_path_for_file(project_root, file));
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let _ = collect_symbols(
            ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
    }
    Ok(symbol_table)
}


pub fn find_project_root(registry: &NodeRegistry) -> Result<Option<PathBuf>> {
    let file = match registry.asts.keys().next() {
        Some(f) => f,
        None => return Ok(None),
    };
    let mut current = file.parent().unwrap_or_else(|| Path::new("/")).to_path_buf();
    loop {
        if current.join("Cargo.toml").exists() {
            return Ok(Some(current));
        }
        if !current.pop() {
            break;
        }
    }
    Ok(None)
}


pub fn find_project_root_sync(registry: &NodeRegistry) -> Option<PathBuf> {
    let file = registry.asts.keys().next()?;
    let mut cur = file.parent()?.to_path_buf();
    loop {
        if cur.join("Cargo.toml").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}
