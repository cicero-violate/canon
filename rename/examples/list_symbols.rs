#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::collect_names;
use serde::Serialize;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Avoid parsing materialized_project (may contain generated / invalid code)
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/rename");

    let report = collect_names(project_path)?;

    // Build aligned table with Kind / Symbol / File
    let mut rows = Vec::new();
    let mut max_kind = "Kind".len();
    let mut max_symbol = "Symbol".len();
    let mut max_file = "File".len();

    use std::collections::HashSet;

    let mut seen = HashSet::new();

    for symbol in &report.symbols {
        // Exclude anything under /tests/
        if symbol.file.contains("/tests/") {
            continue;
        }

        let kind = format!("{:?}", symbol.kind);

        // Remove redundant trailing module segment (e.g. helpers::helpers)
        let mut module_path = symbol.module.clone();
        let name = symbol.name.clone();

        if module_path.ends_with(&format!("::{}", name)) {
            module_path = module_path.rsplit_once("::").map(|(parent, _)| parent.to_string()).unwrap_or(module_path);
        }

        let symbol_path = format!("{}::{}", module_path, name);
        let file = symbol.file.clone();

        // Deduplicate by (kind, symbol_path)
        let key = format!("{}::{}", kind, symbol_path);
        if !seen.insert(key) {
            continue;
        }

        max_kind = max_kind.max(kind.len());
        max_symbol = max_symbol.max(symbol_path.len());
        max_file = max_file.max(file.len());

        rows.push((kind, symbol_path, file));
    }

    let mut output = String::new();

    // Header
    output.push_str(&format!("| {:<max_kind$} | {:<max_symbol$} | {:<max_file$} |\n", "Kind", "Symbol", "File", max_kind = max_kind, max_symbol = max_symbol, max_file = max_file));

    // Separator
    output.push_str(&format!("|-{:-<max_kind$}-|-{:-<max_symbol$}-|-{:-<max_file$}-|\n", "", "", "", max_kind = max_kind, max_symbol = max_symbol, max_file = max_file));

    // Rows
    for (kind, symbol_path, file) in rows {
        output.push_str(&format!("| {:<max_kind$} | {:<max_symbol$} | {:<max_file$} |\n", kind, symbol_path, file, max_kind = max_kind, max_symbol = max_symbol, max_file = max_file));
    }

    // Save in project root directory
    let md_path = project_path.parent().unwrap_or(project_path).join("SYMBOLS.md");

    let json_path = project_path.parent().unwrap_or(project_path).join("SYMBOLS.json");

    fs::write(&md_path, output)?;

    #[derive(Serialize)]
    struct JsonSymbol {
        kind: String,
        symbol: String,
        file: String,
    }

    let json_rows: Vec<JsonSymbol> = report
        .symbols
        .iter()
        .filter(|s| !s.file.contains("/tests/"))
        .map(|s| {
            let mut module_path = s.module.clone();
            let name = s.name.clone();

            if module_path.ends_with(&format!("::{}", name)) {
                module_path = module_path.rsplit_once("::").map(|(parent, _)| parent.to_string()).unwrap_or(module_path);
            }

            JsonSymbol { kind: format!("{:?}", s.kind), symbol: format!("{}::{}", module_path, name), file: s.file.clone() }
        })
        .collect();

    let json = serde_json::to_string_pretty(&json_rows)?;
    fs::write(&json_path, json)?;

    println!("Emitted:");
    println!("  - {}", md_path.display());
    println!("  - {}", json_path.display());
    Ok(())
}
