#![cfg_attr(feature = "rustc_frontend", feature(rustc_private))]

#[cfg(feature = "rustc_frontend")]
extern crate rustc_driver;

use rename::collect_names;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project_path = Path::new("/workspace/ai_sandbox/canon_workspace/canon");

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
            module_path = module_path
                .rsplit_once("::")
                .map(|(parent, _)| parent.to_string())
                .unwrap_or(module_path);
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
    output.push_str(&format!(
        "| {:<max_kind$} | {:<max_symbol$} | {:<max_file$} |\n",
        "Kind",
        "Symbol",
        "File",
        max_kind = max_kind,
        max_symbol = max_symbol,
        max_file = max_file
    ));

    // Separator
    output.push_str(&format!(
        "|-{:-<max_kind$}-|-{:-<max_symbol$}-|-{:-<max_file$}-|\n",
        "",
        "",
        "",
        max_kind = max_kind,
        max_symbol = max_symbol,
        max_file = max_file
    ));

    // Rows
    for (kind, symbol_path, file) in rows {
        output.push_str(&format!(
            "| {:<max_kind$} | {:<max_symbol$} | {:<max_file$} |\n",
            kind,
            symbol_path,
            file,
            max_kind = max_kind,
            max_symbol = max_symbol,
            max_file = max_file
        ));
    }

    fs::write(
        "/workspace/ai_sandbox/canon_workspace/canon/SYMBOLS.md",
        output,
    )?;

    println!("Emitted SYMBOLS.md");
    Ok(())
}
