use std::path::Path;

use crate::layout::{LayoutMap, SemanticGraph};

use super::IngestError;
use super::parser::ParsedWorkspace;

pub(crate) mod ast_lower;
pub(crate) mod edges;
pub(crate) mod functions;
pub(crate) mod layout;
pub(crate) mod modules;
pub(crate) mod types;

pub(crate) struct ModulesBuild {
    pub modules: Vec<crate::ir::Module>,
    pub module_lookup: std::collections::HashMap<String, String>,
    pub file_lookup: std::collections::HashMap<String, String>,
}

/// Convert parsed files into semantic + layout graphs.
pub(crate) fn build_layout_map(
    _root: &Path,
    parsed: ParsedWorkspace,
) -> Result<LayoutMap, IngestError> {
    let ModulesBuild {
        modules,
        module_lookup,
        file_lookup,
    } = modules::build_modules(&parsed)?;
    let module_edges = modules::build_module_edges(&parsed, &module_lookup);
    let mut layout_acc = layout::LayoutAccumulator::default();
    // Register every file so layout can build LayoutFile entries per module
    for file in &parsed.files {
        let path = file.path_string();
        let file_id = format!("file.{}", {
            let mut out = String::new();
            for ch in path.chars() {
                if ch.is_ascii_alphanumeric() {
                    out.push(ch.to_ascii_lowercase());
                } else {
                    out.push('_');
                }
            }
            if out.is_empty() {
                "root".to_string()
            } else {
                out
            }
        });
        let mk = modules::module_key(file);
        if let Some(module_id) = module_lookup.get(&mk) {
            let use_lines = collect_use_lines(file);
            layout_acc.register_file(module_id, &file_id, &path, use_lines);
        }
    }
    let structs = functions::build_structs(&parsed, &module_lookup, &file_lookup, &mut layout_acc);
    let enums = functions::build_enums(&parsed, &module_lookup, &file_lookup, &mut layout_acc);
    let traits = functions::build_traits(&parsed, &module_lookup, &file_lookup, &mut layout_acc);
    let (impl_blocks, fns) = functions::build_impls_and_functions(
        &parsed,
        &module_lookup,
        &file_lookup,
        &mut layout_acc,
    );
    let call_edges = edges::build_call_edges(&parsed, &module_lookup, &fns);
    let semantic = SemanticGraph {
        modules,
        structs,
        enums,
        traits,
        impls: impl_blocks,
        functions: fns,
        module_edges,
        call_edges,
        tick_graphs: Vec::new(),
        system_graphs: Vec::new(),
    };
    let layout = layout_acc.into_graph(&semantic.modules);
    Ok(LayoutMap { semantic, layout })
}

/// Render every `use` item in a parsed file to a string, preserving
/// declaration order. These are stored verbatim in `LayoutFile.use_block`
/// so the materializer can emit them before its synthesised imports.
fn collect_use_lines(file: &super::parser::ParsedFile) -> Vec<String> {
    file.ast
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Use(item_use) = item {
                Some(types::render_use_item(item_use))
            } else {
                None
            }
        })
        .collect()
}
