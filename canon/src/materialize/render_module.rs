use super::render_fn::{render_impl_function, render_type};
use super::render_impl::render_impl;
use super::render_struct::{render_struct, render_visibility};
use super::render_trait::render_trait;
use crate::ir::{CanonicalIr, Function, Struct, Trait, TypeKind};
use crate::layout::{LayoutFile, LayoutGraph, LayoutModule, LayoutNode};
use std::collections::{BTreeSet, HashMap};

/// Render a single named file within a module that has explicit FileNode topology.
/// Emits all impls (and their functions) whose functions are assigned to this file_id.
pub fn render_file(
    file_node: &LayoutFile,
    module: &crate::ir::Module,
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
    use_lines: &[String],
) -> String {
    let mut lines = Vec::new();
    lines.push("// Derived from Canonical IR. Do not edit.".to_owned());
    push_module_attributes(&mut lines, module);
    let mut combined_use_lines =
        collect_intramodule_types(module, file_node.id.as_str(), layout, ir);
    combined_use_lines.extend_from_slice(use_lines);
    combined_use_lines.extend(collect_external_uses(&module.id, ir, function_map));
    let combined_use_lines = sort_use_lines(combined_use_lines);
    if !combined_use_lines.is_empty() {
        lines.push(combined_use_lines.join("\n"));
    }

    // structs assigned to this file
    let layout_file_id = file_node.id.as_str();
    let mut structs: Vec<_> = ir
        .structs
        .iter()
        .filter(|s| s.module == module.id)
        .filter(|s| struct_layout_file(layout, s.id.as_str()) == Some(layout_file_id))
        .collect();
    structs.sort_by_key(|s| s.name.as_str());
    for s in &structs {
        lines.push(render_struct(s));
    }

    // traits assigned to this file (heuristic: same sort order)
    let mut traits: Vec<_> = ir
        .traits
        .iter()
        .filter(|t| t.module == module.id)
        .filter(|t| trait_layout_file(layout, t.id.as_str()) == Some(layout_file_id))
        .collect();
    traits.sort_by_key(|t| t.name.as_str());
    for t in &traits {
        lines.push(render_trait(t));
    }

    let layout_file_id = file_node.id.as_str();
    let mut impls: Vec<_> = ir
        .impls
        .iter()
        .filter(|block| block.module == module.id)
        .filter(|block| {
            block.functions.iter().any(|binding| {
                function_layout_file(layout, binding.function.as_str()) == Some(layout_file_id)
            })
        })
        .collect();
    impls.sort_by_key(|i| i.id.as_str());
    for block in impls {
        lines.push(render_impl(block, struct_map, trait_map, function_map));
    }

    // ── standalone (free) functions assigned to this file ─────────────────────
    let mut free_functions: Vec<_> = ir
        .functions
        .iter()
        .filter(|f| f.module == module.id)
        .filter(|f| f.impl_id.is_empty())
        .filter(|f| function_layout_file(layout, f.id.as_str()) == Some(layout_file_id))
        .collect();
    free_functions.sort_by_key(|f| f.name.as_str());
    for f in free_functions {
        lines.push(render_impl_function(f));
    }

    lines.join("\n\n") + "\n"
}
pub fn render_module(
    module: &crate::ir::Module,
    _layout: &LayoutGraph,
    ir: &CanonicalIr,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
) -> String {
    let mut lines = Vec::new();
    lines.push("// Derived from Canonical IR. Do not edit.".to_owned());
    push_module_attributes(&mut lines, module);
    let incoming = collect_incoming_types(ir, &module.id);
    let mut use_lines = render_use_block(&incoming);
    use_lines.extend(collect_external_uses(&module.id, ir, function_map));
    let use_lines = sort_use_lines(use_lines);
    if !use_lines.is_empty() {
        lines.push(use_lines.join("\n"));
    }
    if let Some(items) = render_module_items_block(module) {
        lines.push(items);
    }

    let mut structs: Vec<_> = ir
        .structs
        .iter()
        .filter(|s| s.module == module.id)
        .collect();
    structs.sort_by_key(|s| s.name.as_str());
    for s in structs {
        lines.push(render_struct(s));
    }

    let mut traits: Vec<_> = ir.traits.iter().filter(|t| t.module == module.id).collect();
    traits.sort_by_key(|t| t.name.as_str());
    for t in traits {
        lines.push(render_trait(t));
    }

    let mut impls: Vec<_> = ir
        .impls
        .iter()
        .filter(|i| i.module == module.id)
        .collect();
    impls.sort_by_key(|i| i.id.as_str());
    for block in impls {
        lines.push(render_impl(block, struct_map, trait_map, function_map));
    }

    // ── standalone (free) functions ───────────────────────────────────────────
    let mut free_functions: Vec<_> = ir
        .functions
        .iter()
        .filter(|f| f.module == module.id)
        .filter(|f| f.impl_id.is_empty())
        .collect();
    free_functions.sort_by_key(|f| f.name.as_str());
    for f in free_functions {
        lines.push(render_impl_function(f));
    }

    lines.join("\n\n") + "\n"
}

pub fn topo_sort_layout_files(files: &[LayoutFile]) -> Vec<LayoutFile> {
    let mut result: Vec<_> = files.to_vec();
    result.sort_by(|a, b| a.id.cmp(&b.id));
    result
}

pub fn render_module_items_block(module: &crate::ir::Module) -> Option<String> {
    let mut lines = Vec::new();
    if !module.pub_uses.is_empty() {
        for item in &module.pub_uses {
            lines.push(format!("pub use {};", item.path));
        }
    }
    if !module.constants.is_empty() {
        for constant in &module.constants {
            lines.push(format!(
                "pub const {}: {} = {};",
                constant.name,
                render_type(&constant.ty),
                constant.value_expr
            ));
        }
    }
    if !module.statics.is_empty() {
        for item in &module.statics {
            if let Some(doc) = &item.doc {
                push_doc_lines(&mut lines, doc);
            }
            let mut_kw = if item.mutable { "mut " } else { "" };
            lines.push(format!(
                "{}static {mut_kw}{}: {} = {};",
                render_visibility(item.visibility),
                item.name,
                render_type(&item.ty),
                item.value_expr
            ));
        }
    }
    if !module.type_aliases.is_empty() {
        for alias in &module.type_aliases {
            lines.push(format!(
                "pub type {} = {};",
                alias.name,
                render_type(&alias.target)
            ));
        }
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn push_doc_lines(lines: &mut Vec<String>, doc: &str) {
    for line in doc.lines() {
        if line.trim().is_empty() {
            lines.push("///".to_owned());
        } else {
            lines.push(format!("/// {}", line));
        }
    }
}

fn push_module_attributes(lines: &mut Vec<String>, module: &crate::ir::Module) {
    for attr in &module.attributes {
        lines.push(format!("#![{}]", attr));
    }
}

pub fn collect_incoming_types(ir: &CanonicalIr, module_id: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    for edge in &ir.module_edges {
        if edge.target != module_id || edge.imported_types.is_empty() {
            continue;
        }
        let source_name = ir
            .modules
            .iter()
            .find(|m| m.id == edge.source)
            .map(|m| m.name.as_str().to_string())
            .unwrap_or_else(|| edge.source.clone());
        out.push((source_name, edge.imported_types.clone()));
    }
    out
}

pub fn render_use_block(incoming: &[(String, Vec<String>)]) -> Vec<String> {
    let mut lines = BTreeSet::new();
    for (source, types) in incoming {
        if types.is_empty() {
            continue;
        }
        let prefix = normalize_use_prefix(source);
        let mut uniq = types.clone();
        uniq.sort();
        uniq.dedup();
        if uniq.len() == 1 {
            lines.insert(format!("use {prefix}::{};", uniq[0]));
        } else {
            lines.insert(format!("use {prefix}::{{{}}};", uniq.join(", ")));
        }
    }
    lines.into_iter().collect()
}

fn normalize_use_prefix(source: &str) -> String {
    // All materialized modules live at crate root.
    // Always qualify with `crate::` unless already absolute.
    if source.starts_with("::") || source.starts_with("crate::") {
        source.to_owned()
    } else {
        format!("crate::{source}")
    }
}

fn collect_intramodule_types(
    module: &crate::ir::Module,
    current_file_id: &str,
    layout: &LayoutGraph,
    ir: &CanonicalIr,
) -> Vec<String> {
    // Intramodule import generation is currently disabled.
    // Returning empty avoids generating incorrect `use super::...` paths.
    let _ = module;
    let _ = current_file_id;
    let _ = layout;
    let _ = ir;
    Vec::new()
}

fn intramodule_use_line(
    type_name: &str,
    explicit_file_id: Option<&str>,
    current_file_id: &str,
    id_to_stem: &HashMap<String, String>,
    stem_to_id: &HashMap<String, String>,
) -> Option<String> {
    let _ = type_name;
    let _ = explicit_file_id;
    let _ = current_file_id;
    let _ = id_to_stem;
    let _ = stem_to_id;
    None
}

fn build_file_maps(module: &LayoutModule) -> (HashMap<String, String>, HashMap<String, String>) {
    let mut id_to_stem = HashMap::new();
    let mut stem_to_id = HashMap::new();
    for file in &module.files {
        let stem = file_stem(&file.path);
        if stem == "mod" {
            continue;
        }
        id_to_stem.insert(file.id.clone(), stem.to_owned());
        stem_to_id.insert(stem.to_owned(), file.id.clone());
    }
    (id_to_stem, stem_to_id)
}

// derive module name from layout file id map
// removed incorrect path expansion

fn collect_external_uses(
    module_id: &str,
    ir: &CanonicalIr,
    _function_map: &HashMap<&str, &Function>,
) -> Vec<String> {
    // Fully-qualified type paths are emitted directly in signatures.
    // We do NOT auto-generate `use` statements.
    let _ = module_id;
    let _ = ir;
    Vec::new()
}

fn collect_external_types(ty: &crate::ir::TypeRef, out: &mut BTreeSet<String>) {
    if ty.kind == TypeKind::External {
        let path = ty.name.as_str();

        // Only generate `use` lines for fully-qualified paths.
        // Bare identifiers like `String`, `HashMap`, `CanonicalIr`
        // must NOT generate `use` statements.
        if path.contains("::") {
            out.insert(format!("use {};", path));
        }
    }
    for param in &ty.params {
        collect_external_types(param, out);
    }
}

fn sort_use_lines(mut lines: Vec<String>) -> Vec<String> {
    if lines.is_empty() {
        return lines;
    }
    lines.sort_by(|a, b| {
        use_line_rank(a)
            .cmp(&use_line_rank(b))
            .then_with(|| a.cmp(b))
    });
    lines.dedup();
    lines
}

fn use_line_rank(line: &str) -> u8 {
    if line.starts_with("use super::") {
        0
    } else if line.starts_with("use crate::") {
        1
    } else {
        2
    }
}

fn to_snake_case(value: &str) -> String {
    let mut out = String::new();
    let mut prev_is_lowercase_or_digit = false;
    for ch in value.chars() {
        if ch.is_ascii_uppercase() {
            if prev_is_lowercase_or_digit {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
            prev_is_lowercase_or_digit = false;
        } else {
            if ch.is_ascii_digit() && !prev_is_lowercase_or_digit {
                // keep digits grouped with previous uppercase sequences
            } else if ch.is_ascii_digit() && prev_is_lowercase_or_digit {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_is_lowercase_or_digit = ch.is_ascii_alphanumeric();
        }
    }
    out
}

fn file_stem(name: &str) -> &str {
    name.trim_end_matches(".rs")
}
fn function_layout_file<'a>(layout: &'a LayoutGraph, function_id: &str) -> Option<&'a str> {
    layout
        .routing
        .iter()
        .find(|assign| matches!(assign.node, LayoutNode::Function(ref id) if id == function_id))
        .map(|assign| assign.file_id.as_str())
}

fn struct_layout_file<'a>(layout: &'a LayoutGraph, struct_id: &str) -> Option<&'a str> {
    layout
        .routing
        .iter()
        .find(|assign| matches!(assign.node, LayoutNode::Struct(ref id) if id == struct_id))
        .map(|assign| assign.file_id.as_str())
}

fn trait_layout_file<'a>(layout: &'a LayoutGraph, trait_id: &str) -> Option<&'a str> {
    layout
        .routing
        .iter()
        .find(|assign| matches!(assign.node, LayoutNode::Trait(ref id) if id == trait_id))
        .map(|assign| assign.file_id.as_str())
}
