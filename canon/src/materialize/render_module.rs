use std::collections::{HashMap, VecDeque};
use crate::ir::{CanonicalIr, FileNode, Function, Struct, Trait};
use super::render_impl::render_impl;
use super::render_struct::render_struct;
use super::render_trait::render_trait;

pub fn render_module(
    module: &crate::ir::Module,
    ir: &CanonicalIr,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
) -> String {
    let mut lines = Vec::new();
    lines.push("// Derived from Canonical IR. Do not edit.".to_owned());

    let mut structs: Vec<_> = ir.structs.iter().filter(|s| s.module == module.id).collect();
    structs.sort_by_key(|s| s.name.as_str());
    for s in structs {
        lines.push(render_struct(s));
    }

    let mut traits: Vec<_> = ir.traits.iter().filter(|t| t.module == module.id).collect();
    traits.sort_by_key(|t| t.name.as_str());
    for t in traits {
        lines.push(render_trait(t));
    }

    let mut impls: Vec<_> = ir.impl_blocks.iter().filter(|i| i.module == module.id).collect();
    impls.sort_by_key(|i| i.id.as_str());
    for block in impls {
        lines.push(render_impl(block, struct_map, trait_map, function_map));
    }

    lines.join("\n\n") + "\n"
}

pub fn topo_sort_files(
    files: &[FileNode],
    edges: &[(String, String)],
) -> Vec<FileNode> {
    let mut in_degree: HashMap<String, usize> =
        files.iter().map(|f| (f.id.clone(), 0)).collect();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();

    for (from, to) in edges {
        adj.entry(from.clone()).or_default().push(to.clone());
        *in_degree.entry(to.clone()).or_insert(0) += 1;
    }

    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| id.clone())
        .collect();
    queue.make_contiguous().sort();

    let id_to_file: HashMap<String, FileNode> =
        files.iter().map(|f| (f.id.clone(), f.clone())).collect();

    let mut result = Vec::new();
    while let Some(id) = queue.pop_front() {
        if let Some(file) = id_to_file.get(&id) {
            result.push(file.clone());
        }
        if let Some(neighbors) = adj.get(&id) {
            let mut next = neighbors.clone();
            next.sort();
            for nb in next {
                let deg = in_degree.entry(nb.clone()).or_insert(0);
                *deg = deg.saturating_sub(1);
                if *deg == 0 {
                    queue.push_back(nb);
                }
            }
        }
    }

    let seen: std::collections::HashSet<String> =
        result.iter().map(|f| f.id.clone()).collect();
    for file in files {
        if !seen.contains(&file.id) {
            result.push(file.clone());
        }
    }
    result
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

pub fn render_use_block(incoming: &[(String, Vec<String>)]) -> String {
    let mut out = String::new();
    for (source, types) in incoming {
        let slug: String = source
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
            .collect();
        if types.len() == 1 {
            out.push_str(&format!("use{}::{};\n", slug, types[0]));
        } else {
            out.push_str(&format!("use {}::{{{}}};\n", slug, types.join(", ")));
        }
    }
    out.push('\n');
    out
}
