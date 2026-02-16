use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::ir::{CanonicalIr, Function, Struct, Trait};
use crate::layout::LayoutGraph;
use render_module::render_file;

pub use self::file_tree::{FileEntry, FileTree};
pub use self::render_fn::render_impl_function;

mod file_tree;
mod render_cargo;
mod render_common;
pub mod render_fn;
mod render_impl;
mod render_module;
mod render_struct;
mod render_trait;

pub struct MaterializeResult {
    pub tree: FileTree,
    pub file_hashes: HashMap<String, String>,
}

pub fn materialize(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    existing_root: Option<&Path>,
) -> MaterializeResult {
    let mut tree = FileTree::new();
    tree.add_directory("src");
    let mut next_hashes = HashMap::new();

    let mut modules: Vec<_> = ir.modules.iter().collect();
    modules.sort_by_key(|m| m.name.as_str());

    let struct_map: HashMap<&str, &Struct> =
        ir.structs.iter().map(|s| (s.id.as_str(), s)).collect();
    let trait_map: HashMap<&str, &Trait> = ir.traits.iter().map(|t| (t.id.as_str(), t)).collect();
    let function_map: HashMap<&str, &Function> =
        ir.functions.iter().map(|f| (f.id.as_str(), f)).collect();

    let mut lib_rs = String::from("// Derived from Canonical IR. Do not edit.\n\n");

    for module in &modules {
        let module_dir = format!("src/{}", module.name.as_str());
        tree.add_directory(module_dir.clone());
        lib_rs.push_str(&format!("pub mod {};\n", module.name.as_str()));

        let layout_module = layout.modules.iter().find(|m| m.id == module.id);
        if layout_module.map(|m| m.files.is_empty()).unwrap_or(true) {
            let contents = render_module::render_module(
                module,
                layout,
                ir,
                &struct_map,
                &trait_map,
                &function_map,
            );
            finalize_file(
                &mut tree,
                &mut next_hashes,
                &ir.file_hashes,
                &format!("{module_dir}/mod.rs"),
                contents,
                existing_root,
            );
        } else {
            let layout_module = layout_module.expect("layout module missing");
            let ordered = render_module::topo_sort_layout_files(&layout_module.files);
            let incoming = render_module::collect_incoming_types(ir, &module.id);
            let use_lines = render_module::render_use_block(&incoming);

            let mut mod_sections = vec!["// Derived from Canonical IR. Do not edit.".to_owned()];
            if !module.attributes.is_empty() {
                let attrs = module
                    .attributes
                    .iter()
                    .map(|attr| format!("#![{}]", attr))
                    .collect::<Vec<_>>()
                    .join("\n");
                mod_sections.push(attrs);
            }
            if let Some(items) = render_module::render_module_items_block(module) {
                mod_sections.push(items);
            }
            let mut submods = Vec::new();
            // Support nested layout paths (e.g. "agent/bootstrap.rs")
            use std::collections::BTreeMap;
            let mut dir_children: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

            for f in &ordered {
                let path = f.path.as_str();
                let parts: Vec<&str> = path.split('/').collect();

                if parts.len() == 1 {
                    let stem = file_stem(parts[0]);
                    if stem != "lib" && stem != "mod" {
                        submods.push(format!("pub mod {};", stem));
                    }
                } else {
                    let dir = parts[0].to_string();
                    let file = file_stem(parts[1]).to_string();
                    dir_children.entry(dir).or_default().insert(file);
                }
            }

            // Emit top-level directory modules
            for dir in dir_children.keys() {
                submods.push(format!("pub mod {};", dir));
            }

            if !submods.is_empty() {
                mod_sections.push(submods.join("\n"));
            }
            let mod_rs = mod_sections.join("\n\n") + "\n";
            finalize_file(
                &mut tree,
                &mut next_hashes,
                &ir.file_hashes,
                &format!("{module_dir}/mod.rs"),
                mod_rs,
                existing_root,
            );

            let empty_use: Vec<String> = Vec::new();
            for (idx, file_node) in ordered.iter().enumerate() {
                let stem = file_stem(&file_node.path);
                let path = file_node.path.as_str();
                let parts: Vec<&str> = path.split('/').collect();

                let use_block: &[String] = if idx == 0 { &use_lines } else { &empty_use };
                let contents = render_file(
                    file_node,
                    module,
                    ir,
                    layout,
                    &struct_map,
                    &trait_map,
                    &function_map,
                    use_block,
                );

                if parts.len() == 1 {
                    if stem == "lib" || stem == "mod" {
                        continue;
                    }
                    let file = if parts[0].ends_with(".rs") {
                        parts[0].to_string()
                    } else {
                        format!("{}.rs", parts[0])
                    };
                    finalize_file(
                        &mut tree,
                        &mut next_hashes,
                        &ir.file_hashes,
                        &format!("{module_dir}/{file}"),
                        contents,
                        existing_root,
                    );
                } else {
                    let dir = parts[0];
                    let file = if parts[1].ends_with(".rs") {
                        parts[1].to_string()
                    } else {
                        format!("{}.rs", parts[1])
                    };

                    tree.add_directory(format!("{module_dir}/{dir}"));

                    finalize_file(
                        &mut tree,
                        &mut next_hashes,
                        &ir.file_hashes,
                        &format!("{module_dir}/{dir}/{file}"),
                        contents,
                        existing_root,
                    );
                }
            }

            // Emit nested mod.rs files for directories
            for (dir, children) in dir_children {
                let nested_mod_path = format!("{module_dir}/{dir}/mod.rs");

                let mut nested_lines =
                    vec!["// Derived from Canonical IR. Do not edit.".to_owned()];

                for child in children {
                    if child != "mod" {
                        nested_lines.push(format!("pub mod {};", child));
                    }
                }

                let nested_contents = nested_lines.join("\n") + "\n";

                finalize_file(
                    &mut tree,
                    &mut next_hashes,
                    &ir.file_hashes,
                    &nested_mod_path,
                    nested_contents,
                    existing_root,
                );
            }
        }
    }

    finalize_file(
        &mut tree,
        &mut next_hashes,
        &ir.file_hashes,
        "src/lib.rs",
        lib_rs,
        existing_root,
    );
    let cargo = render_cargo::render_cargo_toml(&ir.project, &ir.dependencies);
    finalize_file(
        &mut tree,
        &mut next_hashes,
        &ir.file_hashes,
        "Cargo.toml",
        cargo,
        existing_root,
    );
    MaterializeResult {
        tree,
        file_hashes: next_hashes,
    }
}

fn finalize_file(
    tree: &mut FileTree,
    next_hashes: &mut HashMap<String, String>,
    previous_hashes: &HashMap<String, String>,
    path: &str,
    rendered: String,
    existing_root: Option<&Path>,
) {
    let content = apply_preserve_regions(existing_root, path, rendered);
    let hash = hash_contents(&content);
    next_hashes.insert(path.to_owned(), hash.clone());
    let unchanged = previous_hashes
        .get(path)
        .map(|prev| prev == &hash)
        .unwrap_or(false);
    if unchanged {
        return;
    }
    tree.add_file(path.to_owned(), content);
}

fn hash_contents(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn apply_preserve_regions(
    existing_root: Option<&Path>,
    relative_path: &str,
    rendered: String,
) -> String {
    let Some(root) = existing_root else {
        return rendered;
    };
    let existing_path = root.join(relative_path);
    let Ok(existing) = fs::read_to_string(&existing_path) else {
        return rendered;
    };
    splice_preserve_blocks(&rendered, &existing)
}

fn splice_preserve_blocks(new_content: &str, existing_content: &str) -> String {
    let preserved = extract_preserve_blocks(existing_content);
    if preserved.is_empty() {
        return new_content.to_owned();
    }
    let new_lines = split_lines_with_endings(new_content);
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < new_lines.len() {
        let line = new_lines[idx].clone();
        idx += 1;
        if let Some(label) = parse_preserve_start(&line) {
            out.push(line);
            let mut block = Vec::new();
            let mut closed = false;
            while idx < new_lines.len() {
                let candidate = new_lines[idx].clone();
                idx += 1;
                if let Some(end_label) = parse_preserve_end(&candidate) {
                    if end_label == label {
                        if let Some(existing_block) = preserved.get(&label) {
                            out.extend(existing_block.clone());
                        } else {
                            out.extend(block.clone());
                        }
                        out.push(candidate);
                        closed = true;
                        break;
                    } else {
                        block.push(candidate);
                    }
                } else {
                    block.push(candidate);
                }
            }
            if !closed {
                out.extend(block);
            }
        } else {
            out.push(line);
        }
    }
    out.concat()
}

fn extract_preserve_blocks(content: &str) -> HashMap<String, Vec<String>> {
    let mut blocks = HashMap::new();
    let lines = split_lines_with_endings(content);
    let mut idx = 0;
    while idx < lines.len() {
        let line = lines[idx].clone();
        idx += 1;
        if let Some(label) = parse_preserve_start(&line) {
            let mut block = Vec::new();
            while idx < lines.len() {
                let candidate = lines[idx].clone();
                idx += 1;
                if let Some(end_label) = parse_preserve_end(&candidate) {
                    if end_label == label {
                        blocks.insert(label.clone(), block);
                        break;
                    } else {
                        block.push(candidate);
                    }
                } else {
                    block.push(candidate);
                }
            }
        }
    }
    blocks
}

fn split_lines_with_endings(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut lines: Vec<String> = content
        .split_inclusive('\n')
        .map(|line| line.to_string())
        .collect();
    if !content.ends_with('\n') {
        let remainder = content
            .rsplit_once('\n')
            .map(|(_, trailing)| trailing)
            .unwrap_or(content);
        if remainder != lines.last().map(|s| s.as_str()).unwrap_or("") {
            lines.push(remainder.to_string());
        }
    }
    lines
}

fn parse_preserve_start(line: &str) -> Option<String> {
    parse_preserve_marker(line, "// canon:preserve:start")
}

fn parse_preserve_end(line: &str) -> Option<String> {
    parse_preserve_marker(line, "// canon:preserve:end")
}

fn parse_preserve_marker(line: &str, marker: &str) -> Option<String> {
    let trimmed = line.trim();
    trimmed
        .strip_prefix(marker)
        .map(|rest| rest.trim().to_string())
}

pub fn write_file_tree(tree: &FileTree, root: impl AsRef<Path>) -> std::io::Result<()> {
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    for dir in tree.directories() {
        fs::create_dir_all(root.join(dir))?;
    }
    for (path, entry) in tree.files() {
        let file_path = root.join(path);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(file_path, &entry.contents)?;
    }
    Ok(())
}

fn file_stem(name: &str) -> &str {
    // strip directories and extension
    let last = name.rsplit('/').next().unwrap_or(name);
    last.trim_end_matches(".rs")
}
