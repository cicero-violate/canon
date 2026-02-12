use std::collections::HashMap;
use std::path::Path;
use std::fs;

use crate::ir::{CanonicalIr, Function, Struct, Trait};

pub use self::file_tree::{FileEntry, FileTree};

mod file_tree;
mod render_cargo;
mod render_fn;
mod render_impl;
mod render_module;
mod render_struct;
mod render_trait;

pub fn materialize(ir: &CanonicalIr) -> FileTree {
    let mut tree = FileTree::new();
    tree.add_directory("src");

    let mut modules: Vec<_> = ir.modules.iter().collect();
    modules.sort_by_key(|m| m.name.as_str());

    let struct_map: HashMap<&str, &Struct> =
        ir.structs.iter().map(|s| (s.id.as_str(), s)).collect();
    let trait_map: HashMap<&str, &Trait> =
        ir.traits.iter().map(|t| (t.id.as_str(), t)).collect();
    let function_map: HashMap<&str, &Function> =
        ir.functions.iter().map(|f| (f.id.as_str(), f)).collect();

    let mut lib_rs = String::from("// Derived from Canonical IR. Do not edit.\n\n");

    for module in &modules {
        let module_dir = format!("src/{}", module.name.as_str());
        tree.add_directory(module_dir.clone());
        lib_rs.push_str(&format!("pub mod {};\n", module.name.as_str()));

        if module.files.is_empty() {
            let contents = render_module::render_module(
                module, ir, &struct_map, &trait_map, &function_map,
            );
            tree.add_file(format!("{module_dir}/mod.rs"), contents);
        } else {
            let ordered = render_module::topo_sort_files(
                &module.files,
                &module.file_edges_as_pairs(),
            );
            let incoming = render_module::collect_incoming_types(ir, &module.id);

            let mut mod_rs = String::from("// Derived from Canonical IR. Do not edit.\n\n");
            for f in &ordered {
                let stem = file_stem(&f.name);
                if stem != "lib" && stem != "mod" {
                    mod_rs.push_str(&format!("pub mod {};\n", stem));
                }
            }
            tree.add_file(format!("{module_dir}/mod.rs"), mod_rs);

            for (idx, file_node) in ordered.iter().enumerate() {
                let stem = file_stem(&file_node.name);
                if stem == "lib" || stem == "mod" {
                    continue;
                }
                let use_block = if idx == 0 && !incoming.is_empty() {
                    render_module::render_use_block(&incoming)
                } else {
                    String::new()
                };
                let contents = format!(
                    "// Derived from Canonical IR. Do not edit.\n\n{use_block}// {}\n",
                    file_node.name
                );
                tree.add_file(format!("{module_dir}/{}", file_node.name), contents);
            }
        }
    }

    tree.add_file("src/lib.rs", lib_rs);
    let cargo = render_cargo::render_cargo_toml(&ir.project, &ir.dependencies);
    tree.add_file("Cargo.toml", cargo);
    tree
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
    name.trim_end_matches(".rs")
}

impl crate::ir::Module {
    pub fn file_edges_as_pairs(&self) -> Vec<(String, String)> {
        self.file_edges
            .iter()
            .map(|e| (e.from.clone(), e.to.clone()))
            .collect()
    }
}
