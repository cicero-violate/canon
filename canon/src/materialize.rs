use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use crate::ir::{
    CanonicalIr, ExternalDependency, FileNode, Function, ImplBlock, Module, ModuleEdge, Project,
    Struct, Trait, TypeRef, ValuePort, Visibility,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileTree {
    directories: BTreeSet<String>,
    files: BTreeMap<String, FileEntry>,
}

impl FileTree {
    pub fn new() -> Self {
        Self {
            directories: BTreeSet::new(),
            files: BTreeMap::new(),
        }
    }

    pub fn add_directory(&mut self, path: impl Into<String>) {
        self.directories.insert(path.into());
    }

    pub fn add_file(&mut self, path: impl Into<String>, contents: impl Into<String>) {
        self.files
            .insert(path.into(), FileEntry::new(contents.into()));
    }

    pub fn directories(&self) -> &BTreeSet<String> {
        &self.directories
    }

    pub fn files(&self) -> &BTreeMap<String, FileEntry> {
        &self.files
    }
}

impl Default for FileTree {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub contents: String,
}

impl FileEntry {
    fn new(contents: String) -> Self {
        Self { contents }
    }
}

pub fn materialize(ir: &CanonicalIr) -> FileTree {
    let mut tree = FileTree::new();
    tree.add_directory("src");

    let mut modules: Vec<_> = ir.modules.iter().collect();
    modules.sort_by_key(|m| m.name.as_str());

    let struct_map: HashMap<&str, &Struct> =
        ir.structs.iter().map(|s| (s.id.as_str(), s)).collect();
    let trait_map: HashMap<&str, &Trait> = ir.traits.iter().map(|t| (t.id.as_str(), t)).collect();
    let function_map: HashMap<&str, &Function> =
        ir.functions.iter().map(|f| (f.id.as_str(), f)).collect();

    let mut lib_rs = String::from("// Derived from Canonical IR. Do not edit.\n\n");

    for module in modules {
        let module_dir = format!("src/{}", module.name.as_str());
        tree.add_directory(module_dir.clone());
        lib_rs.push_str(&format!("pub mod {};\n", module.name.as_str()));

        if module.files.is_empty() {
            // no file topology — emit everything into mod.rs as before
            let module_contents =
                render_module(module, ir, &struct_map, &trait_map, &function_map);
            tree.add_file(format!("{module_dir}/mod.rs"), module_contents);
        } else {
            // file topology present — emit one file per FileNode
            let ordered = topo_sort_files(&module.files, &module.file_edges_as_pairs());
            let incoming_types = collect_incoming_types(ir, &module.id);

            let mut mod_rs = String::from("// Derived from Canonical IR. Do not edit.\n\n");
            for file_node in &ordered {
                let stem = file_stem(&file_node.name);
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
                // imports from other crates flow into the first non-lib file
                let use_block = if idx == 0 && !incoming_types.is_empty() {
                    render_use_block(&incoming_types)
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
    let cargo = render_cargo_toml(&ir.project, &ir.dependencies);
    tree.add_file("Cargo.toml", cargo);
    tree
}

// ── file topology helpers ────────────────────────────────────────────────────

impl crate::ir::Module {
    fn file_edges_as_pairs(&self) -> Vec<(String, String)> {
        self.file_edges
            .iter()
            .map(|e| (e.from.clone(), e.to.clone()))
            .collect()
    }
}

/// Topological sort of FileNodes by their intra-module edges.
/// Nodes with no edges are returned in declaration order.
fn topo_sort_files(
    files: &[FileNode],
    edges: &[(String, String)],
) -> Vec<FileNode> {
    use std::collections::{HashSet, VecDeque};

    let mut in_degree: HashMap<String, usize> =
        files.iter().map(|f| (f.id.clone(), 0)).collect();
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();

    for (from, to) in edges {
        adj.entry(from.clone()).or_default().push(to.clone());
        *in_degree.entry(to.clone()).or_insert(0) += 1;
    }

    let mut zero: Vec<String> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(id, _)| id.clone())
        .collect();
    zero.sort();

    let mut queue: VecDeque<String> = zero.into();

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

    // append any remaining (cycles or disconnected nodes)
    let seen: HashSet<String> = result.iter().map(|f| f.id.clone()).collect();
    for file in files {
        if !seen.contains(&file.id) {
            result.push(file.clone());
        }
    }

    result
}

/// Collect all types imported into `module_id` from other modules.
fn collect_incoming_types(ir: &CanonicalIr, module_id: &str) -> Vec<(String, Vec<String>)> {
    let mut out: Vec<(String, Vec<String>)> = Vec::new();

    for edge in &ir.module_edges {
        if edge.target != module_id || edge.imported_types.is_empty() {
            continue;
        }

        let source_name: String = ir
            .modules
            .iter()
            .find(|m| m.id == edge.source)
            .map(|m| m.name.as_str().to_string())
            .unwrap_or_else(|| edge.source.clone());

        out.push((source_name, edge.imported_types.clone()));
    }

    out
}

/// Render `use crate_name::{TypeA, TypeB};` lines.
fn render_use_block(incoming: &[(String, Vec<String>)]) -> String {
    let mut out = String::new();

    for (source, types) in incoming {
        let slugged: String = source
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect();

        if types.len() == 1 {
            out.push_str(&format!("use {}::{};\n", slugged, types[0]));
        } else {
            out.push_str(&format!("use {}::{{{}}};\n", slugged, types.join(", ")));
        }
    }

    out.push('\n');
    out
}

fn file_stem(name: &str) -> &str {
    name.trim_end_matches(".rs")
}

fn render_module(
    module: &crate::ir::Module,
    ir: &CanonicalIr,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
) -> String {
    let mut lines = Vec::new();
    lines.push("// Derived from Canonical IR. Do not edit.".to_owned());

    let mut structs: Vec<_> = ir
        .structs
        .iter()
        .filter(|s| s.module == module.id)
        .collect();
    structs.sort_by_key(|s| s.name.as_str());

    for structure in structs {
        lines.push(render_struct(structure));
    }

    let mut traits: Vec<_> = ir.traits.iter().filter(|t| t.module == module.id).collect();
    traits.sort_by_key(|t| t.name.as_str());
    for trait_def in traits {
        lines.push(render_trait(trait_def));
    }

    let mut impls: Vec<_> = ir
        .impl_blocks
        .iter()
        .filter(|i| i.module == module.id)
        .collect();
    impls.sort_by_key(|i| i.id.as_str());

    for block in impls {
        lines.push(render_impl(block, struct_map, trait_map, function_map));
    }

    lines.join("\n\n") + "\n"
}

fn render_struct(structure: &Struct) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{}struct {} {{",
        render_visibility(structure.visibility),
        structure.name
    ));

    let mut fields = structure.fields.clone();
    fields.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    if fields.is_empty() {
        lines.push(String::from("    // no fields"));
    } else {
        for field in fields {
            lines.push(format!(
                "    {}{}: {},",
                render_visibility(field.visibility),
                field.name,
                render_type(&field.ty)
            ));
        }
    }

    lines.push("}".to_owned());
    lines.join("\n")
}

fn render_trait(trait_def: &Trait) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{}trait {} {{",
        render_visibility(trait_def.visibility),
        trait_def.name
    ));

    let mut functions = trait_def.functions.clone();
    functions.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
    if functions.is_empty() {
        lines.push(String::from("    // no capabilities"));
    } else {
        for func in functions {
            lines.push(format!(
                "    fn {}{};",
                func.name,
                render_fn_signature(&func.inputs, &func.outputs)
            ));
        }
    }

    lines.push("}".to_owned());
    lines.join("\n")
}

fn render_impl(
    block: &ImplBlock,
    struct_map: &HashMap<&str, &Struct>,
    trait_map: &HashMap<&str, &Trait>,
    function_map: &HashMap<&str, &Function>,
) -> String {
    let struct_name = struct_map
        .get(block.struct_id.as_str())
        .map(|s| s.name.as_str())
        .unwrap_or("<UnknownStruct>");
    let trait_name = trait_map
        .get(block.trait_id.as_str())
        .map(|t| t.name.as_str())
        .unwrap_or("<UnknownTrait>");

    let mut lines = Vec::new();
    lines.push(format!("impl {} for {} {{", trait_name, struct_name));

    let mut bindings = block.functions.clone();
    bindings.sort_by(|a, b| a.trait_fn.as_str().cmp(b.trait_fn.as_str()));

    if bindings.is_empty() {
        lines.push(String::from("    // no bindings"));
    } else {
        for binding in bindings {
            if let Some(function) = function_map.get(binding.function.as_str()) {
                lines.push(render_impl_function(function));
            } else {
                lines.push(format!(
                    "    // missing function `{}` for trait fn `{}`",
                    binding.function, binding.trait_fn
                ));
            }
        }
    }

    lines.push("}".to_owned());
    lines.join("\n\n")
}

fn render_impl_function(function: &Function) -> String {
    let mut output = String::new();
    output.push_str("    ");
    output.push_str(render_visibility(function.visibility));
    output.push_str("fn ");
    output.push_str(function.name.as_str());
    output.push_str(&render_fn_signature(&function.inputs, &function.outputs));
    output.push_str(" {\n");
    output.push_str("        // Invoke Canon runtime interpreter (generated stub)\n");
    output.push_str(&format!(
        "        canon_runtime::execute_function(\"{}\");\n",
        function.id
    ));
    output.push_str("    }");
    output
}

fn render_fn_signature(inputs: &[ValuePort], outputs: &[ValuePort]) -> String {
    let params = inputs
        .iter()
        .map(|input| format!("{}: {}", input.name, render_type(&input.ty)))
        .collect::<Vec<_>>()
        .join(", ");

    let mut signature = String::new();
    signature.push('(');
    signature.push_str(&params);
    signature.push(')');

    if let Some(output_ty) = render_output_types(outputs) {
        signature.push_str(" -> ");
        signature.push_str(&output_ty);
    }

    signature
}

fn render_output_types(outputs: &[ValuePort]) -> Option<String> {
    match outputs.len() {
        0 => None,
        1 => Some(render_type(&outputs[0].ty)),
        _ => Some(format!(
            "({})",
            outputs
                .iter()
                .map(|output| render_type(&output.ty))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn render_type(ty: &TypeRef) -> String {
    ty.name.as_str().to_owned()
}

fn render_visibility(vis: Visibility) -> &'static str {
    match vis {
        Visibility::Public => "pub ",
        Visibility::Private => "",
    }
}

pub fn write_file_tree(tree: &FileTree, root: impl AsRef<Path>) -> std::io::Result<()> {
    let root = root.as_ref();
    fs::create_dir_all(root)?;
    for dir in tree.directories() {
        let path = root.join(dir);
        fs::create_dir_all(path)?;
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

fn render_cargo_toml(project: &Project, dependencies: &[ExternalDependency]) -> String {
    let mut out = String::new();
    out.push_str("# Derived from Canonical IR. Do not edit.\n");
    out.push_str("[package]\n");
    out.push_str(&format!("name = \"{}\"\n", project.name));
    out.push_str(&format!("version = \"{}\"\n", project.version));
    out.push_str("edition = \"2024\"\n\n");
    out.push_str("[dependencies]\n");
    if dependencies.is_empty() {
        out.push_str("# no external dependencies\n");
    } else {
        let mut deps = dependencies.to_vec();
        deps.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
        for dep in deps {
            if dep.name == dep.source {
                out.push_str(&format!("{} = \"{}\"\n", dep.name, dep.version));
            } else {
                out.push_str(&format!(
                    "{} = {{ package = \"{}\", version = \"{}\" }}\n",
                    dep.name, dep.source, dep.version
                ));
            }
        }
    }
    out
}
