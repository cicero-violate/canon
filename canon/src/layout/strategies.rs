use std::collections::HashMap;
use super::{
    FileBinding, LayoutFile, FileTopology, LayoutModule, LayoutNode, LayoutStrategy,
    ParsedModel,
};
/// Preserves the original layout graph exactly as provided.
pub struct OriginalLayoutStrategy {
    base: FileTopology,
}
impl OriginalLayoutStrategy {
    pub fn new(base: FileTopology) -> Self {
        Self { base }
    }
}
impl LayoutStrategy for OriginalLayoutStrategy {
    fn name(&self) -> &'static str {
        "original"
    }
    fn description(&self) -> &'static str {
        "Preserve the ingested layout graph without modification."
    }
    fn plan(&self, _semantic: &ParsedModel) -> FileTopology {
        self.base.clone()
    }
}
/// Routes all module contents into a single `mod.rs` file per module.
pub struct SingleFileLayoutStrategy;
impl LayoutStrategy for SingleFileLayoutStrategy {
    fn name(&self) -> &'static str {
        "single_file"
    }
    fn description(&self) -> &'static str {
        "Place every struct/trait/enum/impl/function for a module into mod.rs."
    }
    fn plan(&self, semantic: &ParsedModel) -> FileTopology {
        let mut modules = Vec::new();
        let mut routing = Vec::new();
        for module in &semantic.modules {
            let file_id = default_file_id(module.id.as_str());
            modules
                .push(LayoutModule {
                    id: module.id.clone(),
                    name: module.name.clone(),
                    files: vec![
                        LayoutFile { id : file_id.clone(), path : "mod.rs".to_owned(),
                        use_block : Vec::new() }
                    ],
                    imports: Vec::new(),
                });
            route_all_nodes(
                semantic,
                module.id.as_str(),
                &file_id,
                "LAY-005: single-file strategy",
                &mut routing,
            );
        }
        FileTopology { modules, routing }
    }
}
/// Routes each struct/trait/enum into its own file (plus `mod.rs` for glue).
pub struct PerTypeLayoutStrategy;
impl LayoutStrategy for PerTypeLayoutStrategy {
    fn name(&self) -> &'static str {
        "per_type"
    }
    fn description(&self) -> &'static str {
        "Create one file per struct/trait/enum while keeping mod.rs as the root."
    }
    fn plan(&self, semantic: &ParsedModel) -> FileTopology {
        let mut modules = Vec::new();
        let mut routing = Vec::new();
        let mut impl_to_struct: HashMap<&str, &str> = HashMap::new();
        for block in &semantic.impls {
            impl_to_struct.insert(block.id.as_str(), block.struct_id.as_str());
        }
        for module in &semantic.modules {
            let mut files = vec![
                LayoutFile { id : default_file_id(module.id.as_str()), path : "mod.rs"
                .to_owned(), use_block : Vec::new() }
            ];
            let mut named_files: HashMap<String, String> = HashMap::new();
            let mut file_for_struct: HashMap<&str, String> = HashMap::new();
            for structure in semantic.structs.iter().filter(|s| s.module == module.id) {
                let slug = slugify(structure.name.as_str());
                let file_id = ensure_named_file(
                    &mut files,
                    &mut named_files,
                    module.id.as_str(),
                    &slug,
                    format!("{}.rs", slug),
                );
                file_for_struct.insert(structure.id.as_str(), file_id.clone());
                routing
                    .push(FileBinding {
                        node: LayoutNode::Struct(structure.id.clone()),
                        file_id,
                        rationale: "LAY-005: per-type strategy".to_owned(),
                    });
            }
            for enumeration in semantic.enums.iter().filter(|e| e.module == module.id) {
                let slug = slugify(enumeration.name.as_str());
                let file_id = ensure_named_file(
                    &mut files,
                    &mut named_files,
                    module.id.as_str(),
                    &slug,
                    format!("{}.rs", slug),
                );
                routing
                    .push(FileBinding {
                        node: LayoutNode::Enum(enumeration.id.clone()),
                        file_id,
                        rationale: "LAY-005: per-type strategy".to_owned(),
                    });
            }
            for tr in semantic.traits.iter().filter(|t| t.module == module.id) {
                let slug = slugify(tr.name.as_str());
                let file_id = ensure_named_file(
                    &mut files,
                    &mut named_files,
                    module.id.as_str(),
                    &slug,
                    format!("{}.rs", slug),
                );
                routing
                    .push(FileBinding {
                        node: LayoutNode::Trait(tr.id.clone()),
                        file_id,
                        rationale: "LAY-005: per-type strategy".to_owned(),
                    });
            }
            for block in semantic.impls.iter().filter(|b| b.module == module.id) {
                let file_id = file_for_struct
                    .get(block.struct_id.as_str())
                    .cloned()
                    .unwrap_or_else(|| default_file_id(module.id.as_str()));
                routing
                    .push(FileBinding {
                        node: LayoutNode::Impl(block.id.clone()),
                        file_id,
                        rationale: "LAY-005: per-type strategy".to_owned(),
                    });
            }
            for function in semantic.functions.iter().filter(|f| f.module == module.id) {
                let file_id = if let Some(struct_id) = impl_to_struct
                    .get(function.impl_id.as_str())
                {
                    file_for_struct
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_else(|| default_file_id(module.id.as_str()))
                } else {
                    default_file_id(module.id.as_str())
                };
                routing
                    .push(FileBinding {
                        node: LayoutNode::Function(function.id.clone()),
                        file_id,
                        rationale: "LAY-005: per-type strategy".to_owned(),
                    });
            }
            modules
                .push(LayoutModule {
                    id: module.id.clone(),
                    name: module.name.clone(),
                    files,
                    imports: Vec::new(),
                });
        }
        FileTopology { modules, routing }
    }
}
fn route_all_nodes(
    semantic: &ParsedModel,
    module_id: &str,
    file_id: &str,
    rationale: &str,
    routing: &mut Vec<FileBinding>,
) {
    for structure in semantic.structs.iter().filter(|s| s.module.as_str() == module_id) {
        routing
            .push(FileBinding {
                node: LayoutNode::Struct(structure.id.clone()),
                file_id: file_id.to_owned(),
                rationale: rationale.to_owned(),
            });
    }
    for enumeration in semantic.enums.iter().filter(|e| e.module.as_str() == module_id) {
        routing
            .push(FileBinding {
                node: LayoutNode::Enum(enumeration.id.clone()),
                file_id: file_id.to_owned(),
                rationale: rationale.to_owned(),
            });
    }
    for tr in semantic.traits.iter().filter(|t| t.module.as_str() == module_id) {
        routing
            .push(FileBinding {
                node: LayoutNode::Trait(tr.id.clone()),
                file_id: file_id.to_owned(),
                rationale: rationale.to_owned(),
            });
    }
    for block in semantic.impls.iter().filter(|b| b.module.as_str() == module_id) {
        routing
            .push(FileBinding {
                node: LayoutNode::Impl(block.id.clone()),
                file_id: file_id.to_owned(),
                rationale: rationale.to_owned(),
            });
    }
    for function in semantic.functions.iter().filter(|f| f.module.as_str() == module_id)
    {
        routing
            .push(FileBinding {
                node: LayoutNode::Function(function.id.clone()),
                file_id: file_id.to_owned(),
                rationale: rationale.to_owned(),
            });
    }
}
fn ensure_named_file(
    files: &mut Vec<LayoutFile>,
    named_files: &mut HashMap<String, String>,
    module_id: &str,
    slug: &str,
    path: String,
) -> String {
    if let Some(existing) = named_files.get(slug) {
        return existing.clone();
    }
    let file_id = format!("file.{}.{}", sanitize_module(module_id), slug);
    files
        .push(LayoutFile {
            id: file_id.clone(),
            path,
            use_block: Vec::new(),
        });
    named_files.insert(slug.to_owned(), file_id.clone());
    file_id
}
fn default_file_id(module_id: &str) -> String {
    format!("file.{}.mod", sanitize_module(module_id))
}
fn sanitize_module(module_id: &str) -> String {
    module_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}
fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' })
        .collect()
}
/// Fully atomic layout:
/// Every semantic node gets its own file.
pub struct FlatNodeLayoutStrategy;
impl LayoutStrategy for FlatNodeLayoutStrategy {
    fn name(&self) -> &'static str {
        "flat_nodes"
    }
    fn description(&self) -> &'static str {
        "Every struct/enum/trait/impl/function is placed into its own file."
    }
    fn plan(&self, semantic: &ParsedModel) -> FileTopology {
        let mut modules = Vec::new();
        let mut routing = Vec::new();
        let mut sorted_modules = semantic.modules.clone();
        sorted_modules.sort_by(|a, b| a.id.cmp(&b.id));
        for module in &sorted_modules {
            let mut files = Vec::new();
            let mut structs: Vec<_> = semantic
                .structs
                .iter()
                .filter(|s| s.module == module.id)
                .cloned()
                .collect();
            structs.sort_by(|a, b| a.id.cmp(&b.id));
            for structure in structs {
                let slug = slugify(structure.name.as_str());
                let file_id = format!(
                    "file.{}.struct.{}", sanitize_module(module.id.as_str()), slug
                );
                files
                    .push(LayoutFile {
                        id: file_id.clone(),
                        path: format!("struct_{}.rs", slug),
                        use_block: Vec::new(),
                    });
                routing
                    .push(FileBinding {
                        node: LayoutNode::Struct(structure.id),
                        file_id,
                        rationale: "LAY-ATOM-001".to_owned(),
                    });
            }
            let mut enums: Vec<_> = semantic
                .enums
                .iter()
                .filter(|e| e.module == module.id)
                .cloned()
                .collect();
            enums.sort_by(|a, b| a.id.cmp(&b.id));
            for enumeration in enums {
                let slug = slugify(enumeration.name.as_str());
                let file_id = format!(
                    "file.{}.enum.{}", sanitize_module(module.id.as_str()), slug
                );
                files
                    .push(LayoutFile {
                        id: file_id.clone(),
                        path: format!("enum_{}.rs", slug),
                        use_block: Vec::new(),
                    });
                routing
                    .push(FileBinding {
                        node: LayoutNode::Enum(enumeration.id),
                        file_id,
                        rationale: "LAY-ATOM-001".to_owned(),
                    });
            }
            let mut traits: Vec<_> = semantic
                .traits
                .iter()
                .filter(|t| t.module == module.id)
                .cloned()
                .collect();
            traits.sort_by(|a, b| a.id.cmp(&b.id));
            for tr in traits {
                let slug = slugify(tr.name.as_str());
                let file_id = format!(
                    "file.{}.trait.{}", sanitize_module(module.id.as_str()), slug
                );
                files
                    .push(LayoutFile {
                        id: file_id.clone(),
                        path: format!("trait_{}.rs", slug),
                        use_block: Vec::new(),
                    });
                routing
                    .push(FileBinding {
                        node: LayoutNode::Trait(tr.id),
                        file_id,
                        rationale: "LAY-ATOM-001".to_owned(),
                    });
            }
            let mut impls: Vec<_> = semantic
                .impls
                .iter()
                .filter(|b| b.module == module.id)
                .cloned()
                .collect();
            impls.sort_by(|a, b| a.id.cmp(&b.id));
            for block in impls {
                let slug = slugify(block.id.as_str());
                let file_id = format!(
                    "file.{}.impl.{}", sanitize_module(module.id.as_str()), slug
                );
                files
                    .push(LayoutFile {
                        id: file_id.clone(),
                        path: format!("impl_{}.rs", slug),
                        use_block: Vec::new(),
                    });
                routing
                    .push(FileBinding {
                        node: LayoutNode::Impl(block.id),
                        file_id,
                        rationale: "LAY-ATOM-001".to_owned(),
                    });
            }
            let mut functions: Vec<_> = semantic
                .functions
                .iter()
                .filter(|f| f.module == module.id)
                .cloned()
                .collect();
            functions.sort_by(|a, b| a.id.cmp(&b.id));
            for function in functions {
                let slug = slugify(function.name.as_str());
                let file_id = format!(
                    "file.{}.fn.{}", sanitize_module(module.id.as_str()), slug
                );
                files
                    .push(LayoutFile {
                        id: file_id.clone(),
                        path: format!("fn_{}.rs", slug),
                        use_block: Vec::new(),
                    });
                routing
                    .push(FileBinding {
                        node: LayoutNode::Function(function.id),
                        file_id,
                        rationale: "LAY-ATOM-001".to_owned(),
                    });
            }
            files.sort_by(|a, b| a.id.cmp(&b.id));
            modules
                .push(LayoutModule {
                    id: module.id.clone(),
                    name: module.name.clone(),
                    files,
                    imports: Vec::new(),
                });
        }
        routing
            .sort_by(|a, b| {
                let (a_id, a_kind) = super::layout_node_key(&a.node);
                let (b_id, b_kind) = super::layout_node_key(&b.node);
                (a_kind, a_id).cmp(&(b_kind, b_id))
            });
        FileTopology { modules, routing }
    }
}
/// Canonical normalization strategy:
/// Deterministic, sorted, stable ordering of files and routing.
pub struct CanonicalNormalizationStrategy;
impl LayoutStrategy for CanonicalNormalizationStrategy {
    fn name(&self) -> &'static str {
        "canonical_normalized"
    }
    fn description(&self) -> &'static str {
        "Deterministic stable layout ordering for hashing and reproducibility."
    }
    fn plan(&self, semantic: &ParsedModel) -> FileTopology {
        let base = FlatNodeLayoutStrategy.plan(semantic);
        normalize_layout_graph(base)
    }
}
fn normalize_layout_graph(mut graph: FileTopology) -> FileTopology {
    graph.modules.sort_by(|a, b| a.id.cmp(&b.id));
    for module in &mut graph.modules {
        module.files.sort_by(|a, b| a.id.cmp(&b.id));
    }
    graph.routing.sort_by(|a, b| format!("{:?}", a.node).cmp(&format!("{:?}", b.node)));
    graph
}
