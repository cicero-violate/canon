use std::collections::HashMap;

use super::{
    LayoutAssignment, LayoutFile, LayoutGraph, LayoutModule, LayoutNode, LayoutStrategy,
    SemanticGraph,
};
/// Preserves the original layout graph exactly as provided.
pub struct OriginalLayoutStrategy {
    base: LayoutGraph,
}

impl OriginalLayoutStrategy {
    pub fn new(base: LayoutGraph) -> Self {
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

    fn plan(&self, _semantic: &SemanticGraph) -> LayoutGraph {
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

    fn plan(&self, semantic: &SemanticGraph) -> LayoutGraph {
        let mut modules = Vec::new();
        let mut routing = Vec::new();

        for module in &semantic.modules {
            let file_id = default_file_id(module.id.as_str());
            modules.push(LayoutModule {
                id: module.id.clone(),
                name: module.name.clone(),
                files: vec![LayoutFile {
                    id: file_id.clone(),
                    path: "mod.rs".to_owned(),
                    use_block: Vec::new(),
                }],
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

        LayoutGraph { modules, routing }
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

    fn plan(&self, semantic: &SemanticGraph) -> LayoutGraph {
        let mut modules = Vec::new();
        let mut routing = Vec::new();

        // Map impl id -> struct id for quick lookup.
        let mut impl_to_struct: HashMap<&str, &str> = HashMap::new();
        for block in &semantic.impls {
            impl_to_struct.insert(block.id.as_str(), block.struct_id.as_str());
        }

        for module in &semantic.modules {
            let mut files = vec![LayoutFile {
                id: default_file_id(module.id.as_str()),
                path: "mod.rs".to_owned(),
                use_block: Vec::new(),
            }];
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
                routing.push(LayoutAssignment {
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
                routing.push(LayoutAssignment {
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
                routing.push(LayoutAssignment {
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
                routing.push(LayoutAssignment {
                    node: LayoutNode::Impl(block.id.clone()),
                    file_id,
                    rationale: "LAY-005: per-type strategy".to_owned(),
                });
            }

            for function in semantic.functions.iter().filter(|f| f.module == module.id) {
                let file_id = if let Some(struct_id) = impl_to_struct.get(function.impl_id.as_str())
                {
                    file_for_struct
                        .get(struct_id)
                        .cloned()
                        .unwrap_or_else(|| default_file_id(module.id.as_str()))
                } else {
                    default_file_id(module.id.as_str())
                };
                routing.push(LayoutAssignment {
                    node: LayoutNode::Function(function.id.clone()),
                    file_id,
                    rationale: "LAY-005: per-type strategy".to_owned(),
                });
            }

            modules.push(LayoutModule {
                id: module.id.clone(),
                name: module.name.clone(),
                files,
                imports: Vec::new(),
            });
        }

        LayoutGraph { modules, routing }
    }
}

fn route_all_nodes(
    semantic: &SemanticGraph,
    module_id: &str,
    file_id: &str,
    rationale: &str,
    routing: &mut Vec<LayoutAssignment>,
) {
    for structure in semantic
        .structs
        .iter()
        .filter(|s| s.module.as_str() == module_id)
    {
        routing.push(LayoutAssignment {
            node: LayoutNode::Struct(structure.id.clone()),
            file_id: file_id.to_owned(),
            rationale: rationale.to_owned(),
        });
    }
    for enumeration in semantic
        .enums
        .iter()
        .filter(|e| e.module.as_str() == module_id)
    {
        routing.push(LayoutAssignment {
            node: LayoutNode::Enum(enumeration.id.clone()),
            file_id: file_id.to_owned(),
            rationale: rationale.to_owned(),
        });
    }
    for tr in semantic
        .traits
        .iter()
        .filter(|t| t.module.as_str() == module_id)
    {
        routing.push(LayoutAssignment {
            node: LayoutNode::Trait(tr.id.clone()),
            file_id: file_id.to_owned(),
            rationale: rationale.to_owned(),
        });
    }
    for block in semantic
        .impls
        .iter()
        .filter(|b| b.module.as_str() == module_id)
    {
        routing.push(LayoutAssignment {
            node: LayoutNode::Impl(block.id.clone()),
            file_id: file_id.to_owned(),
            rationale: rationale.to_owned(),
        });
    }
    for function in semantic
        .functions
        .iter()
        .filter(|f| f.module.as_str() == module_id)
    {
        routing.push(LayoutAssignment {
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
    files.push(LayoutFile {
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
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn slugify(value: &str) -> String {
    value
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
