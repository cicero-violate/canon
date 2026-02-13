use std::collections::{HashMap, HashSet};

use crate::ir::{CanonicalIr, Module};
use thiserror::Error;

use super::{
    enforce_routing_completeness, layout_node_key, LayoutGraph, LayoutNode,
};

#[derive(Debug, Error)]
#[error("layout validation failed:\n{message}")]
pub struct LayoutValidationError {
    message: String,
}

impl LayoutValidationError {
    fn new(lines: Vec<String>) -> Self {
        let message = lines.join("\n");
        Self { message }
    }
}

pub fn validate_layout(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
) -> Result<(), LayoutValidationError> {
    let mut violations = Vec::new();

    validate_modules_and_files(ir, layout, &mut violations);

    let file_lookup = build_file_lookup(layout);
    let module_maps = build_module_maps(ir);
    let assigned_nodes =
        validate_routing(ir, layout, &file_lookup, &module_maps, &mut violations);

    enforce_routing_completeness(
        "struct",
        ir.structs.iter().map(|s| s.id.as_str()),
        &assigned_nodes,
        &mut violations,
    );
    enforce_routing_completeness(
        "enum",
        ir.enums.iter().map(|e| e.id.as_str()),
        &assigned_nodes,
        &mut violations,
    );
    enforce_routing_completeness(
        "trait",
        ir.traits.iter().map(|t| t.id.as_str()),
        &assigned_nodes,
        &mut violations,
    );
    enforce_routing_completeness(
        "impl",
        ir.impl_blocks.iter().map(|b| b.id.as_str()),
        &assigned_nodes,
        &mut violations,
    );
    enforce_routing_completeness(
        "function",
        ir.functions.iter().map(|f| f.id.as_str()),
        &assigned_nodes,
        &mut violations,
    );

    if violations.is_empty() {
        Ok(())
    } else {
        Err(LayoutValidationError::new(violations))
    }
}

fn validate_modules_and_files(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    violations: &mut Vec<String>,
) {
    let module_lookup: HashMap<&str, &Module> =
        ir.modules.iter().map(|m| (m.id.as_str(), m)).collect();
    let mut seen_layout_modules: HashSet<&str> = HashSet::new();
    let mut file_lookup: HashMap<&str, &str> = HashMap::new();

    for module in &layout.modules {
        if !seen_layout_modules.insert(module.id.as_str()) {
            violations.push(format!(
                "layout module `{}` appears more than once",
                module.id
            ));
        }
        if !module_lookup.contains_key(module.id.as_str()) {
            violations.push(format!(
                "layout module `{}` does not exist in Canon",
                module.id
            ));
        }
        for file in &module.files {
            if file_lookup
                .insert(file.id.as_str(), module.id.as_str())
                .is_some()
            {
                violations.push(format!(
                    "file `{}` is attached to multiple modules",
                    file.id
                ));
            }
            for entry in &file.use_block {
                if entry.trim().is_empty() {
                    violations.push(format!(
                        "file `{}` contains an empty use_block entry",
                        file.id
                    ));
                }
            }
        }
    }

    for module in &ir.modules {
        if !seen_layout_modules.contains(module.id.as_str()) {
            violations.push(format!("module `{}` missing from layout graph", module.id));
        }
    }
}

fn build_file_lookup(layout: &LayoutGraph) -> HashMap<&str, &str> {
    let mut file_lookup: HashMap<&str, &str> = HashMap::new();
    for module in &layout.modules {
        for file in &module.files {
            file_lookup.insert(file.id.as_str(), module.id.as_str());
        }
    }
    file_lookup
}

struct ModuleMaps<'a> {
    struct_modules: HashMap<&'a str, &'a str>,
    enum_modules: HashMap<&'a str, &'a str>,
    trait_modules: HashMap<&'a str, &'a str>,
    impl_modules: HashMap<&'a str, &'a str>,
    function_modules: HashMap<&'a str, &'a str>,
}

fn build_module_maps(ir: &CanonicalIr) -> ModuleMaps<'_> {
    ModuleMaps {
        struct_modules: ir
            .structs
            .iter()
            .map(|s| (s.id.as_str(), s.module.as_str()))
            .collect(),
        enum_modules: ir
            .enums
            .iter()
            .map(|e| (e.id.as_str(), e.module.as_str()))
            .collect(),
        trait_modules: ir
            .traits
            .iter()
            .map(|t| (t.id.as_str(), t.module.as_str()))
            .collect(),
        impl_modules: ir
            .impl_blocks
            .iter()
            .map(|i| (i.id.as_str(), i.module.as_str()))
            .collect(),
        function_modules: ir
            .functions
            .iter()
            .map(|f| (f.id.as_str(), f.module.as_str()))
            .collect(),
    }
}

fn validate_routing(
    ir: &CanonicalIr,
    layout: &LayoutGraph,
    file_lookup: &HashMap<&str, &str>,
    maps: &ModuleMaps<'_>,
    violations: &mut Vec<String>,
) -> HashSet<String> {
    let mut assigned_nodes: HashSet<String> = HashSet::new();

    for assignment in &layout.routing {
        let (node_id, node_label) = layout_node_key(&assignment.node);
        let key = format!("{node_label}:{node_id}");
        if !assigned_nodes.insert(key) {
            violations.push(format!(
                "node `{}` ({}) assigned more than once",
                node_id, node_label
            ));
        }

        let Some(module_for_file) = file_lookup.get(assignment.file_id.as_str()) else {
            violations.push(format!(
                "{} `{}` routed to unknown file `{}`",
                node_label, node_id, assignment.file_id
            ));
            continue;
        };

        let expected_module = match &assignment.node {
            LayoutNode::Struct(id) => maps.struct_modules.get(id.as_str()).copied(),
            LayoutNode::Enum(id) => maps.enum_modules.get(id.as_str()).copied(),
            LayoutNode::Trait(id) => maps.trait_modules.get(id.as_str()).copied(),
            LayoutNode::Impl(id) => maps.impl_modules.get(id.as_str()).copied(),
            LayoutNode::Function(id) => maps.function_modules.get(id.as_str()).copied(),
        };

        match expected_module {
            Some(module_id) => {
                if module_id != *module_for_file {
                    violations.push(format!(
                        "{} `{}` belongs to module `{}` but is routed to file in `{}`",
                        node_label, node_id, module_id, module_for_file
                    ));
                }
            }
            None => violations.push(format!(
                "{} `{}` referenced in layout does not exist in Canon",
                node_label, node_id
            )),
        }
    }

    assigned_nodes
}

