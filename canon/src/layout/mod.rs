//! Layout graph definitions (LAY-002).
//!
//! These types describe how semantic Canon IR nodes are split from their
//! filesystem layout. Future work will teach the ingestor/materializer to
//! consume these structures instead of mixing semantics + layout data.

use std::collections::{HashMap, HashSet};

use crate::ir::{
    CallEdge, CanonicalIr, EnumId, EnumNode, Function, FunctionId, ImplBlock, ImplId, Module,
    ModuleEdge, ModuleId, Struct, StructId, SystemGraph, TickGraph, Trait, TraitId, Word,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod strategies;
mod validation;

pub use validation::{LayoutValidationError, validate_layout};
pub use strategies::{OriginalLayoutStrategy, PerTypeLayoutStrategy, SingleFileLayoutStrategy, FlatNodeLayoutStrategy};

/// Container that pairs the semantic and layout graphs used during
/// ingestion/materialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutMap {
    pub semantic: SemanticGraph,
    pub layout: LayoutGraph,
}

impl Default for LayoutMap {
    fn default() -> Self {
        Self {
            semantic: SemanticGraph::default(),
            layout: LayoutGraph::default(),
        }
    }
}

/// A purely semantic projection of Canon IR.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticGraph {
    pub modules: Vec<Module>,
    pub structs: Vec<Struct>,
    pub enums: Vec<EnumNode>,
    pub traits: Vec<Trait>,
    pub impls: Vec<ImplBlock>,
    pub functions: Vec<Function>,
    pub module_edges: Vec<ModuleEdge>,
    pub call_edges: Vec<CallEdge>,
    pub tick_graphs: Vec<TickGraph>,
    pub system_graphs: Vec<SystemGraph>,
}

/// Filesystem layout for the semantic nodes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayoutGraph {
    pub modules: Vec<LayoutModule>,
    pub routing: Vec<LayoutAssignment>,
}

/// Modules describe their explicit file nodes and imports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutModule {
    pub id: ModuleId,
    pub name: Word,
    pub files: Vec<LayoutFile>,
    pub imports: Vec<LayoutImport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutFile {
    pub id: String,
    pub path: String,
    pub use_block: Vec<String>,
}

/// Maps a semantic node to the file that should render it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutAssignment {
    pub node: LayoutNode,
    pub file_id: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutImport {
    pub from: ModuleId,
    pub to: ModuleId,
    pub symbols: Vec<String>,
}

/// Identifiers that can be routed to a file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LayoutNode {
    Struct(StructId),
    Enum(EnumId),
    Trait(TraitId),
    Impl(ImplId),
    Function(FunctionId),
}

/// Strategies describe how to derive a `LayoutGraph` for a given semantic view.
pub trait LayoutStrategy {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str {
        ""
    }
    fn plan(&self, semantic: &SemanticGraph) -> LayoutGraph;
}

/// Represents helpers that can strip layout metadata from semantic nodes.
pub trait LayoutCleaner {
    fn drop_layout_fields(&self, map: LayoutMap) -> LayoutMap;
}

/// Apply a file topology (usually derived from DOT) to the given layout graph.
pub fn apply_topology_to_layout(
    layout: &mut LayoutGraph,
    modules: &[Module],
    topology: HashMap<String, Vec<LayoutFile>>,
) {
    let module_lookup: HashMap<&str, &Module> =
        modules.iter().map(|m| (m.id.as_str(), m)).collect();
    for (module_id, files) in topology {
        if let Some(existing) = layout.modules.iter_mut().find(|m| m.id == module_id) {
            existing.files = files;
            continue;
        }
        let Some(ir_module) = module_lookup.get(module_id.as_str()) else {
            continue;
        };
        layout.modules.push(LayoutModule {
            id: module_id,
            name: ir_module.name.clone(),
            files,
            imports: Vec::new(),
        });
    }
}

fn layout_node_key(node: &LayoutNode) -> (&str, &'static str) {
    match node {
        LayoutNode::Struct(id) => (id.as_str(), "struct"),
        LayoutNode::Enum(id) => (id.as_str(), "enum"),
        LayoutNode::Trait(id) => (id.as_str(), "trait"),
        LayoutNode::Impl(id) => (id.as_str(), "impl"),
        LayoutNode::Function(id) => (id.as_str(), "function"),
    }
}

fn enforce_routing_completeness<'a>(
    label: &'static str,
    ids: impl Iterator<Item = &'a str>,
    assigned: &HashSet<String>,
    violations: &mut Vec<String>,
) {
    for id in ids {
        let key = format!("{label}:{id}");
        if !assigned.contains(&key) {
            violations.push(format!("{} `{}` is missing from layout routing", label, id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        CanonicalIr, CanonicalMeta, Function, FunctionContract, FunctionMetadata, Language, Module,
        Project, Receiver, Struct, StructKind, VersionContract, Visibility, Word,
    };
    use std::collections::HashMap;

    #[test]
    fn validate_layout_accepts_consistent_graph() {
        let ir = sample_ir();
        let layout = sample_layout(&ir);
        validate_layout(&ir, &layout).expect("layout should be valid");
    }

    #[test]
    fn validate_layout_rejects_unknown_file() {
        let ir = sample_ir();
        let mut layout = sample_layout(&ir);
        layout.routing[0].file_id = "file.missing".to_owned();
        let err = validate_layout(&ir, &layout).expect_err("missing file should fail");
        assert!(
            err.to_string().contains("unknown file"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_layout_rejects_duplicate_nodes() {
        let ir = sample_ir();
        let mut layout = sample_layout(&ir);
        layout.routing.push(layout.routing[0].clone());
        let err =
            validate_layout(&ir, &layout).expect_err("duplicate node assignments should fail");
        assert!(
            err.to_string().contains("assigned more than once"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_layout_rejects_missing_assignment() {
        let mut ir = sample_ir();
        ir.functions.push(sample_function(
            "function.core.do_it",
            ir.modules[0].id.as_str(),
        ));
        let layout = sample_layout(&ir);
        let err =
            validate_layout(&ir, &layout).expect_err("missing function assignment should fail");
        assert!(
            err.to_string()
                .contains("function `function.core.do_it` is missing"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn validate_layout_rejects_empty_use_block_entry() {
        let ir = sample_ir();
        let mut layout = sample_layout(&ir);
        layout.modules[0].files[0].use_block.push(" ".to_owned());
        let err = validate_layout(&ir, &layout).expect_err("empty use block entry should fail");
        assert!(
            err.to_string().contains("empty use_block entry"),
            "unexpected error: {err}"
        );
    }

    fn sample_ir() -> CanonicalIr {
        let module = Module {
            id: "module.core".to_owned(),
            name: Word::new("core").unwrap(),
            visibility: Visibility::Public,
            description: String::new(),
            pub_uses: Vec::new(),
            constants: Vec::new(),
            type_aliases: Vec::new(),
            statics: Vec::new(),
            attributes: Vec::new(),
        };
        let structure = Struct {
            id: "struct.core.Widget".to_owned(),
            name: Word::new("Widget").unwrap(),
            module: module.id.clone(),
            visibility: Visibility::Public,
            derives: Vec::new(),
            doc: None,
            kind: StructKind::Normal,
            fields: Vec::new(),
            history: Vec::new(),
        };
        CanonicalIr {
            meta: CanonicalMeta {
                version: "0.0.0".to_owned(),
                law_revision: Word::new("law").unwrap(),
                description: String::new(),
            },
            version_contract: VersionContract {
                current: "0.0.0".to_owned(),
                compatible_with: Vec::new(),
                migration_proofs: Vec::new(),
            },
            project: Project {
                name: Word::new("canon").unwrap(),
                version: "0.0.0".to_owned(),
                language: Language::Rust,
            },
            modules: vec![module],
            module_edges: Vec::new(),
            structs: vec![structure],
            enums: Vec::new(),
            traits: Vec::new(),
            impl_blocks: Vec::new(),
            functions: Vec::new(),
            call_edges: Vec::new(),
            tick_graphs: Vec::new(),
            system_graphs: Vec::new(),
            loop_policies: Vec::new(),
            ticks: Vec::new(),
            tick_epochs: Vec::new(),
            plans: Vec::new(),
            executions: Vec::new(),
            admissions: Vec::new(),
            applied_deltas: Vec::new(),
            gpu_functions: Vec::new(),
            proposals: Vec::new(),
            judgments: Vec::new(),
            judgment_predicates: Vec::new(),
            deltas: Vec::new(),
            proofs: Vec::new(),
            learning: Vec::new(),
            errors: Vec::new(),
            dependencies: Vec::new(),
            file_hashes: HashMap::new(),

            // --- Added for planning integration ---
            reward_deltas: Vec::new(),
            world_model: Default::default(),
            policy_parameters: Vec::new(),
            goal_mutations: Vec::new(),
        }
    }

    fn sample_layout(ir: &CanonicalIr) -> LayoutGraph {
        LayoutGraph {
            modules: vec![LayoutModule {
                id: ir.modules[0].id.clone(),
                name: ir.modules[0].name.clone(),
                files: vec![LayoutFile {
                    id: "file.core.mod".to_owned(),
                    path: "mod.rs".to_owned(),
                    use_block: Vec::new(),
                }],
                imports: Vec::new(),
            }],
            routing: vec![LayoutAssignment {
                node: LayoutNode::Struct(ir.structs[0].id.clone()),
                file_id: "file.core.mod".to_owned(),
                rationale: "test".to_owned(),
            }],
        }
    }

    fn sample_function(id: &str, module: &str) -> Function {
        Function {
            id: id.to_owned(),
            name: Word::new("generated").unwrap(),
            module: module.to_owned(),
            impl_id: "impl.core.Widget".to_owned(),
            trait_function: "trait_fn.core.Widget".to_owned(),
            visibility: Visibility::Public,
            doc: None,
            lifetime_params: Vec::new(),
            receiver: Receiver::None,
            is_async: false,
            is_unsafe: false,
            generics: Vec::new(),
            where_clauses: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            deltas: Vec::new(),
            contract: FunctionContract {
                total: true,
                deterministic: true,
                explicit_inputs: true,
                explicit_outputs: true,
                effects_are_deltas: true,
            },
            metadata: FunctionMetadata::default(),
        }
    }
}
