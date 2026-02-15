use crate::ir::{CanonicalIr, CanonicalMeta, Language, Project, VersionContract, Word};
use crate::layout::SemanticGraph;

/// Builds a `CanonicalIr` from a `SemanticGraph` by populating the metadata
/// Canon still requires (versioning/project info) while leaving layout data to
/// the caller.
pub struct SemanticIrBuilder {
    meta: CanonicalMeta,
    project: Project,
}

impl SemanticIrBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            meta: CanonicalMeta {
                version: env!("CARGO_PKG_VERSION").to_string(),
                law_revision: Word::new("SemanticOnly").unwrap(),
                description: format!("Semantic graph for `{name}`"),
            },
            project: Project {
                name: Word::new(name).unwrap(),
                version: "0.0.0".to_owned(),
                language: Language::Rust,
            },
        }
    }

    pub fn build(&self, semantic: SemanticGraph) -> CanonicalIr {
        CanonicalIr {
            meta: self.meta.clone(),
            version_contract: VersionContract {
                current: self.meta.version.clone(),
                compatible_with: vec![],
                migration_proofs: vec![],
            },
            project: self.project.clone(),
            modules: semantic.modules,
            module_edges: semantic.module_edges,
            structs: semantic.structs,
            enums: semantic.enums,
            traits: semantic.traits,
            impls: semantic.impls,
            functions: semantic.functions,
            call_edges: semantic.call_edges,
            tick_graphs: semantic.tick_graphs,
            system_graphs: semantic.system_graphs,
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
            file_hashes: std::collections::HashMap::new(),
            reward_deltas: Vec::new(),
            world_model: crate::ir::world_model::WorldModel::new(),
            policy_parameters: Vec::new(),
            goal_mutations: Vec::new(),
        }
    }
}
