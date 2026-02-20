use crate::ir::SystemState;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestEntry {
    pub id: String,
    pub slot: u64,
}
impl ManifestEntry {
    fn new(id: String, slot: u64) -> Self {
        Self { id, slot }
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub modules: Vec<ManifestEntry>,
    pub structs: Vec<ManifestEntry>,
    pub enums: Vec<ManifestEntry>,
    pub traits: Vec<ManifestEntry>,
    pub impls: Vec<ManifestEntry>,
    pub functions: Vec<ManifestEntry>,
    pub module_edges: Vec<ManifestEntry>,
    pub call_edges: Vec<ManifestEntry>,
    pub ticks: Vec<ManifestEntry>,
    pub tick_graphs: Vec<ManifestEntry>,
    pub system_graphs: Vec<ManifestEntry>,
    pub loop_policies: Vec<ManifestEntry>,
    pub tick_epochs: Vec<ManifestEntry>,
    pub policies: Vec<ManifestEntry>,
    pub plans: Vec<ManifestEntry>,
    pub executions: Vec<ManifestEntry>,
    pub admissions: Vec<ManifestEntry>,
    pub applied_deltas: Vec<ManifestEntry>,
    pub gpu_functions: Vec<ManifestEntry>,
    pub proposals: Vec<ManifestEntry>,
    pub judgments: Vec<ManifestEntry>,
    pub judgment_predicates: Vec<ManifestEntry>,
    pub delta_defs: Vec<ManifestEntry>,
    pub proofs: Vec<ManifestEntry>,
    pub learnings: Vec<ManifestEntry>,
    pub errors: Vec<ManifestEntry>,
    pub dependencies: Vec<ManifestEntry>,
    pub file_hashes: Vec<ManifestEntry>,
    pub rewards: Vec<ManifestEntry>,
    pub goal_mutations: Vec<ManifestEntry>,
}
impl ArtifactManifest {
    pub fn from_ir(ir: &SystemState) -> Self {
        Self {
            modules: build_manifest_entries(ir.modules.iter().map(|m| m.id.clone())),
            structs: build_manifest_entries(ir.structs.iter().map(|s| s.id.clone())),
            enums: build_manifest_entries(ir.enums.iter().map(|e| e.id.clone())),
            traits: build_manifest_entries(ir.traits.iter().map(|t| t.id.clone())),
            impls: build_manifest_entries(ir.impls.iter().map(|i| i.id.clone())),
            functions: build_manifest_entries(ir.functions.iter().map(|f| f.id.clone())),
            module_edges: build_manifest_entries(
                ir
                    .module_edges
                    .iter()
                    .map(|edge| format!("edge.module.{}->{}", edge.source, edge.target)),
            ),
            call_edges: build_manifest_entries(
                ir
                    .call_edges
                    .iter()
                    .map(|edge| format!("edge.call.{}->{}", edge.caller, edge.callee)),
            ),
            ticks: build_manifest_entries(ir.ticks.iter().map(|t| t.id.clone())),
            tick_graphs: build_manifest_entries(
                ir.tick_graphs.iter().map(|g| g.id.clone()),
            ),
            system_graphs: build_manifest_entries(
                ir.system_graphs.iter().map(|g| g.id.clone()),
            ),
            loop_policies: build_manifest_entries(
                ir.loop_policies.iter().map(|p| p.id.clone()),
            ),
            tick_epochs: build_manifest_entries(
                ir.tick_epochs.iter().map(|e| e.id.clone()),
            ),
            policies: build_manifest_entries(
                ir.policy_parameters.iter().map(|p| p.id.clone()),
            ),
            plans: build_manifest_entries(ir.plans.iter().map(|p| p.id.clone())),
            executions: build_manifest_entries(
                ir.executions.iter().map(|e| e.id.clone()),
            ),
            admissions: build_manifest_entries(
                ir.admissions.iter().map(|a| a.id.clone()),
            ),
            applied_deltas: build_manifest_entries(
                ir.applied_deltas.iter().map(|d| d.id.clone()),
            ),
            gpu_functions: build_manifest_entries(
                ir.gpu_functions.iter().map(|g| g.id.clone()),
            ),
            proposals: build_manifest_entries(ir.proposals.iter().map(|p| p.id.clone())),
            judgments: build_manifest_entries(ir.judgments.iter().map(|j| j.id.clone())),
            judgment_predicates: build_manifest_entries(
                ir.judgment_predicates.iter().map(|p| p.id.clone()),
            ),
            delta_defs: build_manifest_entries(ir.deltas.iter().map(|d| d.id.clone())),
            proofs: build_manifest_entries(ir.proofs.iter().map(|p| p.id.clone())),
            learnings: build_manifest_entries(ir.learning.iter().map(|l| l.id.clone())),
            errors: build_manifest_entries(ir.errors.iter().map(|e| e.id.clone())),
            dependencies: build_manifest_entries(
                ir.dependencies.iter().map(|d| format!("dependency::{}", d.name)),
            ),
            file_hashes: build_manifest_entries(
                ir.file_hashes.keys().cloned().map(|path| format!("filehash::{path}")),
            ),
            rewards: build_manifest_entries(
                ir.reward_deltas.iter().map(|r| r.id.clone()),
            ),
            goal_mutations: build_manifest_entries(
                ir.goal_mutations.iter().map(|g| g.id.clone()),
            ),
        }
    }
}
fn build_manifest_entries<I>(ids: I) -> Vec<ManifestEntry>
where
    I: IntoIterator<Item = String>,
{
    ids.into_iter()
        .enumerate()
        .map(|(slot, id)| ManifestEntry::new(id, slot as u64))
        .collect()
}
