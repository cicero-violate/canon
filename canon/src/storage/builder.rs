use bincode::{Options, config::DefaultOptions};
use memory_engine::delta::Delta;
use memory_engine::delta::delta_types::Source;
use memory_engine::epoch::Epoch;
use memory_engine::primitives::{DeltaID, Hash as EngineHash, PageID};
use memory_engine::{
    AdmissionError, AdmissionProof, CommitProof, JudgmentProof, MemoryEngine,
};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;
use crate::ir::{
    AppliedDeltaRecord, CallEdge, CanonicalIr, Delta as CanonicalDelta, DeltaAdmission,
    EnumNode, ErrorArtifact, ExecutionRecord, ExternalDependency, Function, GoalMutation,
    GpuFunction, ImplBlock, Judgment, JudgmentPredicate, Learning, LoopPolicy, Module,
    ModuleEdge, Plan, PolicyParameters, Proof, Proposal, RewardRecord, Struct,
    SystemGraph, Tick, TickEpoch, TickGraph, Trait,
};
use crate::storage::layout::{
    ArtifactSegment, LayoutError, META_MANIFEST_SLOT, META_STATE_SLOT,
    MemoryArtifactLayout, PAGE_BYTES, SINGLETON_SLOT,
};
use crate::storage::manifest::{ArtifactManifest, ManifestEntry};
use crate::storage::types::FileHashRecord;
const LENGTH_PREFIX: usize = 8;
/// Unified builder that persists CanonicalIr artifacts through MemoryEngine.
pub struct MemoryIrBuilder<'engine> {
    engine: &'engine MemoryEngine,
    layout: MemoryArtifactLayout,
    judgment_counter: AtomicU64,
}
impl<'engine> MemoryIrBuilder<'engine> {
    pub fn new(engine: &'engine MemoryEngine) -> Self {
        Self {
            engine,
            layout: MemoryArtifactLayout::new(),
            judgment_counter: AtomicU64::new(0),
        }
    }
    /// Persist every artifact in the provided IR snapshot.
    pub fn write_ir_to_disk(&self, ir: &CanonicalIr) -> Result<(), MemoryIrBuilderError> {
        let manifest = ArtifactManifest::from_ir(ir);
        let slot_lookup = ManifestSlotLookup::new(&manifest);
        self.write_artifact_page(
            ArtifactSegment::Meta,
            "manifest",
            META_MANIFEST_SLOT,
            &manifest,
        )?;
        self.write_artifact_page(
            ArtifactSegment::Meta,
            "meta",
            META_STATE_SLOT,
            &ir.meta,
        )?;
        self.write_artifact_page(
            ArtifactSegment::VersionContract,
            "version_contract",
            SINGLETON_SLOT,
            &ir.version_contract,
        )?;
        self.write_artifact_page(
            ArtifactSegment::Project,
            "project",
            SINGLETON_SLOT,
            &ir.project,
        )?;
        self.write_artifact_page(
            ArtifactSegment::WorldModel,
            "world_model",
            SINGLETON_SLOT,
            &ir.world_model,
        )?;
        for module in &ir.modules {
            let slot = slot_lookup.require_slot(ArtifactSegment::Module, &module.id)?;
            self.persist_module(module, slot)?;
        }
        for structure in &ir.structs {
            let slot = slot_lookup.require_slot(ArtifactSegment::Struct, &structure.id)?;
            self.persist_struct(structure, slot)?;
        }
        for enum_node in &ir.enums {
            let slot = slot_lookup.require_slot(ArtifactSegment::Enum, &enum_node.id)?;
            self.persist_enum(enum_node, slot)?;
        }
        for tr in &ir.traits {
            let slot = slot_lookup.require_slot(ArtifactSegment::Trait, &tr.id)?;
            self.persist_trait(tr, slot)?;
        }
        for block in &ir.impls {
            let slot = slot_lookup.require_slot(ArtifactSegment::ImplBlock, &block.id)?;
            self.persist_impl(block, slot)?;
        }
        for function in &ir.functions {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::Function, &function.id)?;
            self.persist_function(function, slot)?;
        }
        for gpu in &ir.gpu_functions {
            let slot = slot_lookup.require_slot(ArtifactSegment::GpuFunction, &gpu.id)?;
            self.write_artifact_page(ArtifactSegment::GpuFunction, &gpu.id, slot, gpu)?;
        }
        for edge in &ir.module_edges {
            let id = format!("edge.module.{}->{}", edge.source, edge.target);
            let slot = slot_lookup.require_slot(ArtifactSegment::ModuleEdge, &id)?;
            self.persist_module_edge(edge, slot)?;
        }
        for edge in &ir.call_edges {
            let id = format!("edge.call.{}->{}", edge.caller, edge.callee);
            let slot = slot_lookup.require_slot(ArtifactSegment::CallEdge, &id)?;
            self.persist_call_edge(edge, slot)?;
        }
        for loop_policy in &ir.loop_policies {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::LoopPolicy, &loop_policy.id)?;
            self.write_artifact_page(
                ArtifactSegment::LoopPolicy,
                &loop_policy.id,
                slot,
                loop_policy,
            )?;
        }
        for tick in &ir.ticks {
            let slot = slot_lookup.require_slot(ArtifactSegment::Tick, &tick.id)?;
            self.write_artifact_page(ArtifactSegment::Tick, &tick.id, slot, tick)?;
        }
        for tick_graph in &ir.tick_graphs {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::TickGraph, &tick_graph.id)?;
            self.persist_tick_graph(&tick_graph.id, slot, tick_graph)?;
        }
        for system in &ir.system_graphs {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::SystemGraph, &system.id)?;
            self.persist_system_graph(&system.id, slot, system)?;
        }
        for epoch in &ir.tick_epochs {
            let slot = slot_lookup.require_slot(ArtifactSegment::TickEpoch, &epoch.id)?;
            self.write_artifact_page(
                ArtifactSegment::TickEpoch,
                &epoch.id,
                slot,
                epoch,
            )?;
        }
        for policy in &ir.policy_parameters {
            let slot = slot_lookup.require_slot(ArtifactSegment::Policy, &policy.id)?;
            self.write_artifact_page(ArtifactSegment::Policy, &policy.id, slot, policy)?;
        }
        for plan in &ir.plans {
            let slot = slot_lookup.require_slot(ArtifactSegment::Plan, &plan.id)?;
            self.write_artifact_page(ArtifactSegment::Plan, &plan.id, slot, plan)?;
        }
        for execution in &ir.executions {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::Execution, &execution.id)?;
            self.write_artifact_page(
                ArtifactSegment::Execution,
                &execution.id,
                slot,
                execution,
            )?;
        }
        for admission in &ir.admissions {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::Admission, &admission.id)?;
            self.write_artifact_page(
                ArtifactSegment::Admission,
                &admission.id,
                slot,
                admission,
            )?;
        }
        for applied in &ir.applied_deltas {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::AppliedDelta, &applied.id)?;
            self.write_artifact_page(
                ArtifactSegment::AppliedDelta,
                &applied.id,
                slot,
                applied,
            )?;
        }
        for proposal in &ir.proposals {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::Proposal, &proposal.id)?;
            self.write_artifact_page(
                ArtifactSegment::Proposal,
                &proposal.id,
                slot,
                proposal,
            )?;
        }
        for judgment in &ir.judgments {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::Judgment, &judgment.id)?;
            self.write_artifact_page(
                ArtifactSegment::Judgment,
                &judgment.id,
                slot,
                judgment,
            )?;
        }
        for predicate in &ir.judgment_predicates {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::JudgmentPredicate, &predicate.id)?;
            self.write_artifact_page(
                ArtifactSegment::JudgmentPredicate,
                &predicate.id,
                slot,
                predicate,
            )?;
        }
        for delta in &ir.deltas {
            let slot = slot_lookup.require_slot(ArtifactSegment::DeltaDef, &delta.id)?;
            self.write_artifact_page(ArtifactSegment::DeltaDef, &delta.id, slot, delta)?;
        }
        for proof in &ir.proofs {
            let slot = slot_lookup.require_slot(ArtifactSegment::Proof, &proof.id)?;
            self.write_artifact_page(ArtifactSegment::Proof, &proof.id, slot, proof)?;
        }
        for learn in &ir.learning {
            let slot = slot_lookup.require_slot(ArtifactSegment::Learning, &learn.id)?;
            self.write_artifact_page(ArtifactSegment::Learning, &learn.id, slot, learn)?;
        }
        for err in &ir.errors {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::ErrorArtifact, &err.id)?;
            self.write_artifact_page(
                ArtifactSegment::ErrorArtifact,
                &err.id,
                slot,
                err,
            )?;
        }
        for dep in &ir.dependencies {
            let key = format!("dependency::{}", dep.name);
            let slot = slot_lookup.require_slot(ArtifactSegment::Dependency, &key)?;
            self.write_artifact_page(ArtifactSegment::Dependency, &key, slot, dep)?;
        }
        for (path, hash) in &ir.file_hashes {
            let key = format!("filehash::{path}");
            let slot = slot_lookup.require_slot(ArtifactSegment::FileHash, &key)?;
            let record = FileHashRecord {
                path: path.clone(),
                hash: hash.clone(),
            };
            self.write_artifact_page(ArtifactSegment::FileHash, &key, slot, &record)?;
        }
        for reward in &ir.reward_deltas {
            let slot = slot_lookup.require_slot(ArtifactSegment::Reward, &reward.id)?;
            self.persist_reward(&reward.id, slot, reward)?;
        }
        for goal in &ir.goal_mutations {
            let slot = slot_lookup
                .require_slot(ArtifactSegment::GoalMutation, &goal.id)?;
            self.write_artifact_page(
                ArtifactSegment::GoalMutation,
                &goal.id,
                slot,
                goal,
            )?;
        }
        Ok(())
    }
    pub fn persist_module(
        &self,
        module: &Module,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Module, &module.id, slot, module)
    }
    pub fn persist_struct(
        &self,
        structure: &Struct,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Struct, &structure.id, slot, structure)
    }
    pub fn persist_enum(
        &self,
        enum_node: &EnumNode,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Enum, &enum_node.id, slot, enum_node)
    }
    pub fn persist_trait(
        &self,
        tr: &Trait,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Trait, &tr.id, slot, tr)
    }
    pub fn persist_impl(
        &self,
        block: &ImplBlock,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::ImplBlock, &block.id, slot, block)
    }
    pub fn persist_function(
        &self,
        function: &Function,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Function, &function.id, slot, function)
    }
    pub fn persist_module_edge(
        &self,
        edge: &ModuleEdge,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        let id = format!("edge.module.{}->{}", edge.source, edge.target);
        self.write_artifact_page(ArtifactSegment::ModuleEdge, &id, slot, edge)
    }
    pub fn persist_call_edge(
        &self,
        edge: &CallEdge,
        slot: u64,
    ) -> Result<(), MemoryIrBuilderError> {
        let id = format!("edge.call.{}->{}", edge.caller, edge.callee);
        self.write_artifact_page(ArtifactSegment::CallEdge, &id, slot, edge)
    }
    pub fn persist_tick_graph(
        &self,
        tick_id: &str,
        slot: u64,
        graph: &TickGraph,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::TickGraph, tick_id, slot, graph)
    }
    pub fn persist_system_graph(
        &self,
        system_id: &str,
        slot: u64,
        graph: &SystemGraph,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::SystemGraph, system_id, slot, graph)
    }
    pub fn persist_reward<T: Serialize>(
        &self,
        key: &str,
        slot: u64,
        value: &T,
    ) -> Result<(), MemoryIrBuilderError> {
        self.write_artifact_page(ArtifactSegment::Reward, key, slot, value)
    }
    fn write_artifact_page<T: Serialize>(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        slot: u64,
        value: &T,
    ) -> Result<(), MemoryIrBuilderError> {
        let bytes = self.encode(value)?;
        let mut payload = Vec::with_capacity(LENGTH_PREFIX + bytes.len());
        payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        payload.extend_from_slice(&bytes);
        let total_len = payload.len();
        let chunk_count = ((total_len + PAGE_BYTES - 1) / PAGE_BYTES) as u64;
        if chunk_count > crate::storage::layout::PAGES_PER_SLOT {
            return Err(
                MemoryIrBuilderError::Layout(LayoutError::ChunkOverflow {
                    segment,
                    artifact_id: artifact_id.to_owned(),
                    chunk_index: chunk_count - 1,
                    limit: crate::storage::layout::PAGES_PER_SLOT,
                }),
            );
        }
        payload.resize(chunk_count as usize * PAGE_BYTES, 0);
        let mut deltas = Vec::new();
        for chunk_index in 0..chunk_count {
            let start = chunk_index as usize * PAGE_BYTES;
            let chunk = &payload[start..start + PAGE_BYTES];
            let mask = vec![true; PAGE_BYTES];
            let address = self.layout.page_for(segment, artifact_id, slot, chunk_index)?;
            let delta = Delta::new_dense(
                    DeltaID(address.page_id),
                    PageID(address.page_id),
                    Epoch(0),
                    chunk.to_vec(),
                    mask,
                    Source(format!("canon.artifact::{:?}::{artifact_id}", segment)),
                )
                .map_err(|err| MemoryIrBuilderError::Delta {
                    artifact_id: artifact_id.to_owned(),
                    source: err,
                })?;
            deltas.push(delta);
        }
        self.admit_and_commit_pages(segment, artifact_id, deltas)
    }
    fn encode<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, MemoryIrBuilderError> {
        DefaultOptions::new()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .serialize(value)
            .map_err(MemoryIrBuilderError::Serialize)
    }
    fn admit_and_commit_pages(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        deltas: Vec<Delta>,
    ) -> Result<(), MemoryIrBuilderError> {
        let mut hashes = Vec::with_capacity(deltas.len());
        for delta in deltas {
            hashes.push(self.engine.register_delta(delta));
        }
        let proof = self.build_builder_judgment_proof(segment, artifact_id, &hashes);
        let admission = self.engine.admit_execution(&proof)?;
        self.engine
            .commit_batch(&admission, &hashes)
            .map_err(MemoryIrBuilderError::Memory)?;
        Ok(())
    }
    fn build_builder_judgment_proof(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        hashes: &[EngineHash],
    ) -> JudgmentProof {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&(segment.ordinal()).to_le_bytes());
        hasher.update(artifact_id.as_bytes());
        for hash in hashes {
            hasher.update(hash);
        }
        let digest = hasher.finalize();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&digest.as_bytes()[..32]);
        let timestamp = self.judgment_counter.fetch_add(1, Ordering::SeqCst);
        JudgmentProof {
            approved: true,
            timestamp,
            hash: bytes,
        }
    }
}
#[derive(Debug, Error)]
pub enum MemoryIrBuilderError {
    #[error("serialization failed: {0}")]
    Serialize(bincode::Error),
    #[error("layout error: {0}")]
    Layout(#[from] LayoutError),
    #[error("manifest missing slot for `{artifact_id}` in {segment:?}")]
    MissingSlot { segment: ArtifactSegment, artifact_id: String },
    #[error("delta encode failed for `{artifact_id}`: {source}")]
    Delta { artifact_id: String, #[source] source: memory_engine::delta::DeltaError },
    #[error("memory engine error: {0}")]
    Memory(#[from] memory_engine::MemoryEngineError),
    #[error("admission failed: {0}")]
    Admission(#[from] AdmissionError),
}
struct ManifestSlotLookup {
    slots: HashMap<ArtifactSegment, HashMap<String, u64>>,
}
impl ManifestSlotLookup {
    fn new(manifest: &ArtifactManifest) -> Self {
        let mut slots: HashMap<ArtifactSegment, HashMap<String, u64>> = HashMap::new();
        Self::index_segment(&mut slots, ArtifactSegment::Module, &manifest.modules);
        Self::index_segment(&mut slots, ArtifactSegment::Struct, &manifest.structs);
        Self::index_segment(&mut slots, ArtifactSegment::Enum, &manifest.enums);
        Self::index_segment(&mut slots, ArtifactSegment::Trait, &manifest.traits);
        Self::index_segment(&mut slots, ArtifactSegment::ImplBlock, &manifest.impls);
        Self::index_segment(&mut slots, ArtifactSegment::Function, &manifest.functions);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::ModuleEdge,
            &manifest.module_edges,
        );
        Self::index_segment(&mut slots, ArtifactSegment::CallEdge, &manifest.call_edges);
        Self::index_segment(&mut slots, ArtifactSegment::Tick, &manifest.ticks);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::TickGraph,
            &manifest.tick_graphs,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::SystemGraph,
            &manifest.system_graphs,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::LoopPolicy,
            &manifest.loop_policies,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::TickEpoch,
            &manifest.tick_epochs,
        );
        Self::index_segment(&mut slots, ArtifactSegment::Policy, &manifest.policies);
        Self::index_segment(&mut slots, ArtifactSegment::Plan, &manifest.plans);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::Execution,
            &manifest.executions,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::Admission,
            &manifest.admissions,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::AppliedDelta,
            &manifest.applied_deltas,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::GpuFunction,
            &manifest.gpu_functions,
        );
        Self::index_segment(&mut slots, ArtifactSegment::Proposal, &manifest.proposals);
        Self::index_segment(&mut slots, ArtifactSegment::Judgment, &manifest.judgments);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::JudgmentPredicate,
            &manifest.judgment_predicates,
        );
        Self::index_segment(&mut slots, ArtifactSegment::DeltaDef, &manifest.delta_defs);
        Self::index_segment(&mut slots, ArtifactSegment::Proof, &manifest.proofs);
        Self::index_segment(&mut slots, ArtifactSegment::Learning, &manifest.learnings);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::ErrorArtifact,
            &manifest.errors,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::Dependency,
            &manifest.dependencies,
        );
        Self::index_segment(
            &mut slots,
            ArtifactSegment::FileHash,
            &manifest.file_hashes,
        );
        Self::index_segment(&mut slots, ArtifactSegment::Reward, &manifest.rewards);
        Self::index_segment(
            &mut slots,
            ArtifactSegment::GoalMutation,
            &manifest.goal_mutations,
        );
        Self { slots }
    }
    fn index_segment(
        slots: &mut HashMap<ArtifactSegment, HashMap<String, u64>>,
        segment: ArtifactSegment,
        entries: &[ManifestEntry],
    ) {
        if entries.is_empty() {
            return;
        }
        let map = slots.entry(segment).or_insert_with(HashMap::new);
        for entry in entries {
            map.insert(entry.id.clone(), entry.slot);
        }
    }
    fn require_slot(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
    ) -> Result<u64, MemoryIrBuilderError> {
        self.slots
            .get(&segment)
            .and_then(|entries| entries.get(artifact_id))
            .copied()
            .ok_or_else(|| MemoryIrBuilderError::MissingSlot {
                segment,
                artifact_id: artifact_id.to_owned(),
            })
    }
}
