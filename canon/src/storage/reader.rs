use crate::ir::world_model::WorldModel;
use crate::ir::{
    AppliedDeltaRecord, CallEdge, SystemState, CanonicalMeta,
    StateChange as CanonicalDelta, ChangeAdmission, EnumNode, ErrorArtifact,
    ExecutionRecord, ExternalDependency, Function, GoalMutation, GpuFunction, ImplBlock,
    Decision, Rule, Learning, LoopPolicy, Module, ModuleEdge, Plan, PolicyParameters,
    Project, Proof, Proposal, RewardRecord, Struct, SystemGraph, Tick, ExecutionEpoch,
    ExecutionGraph, Trait, VersionContract,
};
use crate::storage::layout::{
    ArtifactSegment, LayoutError, MemoryArtifactLayout, META_MANIFEST_SLOT,
    META_STATE_SLOT, PAGE_BYTES, SINGLETON_SLOT,
};
use crate::storage::manifest::{ArtifactManifest, ManifestEntry};
use crate::storage::types::FileHashRecord;
use bincode::{config::DefaultOptions, Options};
use database::canonical_state::MerkleState;
use database::hash::gpu::create_gpu_backend;
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;
const LENGTH_PREFIX: usize = 8;
pub struct StateReader<'a> {
    state: &'a MerkleState,
    layout: MemoryArtifactLayout,
}
impl<'a> StateReader<'a> {
    pub fn new(state: &'a MerkleState) -> Self {
        Self {
            state,
            layout: MemoryArtifactLayout::new(),
        }
    }
    pub fn read_ir(&self) -> Result<SystemState, MemoryIrReadError> {
        let manifest: ArtifactManifest = self
            .read_entry_in_slot(ArtifactSegment::Meta, "manifest", META_MANIFEST_SLOT)?;
        let meta: CanonicalMeta = self
            .read_entry_in_slot(ArtifactSegment::Meta, "meta", META_STATE_SLOT)?;
        let version_contract: VersionContract = self
            .read_singleton(ArtifactSegment::VersionContract, "version_contract")?;
        let project: Project = self.read_singleton(ArtifactSegment::Project, "project")?;
        let world_model: WorldModel = self
            .read_singleton(ArtifactSegment::WorldModel, "world_model")?;
        let modules = self.read_collection(ArtifactSegment::Module, &manifest.modules)?;
        let structs = self.read_collection(ArtifactSegment::Struct, &manifest.structs)?;
        let enums = self.read_collection(ArtifactSegment::Enum, &manifest.enums)?;
        let traits = self.read_collection(ArtifactSegment::Trait, &manifest.traits)?;
        let impls = self.read_collection(ArtifactSegment::ImplBlock, &manifest.impls)?;
        let functions = self
            .read_collection(ArtifactSegment::Function, &manifest.functions)?;
        let module_edges = self
            .read_collection(ArtifactSegment::ModuleEdge, &manifest.module_edges)?;
        let call_edges = self
            .read_collection(ArtifactSegment::CallEdge, &manifest.call_edges)?;
        let ticks = self.read_collection(ArtifactSegment::Tick, &manifest.ticks)?;
        let tick_graphs = self
            .read_collection(ArtifactSegment::TickGraph, &manifest.tick_graphs)?;
        let system_graphs = self
            .read_collection(ArtifactSegment::SystemGraph, &manifest.system_graphs)?;
        let loop_policies = self
            .read_collection(ArtifactSegment::LoopPolicy, &manifest.loop_policies)?;
        let tick_epochs = self
            .read_collection(ArtifactSegment::TickEpoch, &manifest.tick_epochs)?;
        let policy_parameters = self
            .read_collection(ArtifactSegment::Policy, &manifest.policies)?;
        let plans = self.read_collection(ArtifactSegment::Plan, &manifest.plans)?;
        let executions = self
            .read_collection(ArtifactSegment::Execution, &manifest.executions)?;
        let admissions = self
            .read_collection(ArtifactSegment::Admission, &manifest.admissions)?;
        let applied_deltas = self
            .read_collection(ArtifactSegment::AppliedDelta, &manifest.applied_deltas)?;
        let gpu_functions = self
            .read_collection(ArtifactSegment::GpuFunction, &manifest.gpu_functions)?;
        let proposals = self
            .read_collection(ArtifactSegment::Proposal, &manifest.proposals)?;
        let judgments = self
            .read_collection(ArtifactSegment::Judgment, &manifest.judgments)?;
        let predicates = self
            .read_collection(
                ArtifactSegment::JudgmentPredicate,
                &manifest.judgment_predicates,
            )?;
        let deltas = self
            .read_collection(ArtifactSegment::DeltaDef, &manifest.delta_defs)?;
        let proofs = self.read_collection(ArtifactSegment::Proof, &manifest.proofs)?;
        let learnings = self
            .read_collection(ArtifactSegment::Learning, &manifest.learnings)?;
        let errors = self
            .read_collection(ArtifactSegment::ErrorArtifact, &manifest.errors)?;
        let dependencies = self
            .read_collection(ArtifactSegment::Dependency, &manifest.dependencies)?;
        let file_hash_records: Vec<FileHashRecord> = self
            .read_collection(ArtifactSegment::FileHash, &manifest.file_hashes)?;
        let mut file_hashes = HashMap::new();
        for record in file_hash_records {
            file_hashes.insert(record.path, record.hash);
        }
        let reward_deltas = self
            .read_collection(ArtifactSegment::Reward, &manifest.rewards)?;
        let goal_mutations = self
            .read_collection(ArtifactSegment::GoalMutation, &manifest.goal_mutations)?;
        Ok(SystemState {
            meta,
            version_contract,
            project,
            modules,
            module_edges,
            structs,
            enums,
            traits,
            impls,
            functions,
            call_edges,
            tick_graphs,
            system_graphs,
            loop_policies,
            ticks,
            tick_epochs,
            policy_parameters,
            plans,
            executions,
            admissions,
            applied_deltas,
            gpu_functions,
            proposals,
            judgments,
            judgment_predicates: predicates,
            deltas,
            proofs,
            learning: learnings,
            errors,
            dependencies,
            file_hashes,
            reward_deltas,
            world_model,
            goal_mutations,
        })
    }
    pub fn read_ir_from_checkpoint<P: AsRef<Path>>(
        path: P,
    ) -> Result<SystemState, MemoryIrReadError> {
        let backend = create_gpu_backend();
        let state = MerkleState::restore_from_checkpoint(path, backend)?;
        let reader = StateReader::new(&state);
        reader.read_ir()
    }
    fn read_collection<T: DeserializeOwned>(
        &self,
        segment: ArtifactSegment,
        entries: &[ManifestEntry],
    ) -> Result<Vec<T>, MemoryIrReadError> {
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            out.push(self.read_entry_in_slot(segment, &entry.id, entry.slot)?);
        }
        Ok(out)
    }
    fn read_singleton<T: DeserializeOwned>(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
    ) -> Result<T, MemoryIrReadError> {
        self.read_entry_in_slot(segment, artifact_id, SINGLETON_SLOT)
    }
    fn read_entry_in_slot<T: DeserializeOwned>(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        slot: u64,
    ) -> Result<T, MemoryIrReadError> {
        let bytes = self.read_bytes(segment, artifact_id, slot)?;
        DefaultOptions::new()
            .with_fixint_encoding()
            .allow_trailing_bytes()
            .deserialize(&bytes)
            .map_err(MemoryIrReadError::Decode)
    }
    fn read_bytes(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        slot: u64,
    ) -> Result<Vec<u8>, MemoryIrReadError> {
        let first_address = self.layout.page_for(segment, artifact_id, slot, 0)?;
        let first_page = self.state.read_page(first_address.page_id);
        if first_page.len() < LENGTH_PREFIX {
            return Err(MemoryIrReadError::Corrupt {
                segment,
                artifact_id: artifact_id.to_owned(),
                reason: "page shorter than prefix".to_owned(),
            });
        }
        let mut len_buf = [0u8; LENGTH_PREFIX];
        len_buf.copy_from_slice(&first_page[..LENGTH_PREFIX]);
        let payload_len = u64::from_le_bytes(len_buf) as usize;
        let total_len = LENGTH_PREFIX + payload_len;
        let chunk_count = ((total_len + PAGE_BYTES - 1) / PAGE_BYTES) as u64;
        let mut buffer = vec![0u8; chunk_count.max(1) as usize * PAGE_BYTES];
        let first_len = PAGE_BYTES.min(buffer.len());
        buffer[..first_len].copy_from_slice(&first_page[..first_len]);
        if chunk_count > 1 {
            for chunk_index in 1..chunk_count {
                let address = self
                    .layout
                    .page_for(segment, artifact_id, slot, chunk_index)?;
                let page = self.state.read_page(address.page_id);
                let start = chunk_index as usize * PAGE_BYTES;
                let chunk_len = (buffer.len() - start).min(PAGE_BYTES);
                buffer[start..start + chunk_len].copy_from_slice(&page[..chunk_len]);
            }
        }
        buffer.truncate(total_len);
        Ok(buffer[LENGTH_PREFIX..].to_vec())
    }
}
#[derive(Debug, Error)]
pub enum MemoryIrReadError {
    #[error("layout error: {0}")]
    Layout(#[from] LayoutError),
    #[error("decode error: {0}")]
    Decode(#[from] bincode::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("corrupt entry {segment:?}/{artifact_id}: {reason}")]
    Corrupt { segment: ArtifactSegment, artifact_id: String, reason: String },
}
