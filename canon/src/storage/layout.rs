use thiserror::Error;

/// Memory-engine page size in bytes.
pub const PAGE_BYTES: usize = 4096;

/// Number of pages reserved per artifact slot.
pub const PAGES_PER_SLOT: u64 = 32;

/// Total slots per segment (must be large enough for all artifacts).
const SEGMENT_SLOTS: u64 = 1 << 10; // 1,024 slots per segment

/// Fixed slots for singleton artifacts stored in the `Meta` segment.
pub const META_MANIFEST_SLOT: u64 = 0;
pub const META_STATE_SLOT: u64 = 1;

/// Default slot for singleton artifacts stored in their dedicated segment.
pub const SINGLETON_SLOT: u64 = 0;

/// Artifact segments scoped to deterministic page ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactSegment {
    Meta,
    VersionContract,
    Project,
    Module,
    Struct,
    Enum,
    Trait,
    ImplBlock,
    Function,
    ModuleEdge,
    CallEdge,
    Tick,
    TickGraph,
    SystemGraph,
    LoopPolicy,
    TickEpoch,
    Policy,
    Plan,
    Execution,
    Admission,
    AppliedDelta,
    GpuFunction,
    Proposal,
    Judgment,
    JudgmentPredicate,
    DeltaDef,
    Proof,
    Learning,
    ErrorArtifact,
    Dependency,
    FileHash,
    Reward,
    WorldModel,
    GoalMutation,
}

impl ArtifactSegment {
    pub fn ordinal(self) -> u64 {
        match self {
            ArtifactSegment::Meta => 0,
            ArtifactSegment::VersionContract => 1,
            ArtifactSegment::Project => 2,
            ArtifactSegment::Module => 3,
            ArtifactSegment::Struct => 4,
            ArtifactSegment::Enum => 5,
            ArtifactSegment::Trait => 6,
            ArtifactSegment::ImplBlock => 7,
            ArtifactSegment::Function => 8,
            ArtifactSegment::ModuleEdge => 9,
            ArtifactSegment::CallEdge => 10,
            ArtifactSegment::Tick => 11,
            ArtifactSegment::TickGraph => 12,
            ArtifactSegment::SystemGraph => 13,
            ArtifactSegment::LoopPolicy => 14,
            ArtifactSegment::TickEpoch => 15,
            ArtifactSegment::Policy => 16,
            ArtifactSegment::Plan => 17,
            ArtifactSegment::Execution => 18,
            ArtifactSegment::Admission => 19,
            ArtifactSegment::AppliedDelta => 20,
            ArtifactSegment::GpuFunction => 21,
            ArtifactSegment::Proposal => 22,
            ArtifactSegment::Judgment => 23,
            ArtifactSegment::JudgmentPredicate => 24,
            ArtifactSegment::DeltaDef => 25,
            ArtifactSegment::Proof => 26,
            ArtifactSegment::Learning => 27,
            ArtifactSegment::ErrorArtifact => 28,
            ArtifactSegment::Dependency => 29,
            ArtifactSegment::FileHash => 30,
            ArtifactSegment::Reward => 31,
            ArtifactSegment::WorldModel => 32,
            ArtifactSegment::GoalMutation => 33,
        }
    }

    fn base_page(self) -> u64 {
        self.ordinal() * SEGMENT_SLOTS * PAGES_PER_SLOT
    }
}

/// Memory layout helper that maps artifact ids → deterministic page ids.
#[derive(Debug, Clone)]
pub struct MemoryArtifactLayout;

impl MemoryArtifactLayout {
    pub fn new() -> Self {
        Self
    }

    pub fn page_for(
        &self,
        segment: ArtifactSegment,
        artifact_id: &str,
        slot: u64,
        chunk_index: u64,
    ) -> Result<PageAddress, LayoutError> {
        if chunk_index >= PAGES_PER_SLOT {
            return Err(LayoutError::ChunkOverflow {
                segment,
                artifact_id: artifact_id.to_owned(),
                chunk_index,
                limit: PAGES_PER_SLOT,
            });
        }
        if slot >= SEGMENT_SLOTS {
            return Err(LayoutError::SlotOverflow {
                segment,
                artifact_id: artifact_id.to_owned(),
                slot,
                limit: SEGMENT_SLOTS,
            });
        }
        let page_id = segment.base_page() + slot * PAGES_PER_SLOT + chunk_index;
        Ok(PageAddress { page_id, slot })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PageAddress {
    pub page_id: u64,
    pub slot: u64,
}

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error(
        "artifact `{artifact_id}` ({segment:?}) requires chunk {chunk_index} but only {limit} chunks allowed"
    )]
    ChunkOverflow {
        segment: ArtifactSegment,
        artifact_id: String,
        chunk_index: u64,
        limit: u64,
    },
    #[error(
        "artifact `{artifact_id}` ({segment:?}) assigned slot {slot} but only {limit} slots available"
    )]
    SlotOverflow {
        segment: ArtifactSegment,
        artifact_id: String,
        slot: u64,
        limit: u64,
    },
}
