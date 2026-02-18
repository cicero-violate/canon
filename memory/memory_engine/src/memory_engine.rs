//! Memory Engine — orchestration layer only.
//!
//! Responsibilities:
//! - Admission (replay protection + epoch)
//! - Delegation to commit logic
//! - Event hash construction
//!
//! Canonical state + Merkle logic live in dedicated modules.

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};

use crate::hash::gpu::create_gpu_backend;
use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};

use crate::{
    canonical_state::MerkleState,
    delta::Delta,
    epoch::EpochCell,
    graph_log::{GraphDelta, GraphSnapshot},
    persistence::mmap_log::MmapLog,
    persistence::root_header::RootHeader,
    primitives::{DeltaID, Hash, PageID},
    proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof},
};

// ===============================
// Domain Separation
// ===============================

const DOMAIN_DELTA: &[u8] = b"DELTA_V1";
const DOMAIN_EVENT: &[u8] = b"EVENT_V1";
const DOMAIN_TRANSITION_JUDGMENT: &[u8] = b"TRANSITION_JUDGMENT_V1";

// ===============================
// Config
// ===============================

pub struct MemoryEngineConfig {
    pub tlog_path: PathBuf,
}

// ===============================
// Errors
// ===============================

#[derive(Debug, thiserror::Error)]
pub enum MemoryEngineError {
    #[error("Failed to open transaction log: {0}")]
    TlogOpen(#[source] std::io::Error),

    #[error("Failed to open graph delta log: {0}")]
    GraphLogOpen(#[source] std::io::Error),

    #[error("Graph delta log IO error: {0}")]
    GraphLogIo(#[source] std::io::Error),

    #[error("Admission failed: {0}")]
    Admission(#[from] AdmissionError),

    #[error("Commit failed: {0}")]
    Commit(#[from] CommitError),

    #[error("Delta not found for hash")]
    DeltaNotFound,

    #[error("state hash mismatch (expected {expected:?}, got {provided:?})")]
    StateMismatch { expected: Hash, provided: Hash },
}

#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("invalid judgment proof")]
    InvalidJudgmentProof,

    #[error("stale epoch")]
    StaleEpoch,

    #[error("judgment already admitted")]
    AlreadyAdmitted,
}

#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("failed to append to transaction log: {0}")]
    TlogWrite(#[source] std::io::Error),
}

// ===============================
// Engine
// ===============================
pub struct MemoryEngine {
    pub(crate) wal: Arc<Mutex<MmapLog>>,
    pub(crate) epoch: Arc<EpochCell>,
    pub(crate) admitted: RwLock<HashSet<Hash>>,
    pub(crate) deltas: RwLock<HashMap<Hash, Delta>>,
    pub(crate) state: Arc<RwLock<MerkleState>>,
}

impl MemoryEngine {
    pub fn new(config: MemoryEngineConfig) -> Result<Self, MemoryEngineError> {
        let wal = Arc::new(Mutex::new(
            MmapLog::open(&config.tlog_path, 64 * 1024 * 1024)
                .map_err(MemoryEngineError::TlogOpen)?,
        ));

        let backend = create_gpu_backend();

        let mut state = MerkleState::new_empty(backend);

        if let Some(header) = wal
            .lock()
            .read_latest_root()
            .map_err(MemoryEngineError::TlogOpen)?
        {
            // Extract records and drop WAL lock before any heavy work
            let records = {
                let guard = wal.lock();
                guard.scan_records()
            };

            for rec in records {
                let (admission, delta): (AdmissionProof, Delta) = bincode::deserialize(&rec)
                    .map_err(|e| {
                        MemoryEngineError::TlogOpen(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e,
                        ))
                    })?;
                let _ = admission;
                state
                    .apply_delta(&delta)
                    .map_err(|_| MemoryEngineError::DeltaNotFound)?;
            }

            // Single canonical rebuild to compute final root
            state
                .apply_deltas_batch(&[])
                .map_err(|_| MemoryEngineError::DeltaNotFound)?;

            if state.root_hash() != header.root_hash {
                panic!("WAL replay root mismatch");
            }
        }

        let state = Arc::new(RwLock::new(state));

        Ok(Self {
            wal,
            epoch: Arc::new(EpochCell::new(0)),
            admitted: RwLock::new(HashSet::new()),
            deltas: RwLock::new(HashMap::new()),
            state,
        })
    }

    pub fn checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let state = self.state.read();
        state.checkpoint(path)
    }

    pub fn current_root_hash(&self) -> Hash {
        self.state.read().root_hash()
    }

    pub fn compact(&self) -> std::io::Result<()> {
        // 1) Write deterministic snapshot
        {
            let state = self.state.read();
            state.checkpoint("checkpoint.bin")?;
        }

        // 2) Truncate WAL only after snapshot succeeds
        {
            let mut wal = self.wal.lock();
            wal.truncate()?;
            wal.flush()?;
        }

        Ok(())
    }

    // ===============================
    // Admission
    // ===============================

    pub fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, AdmissionError> {
        if !self.verify_judgment_proof(judgment_proof) {
            return Err(AdmissionError::InvalidJudgmentProof);
        }

        let current_epoch = self.epoch.load();

        if judgment_proof.timestamp < current_epoch.0 as u64 {
            return Err(AdmissionError::StaleEpoch);
        }

        let judgment_hash = judgment_proof.hash();
        let mut admitted = self.admitted.write();

        if !admitted.insert(judgment_hash) {
            return Err(AdmissionError::AlreadyAdmitted);
        }

        Ok(AdmissionProof {
            judgment_proof_hash: judgment_hash,
            epoch: current_epoch.0 as u64,
            nonce: admitted.len() as u64,
        })
    }

    // ===============================
    // Outcome
    // ===============================

    pub fn record_outcome(&self, commit: &CommitProof) -> OutcomeProof {
        OutcomeProof {
            commit_proof_hash: commit.hash(),
            success: true,
        }
    }

    // ===============================
    // Delta Registry
    // ===============================

    pub fn register_delta(&self, delta: Delta) -> Hash {
        let hash = Self::hash_delta(&delta);
        self.deltas.write().insert(hash, delta);
        hash
    }

    pub fn fetch_delta_by_hash(&self, hash: &Hash) -> Option<Delta> {
        self.deltas.read().get(hash).cloned()
    }

    // ===============================
    // Commit (delegated)
    // ===============================

    pub fn commit_batch(
        &self,
        admission: &AdmissionProof,
        delta_hashes: &[Hash],
    ) -> Result<Vec<CommitProof>, MemoryEngineError> {
        // Resolve deltas
        let deltas: Vec<Delta> = delta_hashes
            .iter()
            .map(|h| {
                self.fetch_delta_by_hash(h)
                    .ok_or(MemoryEngineError::DeltaNotFound)
            })
            .collect::<Result<_, _>>()?;

        // Apply all deltas (device writes only)
        {
            let mut state = self.state.write();
            state
                .apply_deltas_batch(&deltas)
                .map_err(|_| MemoryEngineError::DeltaNotFound)?;
        }

        self.epoch.increment();

        // Persist log
        for delta in &deltas {
            let encoded = bincode::serialize(&(admission, delta)).map_err(|e| {
                MemoryEngineError::TlogOpen(std::io::Error::new(std::io::ErrorKind::Other, e))
            })?;
            self.wal
                .lock()
                .append(&encoded)
                .map_err(MemoryEngineError::TlogOpen)?;
        }

        let root = self.state.read().root_hash();

        // Crash-consistent root update
        {
            let header = RootHeader {
                generation: self.epoch.load().0 as u64,
                tree_size: self.state.read().tree_size,
                root_hash: root,
            };
            self.wal
                .lock()
                .write_root_header(&header)
                .map_err(MemoryEngineError::TlogOpen)?;
        }

        Ok(delta_hashes
            .iter()
            .map(|delta_hash| CommitProof {
                admission_proof_hash: admission.hash(),
                delta_hash: *delta_hash,
                state_hash: root,
            })
            .collect())
    }

    // ===============================
    // Event Hash
    // ===============================

    pub fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_EVENT);
        hasher.update(admission.hash());
        hasher.update(commit.hash());
        hasher.update(outcome.hash());
        hasher.finalize().into()
    }

    // ===============================
    // Graph Log
    // ===============================

    pub fn commit_graph_delta(&self, delta: GraphDelta) -> Result<(), MemoryEngineError> {
        const GRAPH_BASE: u64 = 1 << 48;

        let page_id = GRAPH_BASE + self.epoch.increment().0 as u64;

        let mask = vec![true; delta.payload.len()];

        let d = Delta::new_dense(
            DeltaID(page_id),
            PageID(page_id),
            crate::epoch::Epoch(0),
            delta.payload,
            mask,
            crate::delta::Source("graph.partition".into()),
        )
        .map_err(|_| MemoryEngineError::DeltaNotFound)?;

        {
            let mut state = self.state.write();
            state
                .apply_delta(&d)
                .map_err(|_| MemoryEngineError::DeltaNotFound)?;
        }

        Ok(())
    }

    pub fn materialized_graph(&self) -> Result<GraphSnapshot, MemoryEngineError> {
        const GRAPH_BASE: u64 = 1 << 48;

        let state = self.state.read();
        let mut snapshot = GraphSnapshot::default();

        for page_id in GRAPH_BASE..state.max_leaves {
            let page = state.read_page(page_id);
            if page.iter().any(|b| *b != 0) {
                snapshot.payload.extend_from_slice(page);
            }
        }

        Ok(snapshot)
    }

    // ===============================
    // Internal
    // ===============================

    fn verify_judgment_proof(&self, proof: &JudgmentProof) -> bool {
        proof.approved && proof.hash != [0u8; 32]
    }

    fn transition_judgment(&self, previous_state: Hash, delta_hash: Hash) -> JudgmentProof {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_TRANSITION_JUDGMENT);
        hasher.update(previous_state);
        hasher.update(delta_hash);

        JudgmentProof {
            approved: true,
            timestamp: self.epoch.load().0 as u64,
            hash: hasher.finalize().into(),
        }
    }

    fn hash_delta(delta: &Delta) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(DOMAIN_DELTA);
        hasher.update(&delta.page_id.0.to_be_bytes());
        hasher.update(&delta.epoch.0.to_be_bytes());

        for bit in &delta.mask {
            hasher.update(&[*bit as u8]);
        }

        hasher.update(&delta.payload);
        hasher.finalize().into()
    }
}

use crate::{engine::Engine, transition::MemoryTransition};

impl Engine for MemoryEngine {
    type Error = MemoryEngineError;

    fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, Self::Error> {
        MemoryEngine::admit_execution(self, judgment_proof).map_err(MemoryEngineError::Admission)
    }

    fn register_delta(&self, delta: Delta) -> Hash {
        MemoryEngine::register_delta(self, delta)
    }

    fn fetch_delta_by_hash(&self, hash: &Hash) -> Option<Delta> {
        MemoryEngine::fetch_delta_by_hash(self, hash)
    }

    fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta_hash: &Hash,
    ) -> Result<CommitProof, Self::Error> {
        MemoryEngine::commit_delta(self, admission, delta_hash)
    }

    fn commit_batch(
        &self,
        admission: &AdmissionProof,
        delta_hashes: &[Hash],
    ) -> Result<Vec<CommitProof>, Self::Error> {
        MemoryEngine::commit_batch(self, admission, delta_hashes)
    }

    fn record_outcome(&self, commit: &CommitProof) -> OutcomeProof {
        MemoryEngine::record_outcome(self, commit)
    }

    fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> Hash {
        MemoryEngine::compute_event_hash(self, admission, commit, outcome)
    }

    fn commit_graph_delta(&self, delta: GraphDelta) -> Result<(), Self::Error> {
        MemoryEngine::commit_graph_delta(self, delta)
    }

    fn materialized_graph(&self) -> Result<GraphSnapshot, Self::Error> {
        MemoryEngine::materialized_graph(self)
    }
}

impl MemoryTransition for MemoryEngine {
    fn genesis(&self) -> Hash {
        self.current_root_hash()
    }

    fn step(&self, state: Hash, delta: Delta) -> Result<(Hash, CommitProof), MemoryEngineError> {
        let current = self.current_root_hash();
        if current != state {
            return Err(MemoryEngineError::StateMismatch {
                expected: current,
                provided: state,
            });
        }

        let delta_hash = self.register_delta(delta);
        let judgment = self.transition_judgment(state, delta_hash);
        let admission = self.admit_execution(&judgment)?;
        let commit = self.commit_delta(&admission, &delta_hash)?;

        Ok((self.current_root_hash(), commit))
    }
}
