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

use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use crate::hash::gpu::create_gpu_backend;

use crate::{
    canonical_state::CanonicalState,
    delta::Delta,
    epoch::EpochCell,
    graph_log::{GraphDelta, GraphDeltaLog, GraphSnapshot},
    primitives::Hash,
    proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof},
    tlog::TransactionLog,
};

// ===============================
// Domain Separation
// ===============================

const DOMAIN_DELTA: &[u8] = b"DELTA_V1";
const DOMAIN_EVENT: &[u8] = b"EVENT_V1";

// ===============================
// Config
// ===============================

pub struct MemoryEngineConfig {
    pub tlog_path: PathBuf,
    pub graph_log_path: PathBuf,
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
    pub(crate) tlog: Arc<TransactionLog>,
    pub(crate) graph_log: Arc<GraphDeltaLog>,
    pub(crate) epoch: Arc<EpochCell>,
    pub(crate) admitted: RwLock<HashSet<Hash>>,
    pub(crate) deltas: RwLock<HashMap<Hash, Delta>>,
    pub(crate) state: Arc<RwLock<CanonicalState>>,
}

impl MemoryEngine {
    pub fn new(config: MemoryEngineConfig) -> Result<Self, MemoryEngineError> {
        let tlog =
            Arc::new(TransactionLog::new(&config.tlog_path).map_err(MemoryEngineError::TlogOpen)?);

        let graph_log =
            Arc::new(GraphDeltaLog::new(&config.graph_log_path)
                .map_err(MemoryEngineError::GraphLogOpen)?);

        Ok(Self {
            tlog,
            graph_log,
            epoch: Arc::new(EpochCell::new(0)),
            admitted: RwLock::new(HashSet::new()),
            deltas: RwLock::new(HashMap::new()),
            state: Arc::new(RwLock::new(
                CanonicalState::new_empty(create_gpu_backend()),
            )),
        })
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
        use rayon::prelude::*;

        let deltas: Vec<Delta> = delta_hashes
            .iter()
            .map(|h| {
                self.fetch_delta_by_hash(h)
                    .ok_or(MemoryEngineError::DeltaNotFound)
            })
            .collect::<Result<_, _>>()?;

        {
            let mut state = self.state.write();
            state
                .apply_deltas_batch(&deltas)
                .map_err(|_| MemoryEngineError::DeltaNotFound)?;
            state.page_store.flush().ok();
        }

        self.epoch.increment();

        for delta in &deltas {
            self.tlog
                .append(admission, delta.clone())
                .map_err(CommitError::TlogWrite)?;
        }

        let root = self.state.read().root_hash();

        Ok(delta_hashes
            .par_iter()
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
        self.graph_log
            .append(delta)
            .map_err(MemoryEngineError::GraphLogIo)
    }

    pub fn materialized_graph(&self) -> Result<GraphSnapshot, MemoryEngineError> {
        self.graph_log
            .replay_snapshot()
            .map_err(|e| MemoryEngineError::GraphLogIo(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            )))
    }

    // ===============================
    // Internal
    // ===============================

    fn verify_judgment_proof(&self, proof: &JudgmentProof) -> bool {
        proof.approved && proof.hash != [0u8; 32]
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

use crate::engine::Engine;

impl Engine for MemoryEngine {
    type Error = MemoryEngineError;

    fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, Self::Error> {
        MemoryEngine::admit_execution(self, judgment_proof)
            .map_err(MemoryEngineError::Admission)
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
