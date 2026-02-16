//! Minimal Delta-Merkle memory engine.
//!
//! Responsibilities:
//! - Gate executions via JudgmentProof (replay + epoch)
//! - Append committed deltas into the transaction log
//! - Emit Admission/Commit/Outcome proofs
//! - Provide deterministic hashing hooks for higher layers

use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    sync::Arc,
};

use bincode;
use hex;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const PAGE_SIZE: usize = 4096;

use crate::{
    delta::delta_validation::validate_delta,
    delta::{Delta, DeltaError},
    epoch::EpochCell,
    graph_log::{GraphDelta, GraphDeltaLog, GraphSnapshot},
    page::{PageAllocator, PageAllocatorConfig, PageLocation},
    primitives::Hash,
    proofs::{AdmissionProof, CommitProof, JudgmentProof, OutcomeProof},
    tlog::TransactionLog,
};

/// Canonical state owned exclusively by the memory engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalState {
    root_hash: Hash,
    page_store: PageStore
    page_hashes: std::collections::BTreeMap<PageID, Hash>,
    merkle_nodes: HashMap<(u8, u64), Hash>, // (level, index)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSlice {
    pub root_hash: String,
}

impl CanonicalState {
    pub fn new_empty() -> Self {
        Self {
            root_hash: [0u8; 32],
            pages: HashMap::new(),
            page_hashes: std::collections::BTreeMap::new(),
            merkle_nodes: HashMap::new(),
        }
    }

    // Removed blob-based constructor — state is now page-based

    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), DeltaError> {
        validate_delta(delta)?;

        self.page_store.write_page(delta.page_id.0, &delta.payload);
        let page_bytes = self.page_store.read_page(delta.page_id.0);

        // Update page payload
        let page_entry = self
            .pages
            .entry(delta.page_id)
            .or_insert_with(|| vec![0u8; PAGE_SIZE]);

        let len = delta.payload.len().min(PAGE_SIZE);
        page_entry[..len].copy_from_slice(&delta.payload[..len]);

        // Recompute page hash
        let mut page_hasher = Sha256::new();
        page_hasher.update(&page_entry);
        let page_hash: Hash = page_hasher.finalize().into();

        self.page_hashes.insert(delta.page_id, page_hash);

        // Incremental Merkle update
        self.update_merkle_tree(delta.page_id);

        Ok(())
    }

    pub fn root_hash(&self) -> Hash {
        self.root_hash
    }

    pub fn to_slice(&self) -> StateSlice {
        StateSlice {
            root_hash: hex::encode(self.root_hash),
        }
    }

    fn compute_root_hash(&self) -> Hash {
        // Only used for snapshot validation
        let mut hasher = Sha256::new();
        hasher.update(&self.payload);
        hasher.finalize().into()
    }

    pub fn load_from_disk(repo_root: &Path) -> io::Result<Option<Self>> {
        let snapshot_path = state_snapshot_path(repo_root);
        if !snapshot_path.exists() {
            return Ok(None);
        }
        let data = fs::read(&snapshot_path)?;
        let state: CanonicalState = bincode::deserialize(&data)
            .map_err(|err| io::Error::new(ErrorKind::InvalidData, err.to_string()))?;
        let expected = state.compute_root_hash();
        if state.root_hash != expected {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "snapshot root hash mismatch",
            ));
        }
        Ok(Some(state))
    }

    pub fn flush_to_disk(&self, repo_root: &Path) -> io::Result<()> {
        let snapshot_path = state_snapshot_path(repo_root);
        if let Some(parent) = snapshot_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = bincode::serialize(self)
            .map_err(|err| io::Error::new(ErrorKind::Other, err.to_string()))?;
        fs::write(snapshot_path, data)?;
        Ok(())
    }
}

fn state_snapshot_path(repo_root: &Path) -> PathBuf {
    repo_root
        .join("canon_store")
        .join("snapshots")
        .join("canonical_state.bin")
}

/// Configuration for initializing the memory engine.
pub struct MemoryEngineConfig {
    pub tlog_path: PathBuf,
    pub default_location: PageLocation,
    pub graph_log_path: PathBuf,
}

/// Errors produced by the memory engine.
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
    #[error("Graph delta replay failed: {0}")]
    GraphReplay(String),
    #[error("state mutation failed: {0}")]
    StateMutation(String),
}

/// Admission-stage errors.
#[derive(Debug, thiserror::Error)]
pub enum AdmissionError {
    #[error("invalid judgment proof")]
    InvalidJudgmentProof,
    #[error("stale epoch")]
    StaleEpoch,
    #[error("judgment already admitted")]
    AlreadyAdmitted,
}

/// Commit-stage errors.
#[derive(Debug, thiserror::Error)]
pub enum CommitError {
    #[error("failed to append to transaction log: {0}")]
    TlogWrite(#[source] std::io::Error),
}

/// Canonical memory engine (truth owner).
pub struct MemoryEngine {
    allocator: Arc<PageAllocator>,
    tlog: Arc<TransactionLog>,
    graph_log: Arc<GraphDeltaLog>,
    epoch: Arc<EpochCell>,
    admitted: RwLock<HashSet<Hash>>,
    deltas: RwLock<HashMap<Hash, Delta>>,
    state: Arc<RwLock<CanonicalState>>,
}

impl MemoryEngine {
    /// Initialize a new engine.
    pub fn new(config: MemoryEngineConfig) -> Result<Self, MemoryEngineError> {
        let allocator = Arc::new(PageAllocator::from_config(PageAllocatorConfig {
            default_location: config.default_location,
            initial_capacity: 1024,
        }));

        let tlog =
            Arc::new(TransactionLog::new(&config.tlog_path).map_err(MemoryEngineError::TlogOpen)?);
        let graph_log = Arc::new(
            GraphDeltaLog::new(&config.graph_log_path).map_err(MemoryEngineError::GraphLogOpen)?,
        );

        Ok(Self {
            allocator,
            tlog,
            graph_log,
            epoch: Arc::new(EpochCell::new(0)),
            admitted: RwLock::new(HashSet::new()),
            deltas: RwLock::new(HashMap::new()),
            state: Arc::new(RwLock::new(CanonicalState::new_empty())),
        })
    }

    /// Verify Merkle proof
    pub fn verify_proof(
        root: Hash,
        page_id: PageID,
        leaf_hash: Hash,
        proof: &[Hash],
    ) -> bool {
        let mut current_hash = leaf_hash;
        let mut index = page_id.0;

        for sibling_hash in proof {
            let (left, right) = if index % 2 == 0 {
                (current_hash, *sibling_hash)
            } else {
                (*sibling_hash, current_hash)
            };

            let mut hasher = Sha256::new();
            hasher.update(left);
            hasher.update(right);
            current_hash = hasher.finalize().into();

            index /= 2;
        }

        current_hash == root
    }

    /// Current epoch.
    pub fn current_epoch(&self) -> u64 {
        self.epoch.load().0 as u64
    }

    /// Track new deltas before they are committed (e.g., ingestion queue).
    pub fn register_delta(&self, delta: Delta) -> Hash {
        let hash = Self::hash_delta(&delta);
        self.deltas.write().insert(hash, delta);
        hash
    }

    /// Check if a judgment hash is already admitted.
    pub fn check_admitted(&self, judgment_hash: &Hash) -> bool {
        self.admitted.read().contains(judgment_hash)
    }

    /// Admission stage (D).
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

        let nonce = admitted.len() as u64;

        Ok(AdmissionProof {
            judgment_proof_hash: judgment_hash,
            epoch: current_epoch.0 as u64,
            nonce,
        })
    }

    /// Commit stage (E).
    pub fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta_hash: &Hash,
    ) -> Result<CommitProof, MemoryEngineError> {
        let delta = self
            .fetch_delta_by_hash(delta_hash)
            .ok_or(MemoryEngineError::DeltaNotFound)?;

        self.state
            .write()
            .apply_delta(&delta)
            .map_err(|err| MemoryEngineError::StateMutation(err.to_string()))?;

        self.epoch.increment();
        self.tlog
            .append(admission, delta.clone())
            .map_err(CommitError::TlogWrite)?;

        Ok(CommitProof {
            admission_proof_hash: admission.hash(),
            delta_hash: *delta_hash,
            state_hash: self.state.read().root_hash(),
        })
    }

    /// Commit batch of deltas in parallel
    pub fn commit_batch(
        &self,
        admission: &AdmissionProof,
        delta_hashes: &[Hash],
    ) -> Result<Vec<CommitProof>, MemoryEngineError> {
        use rayon::prelude::*;

        delta_hashes
            .par_iter()
            .map(|delta_hash| self.commit_delta(admission, delta_hash))
            .collect()
    }

    /// Outcome stage (F) — trivial success path for now.
    pub fn record_outcome(&self, commit: &CommitProof) -> OutcomeProof {
        OutcomeProof {
            commit_proof_hash: commit.hash(),
            success: true,
        }
    }

    /// Retrieve delta by its canonical hash.
    pub fn fetch_delta_by_hash(&self, hash: &Hash) -> Option<Delta> {
        self.deltas.read().get(hash).cloned()
    }

    /// Synthetic event ID hash (Admission + Commit + Outcome).
    pub fn compute_event_hash(
        &self,
        admission: &AdmissionProof,
        commit: &CommitProof,
        outcome: &OutcomeProof,
    ) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(admission.hash());
        hasher.update(commit.hash());
        hasher.update(outcome.hash());
        hasher.finalize().into()
    }

    fn verify_judgment_proof(&self, proof: &JudgmentProof) -> bool {
        proof.approved && proof.hash != [0u8; 32]
    }

    fn hash_delta(delta: &Delta) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(&delta.page_id.0.to_be_bytes());
        hasher.update(&delta.epoch.0.to_be_bytes());
        for bit in &delta.mask {
            hasher.update(&[*bit as u8]);
        }
        hasher.update(&delta.payload);
        hasher.finalize().into()
    }

    fn update_merkle_tree(&mut self, page_id: PageID) {
        let mut current_hash = self.page_hashes[&page_id];
        let mut index = page_id.0;

        // Sparse Merkle Tree with fixed depth = 64 (u64 PageID)
        for level in 0..64 {
            self.merkle_nodes.insert((level, index), current_hash);

            let sibling_index = if index % 2 == 0 {
                index + 1
            } else {
                index - 1
            };

            let sibling_hash = self
                .merkle_nodes
                .get(&(level, sibling_index))
                .copied()
                .unwrap_or([0u8; 32]);

            let (left, right) = if index % 2 == 0 {
                (current_hash, sibling_hash)
            } else {
                (sibling_hash, current_hash)
            };

            let mut hasher = Sha256::new();
            hasher.update(left);
            hasher.update(right);
            current_hash = hasher.finalize().into();

            index /= 2;
        }

        self.root_hash = current_hash;
    }

    /// Generate Merkle inclusion proof for a page
    pub fn merkle_proof(&self, page_id: PageID) -> Option<Vec<Hash>> {
        if !self.page_hashes.contains_key(&page_id) {
            return None;
        }

        let mut proof = Vec::with_capacity(64);
        let mut index = page_id.0;

        for level in 0..64 {
            let sibling_index = if index % 2 == 0 {
                index + 1
            } else {
                index - 1
            };

            let sibling_hash = self
                .merkle_nodes
                .get(&(level, sibling_index))
                .copied()
                .unwrap_or([0u8; 32]);

            proof.push(sibling_hash);
            index /= 2;
        }

        Some(proof)
    }

    fn compute_merkle_root(page_hashes: &HashMap<PageID, Hash>) -> Hash {
        if page_hashes.is_empty() {
            return [0u8; 32];
        }

        let mut hashes: Vec<Hash> = page_hashes.values().cloned().collect();
        hashes.sort(); // deterministic order

        while hashes.len() > 1 {
            let mut next = Vec::new();
            for pair in hashes.chunks(2) {
                let mut hasher = Sha256::new();
                hasher.update(pair[0]);
                if pair.len() == 2 {
                    hasher.update(pair[1]);
                } else {
                    hasher.update(pair[0]); // duplicate last if odd
                }
                next.push(hasher.finalize().into());
            }
            hashes = next;
        }

        hashes[0]
    }

    /// Append a graph delta to the log.
    pub fn commit_graph_delta(&self, delta: GraphDelta) -> Result<(), MemoryEngineError> {
        self.graph_log
            .append(delta)
            .map_err(MemoryEngineError::GraphLogIo)
    }

    /// Replays all committed graph deltas into a snapshot.
    pub fn materialized_graph(&self) -> Result<GraphSnapshot, MemoryEngineError> {
        self.graph_log
            .replay_snapshot()
            .map_err(|e| MemoryEngineError::GraphReplay(e.to_string()))
    }
}
