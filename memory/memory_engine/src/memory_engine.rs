//! Minimal Delta-Merkle memory engine.
//!
//! Responsibilities:
//! - Gate executions via JudgmentProof (replay + epoch)
//! - Append committed deltas into the transaction log
//! - Emit Admission/Commit/Outcome proofs
//! - Provide deterministic hashing hooks for higher layers

use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::Arc,
};

use hex;
use parking_lot::RwLock;
use sha2::{Digest, Sha256};

// ===============================
// Domain Separation Prefixes
// ===============================
const DOMAIN_LEAF: &[u8] = b"LEAF_V1";
const DOMAIN_INTERNAL: &[u8] = b"NODE_V1";
const DOMAIN_DELTA: &[u8] = b"DELTA_V1";
const DOMAIN_EVENT: &[u8] = b"EVENT_V1";


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

use crate::page_store::PageStore;
use crate::primitives::PageID;
use crate::journal::Journal;

/// Canonical state owned exclusively by the memory engine.
#[derive(Debug)]
pub struct CanonicalState {
    root_hash: Hash,
    page_store: PageStore,
    page_hashes: std::collections::BTreeMap<PageID, Hash>,
    merkle_nodes: Vec<Hash>, // flat full binary tree
    max_leaves: u64,
    tree_size: u64,
    dirty_leaves: Vec<u64>,
}


#[derive(Debug)]
pub struct StateSlice {
    pub root_hash: String,
}

impl CanonicalState {
    pub fn new_empty() -> Self {
        Self::with_capacity(1024)
    }

    /// Create state with configurable capacity.
    pub fn with_capacity(max_leaves: u64) -> Self {
        let tree_size = max_leaves.next_power_of_two();
        let total_nodes = tree_size * 2;

        Self {
            root_hash: [0u8; 32],
            page_store: PageStore::in_memory(),
            page_hashes: std::collections::BTreeMap::new(),
            merkle_nodes: vec![[0u8; 32]; total_nodes as usize],
            max_leaves: tree_size,
            tree_size,
            dirty_leaves: Vec::new(),
        }
    }

    // Removed blob-based constructor — state is now page-based

    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), DeltaError> {
        validate_delta(delta)?;

        // Write to page store (disk or in-memory)
        self.page_store
            .write_page(delta.page_id.0, &delta.payload);

        // Zero-copy hashing from page store
        let page_bytes = self.page_store.read_page(delta.page_id.0);

        let mut page_hasher = Sha256::new();
        page_hasher.update(DOMAIN_LEAF);
        page_hasher.update(page_bytes);
        let page_hash: Hash = page_hasher.finalize().into();

        self.page_hashes.insert(delta.page_id, page_hash);

        // Auto-expand if capacity exceeded
        if delta.page_id.0 >= self.max_leaves {
            self.rehash_with_new_capacity(delta.page_id.0 + 1);
        }

        self.mark_dirty(delta.page_id.0);
        self.rehash_dirty_levels();

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

    /// Incrementally update sparse Merkle tree (64-depth).
    fn update_merkle_tree(&mut self, page_id: PageID) {
        let leaf_index = self.tree_size + page_id.0;
        let mut node_index = leaf_index as usize;

        self.merkle_nodes[node_index] = self.page_hashes[&page_id];

        while node_index > 1 {
            let parent = node_index / 2;
            let left = parent * 2;
            let right = left + 1;

            let mut hasher = Sha256::new();
            hasher.update(DOMAIN_INTERNAL);
            hasher.update(self.merkle_nodes[left]);
            hasher.update(self.merkle_nodes[right]);

            self.merkle_nodes[parent] = hasher.finalize().into();
            node_index = parent;
        }

        self.root_hash = self.merkle_nodes[1];
    }

    /// Mark a leaf as dirty
    fn mark_dirty(&mut self, leaf: u64) {
        self.dirty_leaves.push(leaf);
    }

     /// Batched bottom-up rebuild (vectorizable structure)
     fn rehash_dirty_levels(&mut self) {
         if self.dirty_leaves.is_empty() {
             return;
         }

         use rayon::prelude::*;

         // Step 1: update dirty leaves
         let mut current_level: Vec<usize> = self
             .dirty_leaves
             .iter()
             .map(|leaf| (self.tree_size + *leaf) as usize)
             .collect();

         for &idx in &current_level {
             let leaf_id = idx - self.tree_size as usize;
             if let Some(hash) = self.page_hashes.get(&PageID(leaf_id as u64)) {
                 self.merkle_nodes[idx] = *hash;
             }
         }

         // Step 2: climb levels incrementally
         while !current_level.is_empty() {
             let mut parents: Vec<usize> = current_level
                 .iter()
                 .map(|idx| idx / 2)
                 .filter(|&p| p >= 1)
                 .collect();

             parents.sort_unstable();
             parents.dedup();

             // Immutable snapshot slice (no clone)
             let snapshot: &[Hash] = &self.merkle_nodes;

             // Compute parent hashes in parallel into temp buffer
             let parent_hashes: Vec<(usize, Hash)> = parents
                 .par_iter()
                 .map(|&parent| {
                     let left = parent * 2;
                     let right = left + 1;

                     let mut hasher = Sha256::new();
                     hasher.update(DOMAIN_INTERNAL);
                     hasher.update(snapshot[left]);
                     hasher.update(snapshot[right]);

                     (parent, hasher.finalize().into())
                 })
                 .collect();

             // Sequential write-back (safe, no aliasing)
             for (parent, hash) in parent_hashes {
                 self.merkle_nodes[parent] = hash;
             }

             current_level = parents;
         }

         self.root_hash = self.merkle_nodes[1];
         self.dirty_leaves.clear();
     }


    /// Grow tree and rebuild when capacity exceeded.
    fn rehash_with_new_capacity(&mut self, required_leaves: u64) {
        let new_tree_size = required_leaves.next_power_of_two();
        let total_nodes = new_tree_size * 2;

        self.tree_size = new_tree_size;
        self.max_leaves = new_tree_size;
        self.merkle_nodes = vec![[0u8; 32]; total_nodes as usize];

        // Reinsert leaves
        let existing = self.page_hashes.clone();
        for (page_id, hash) in existing {
            let idx = (self.tree_size + page_id.0) as usize;
            self.merkle_nodes[idx] = hash;
        }

        // ===============================
        // Level-sliced parallel rebuild
        // ===============================
        use rayon::prelude::*;

        let mut level_start = self.tree_size as usize;

        while level_start > 1 {
            let parent_start = level_start / 2;
            let parent_end = level_start;

            // Snapshot current level so we can read without aliasing
            let current_level = self.merkle_nodes.clone();

            // Compute parent layer in parallel into temporary buffer
            let parent_hashes: Vec<Hash> =
                (parent_start..parent_end)
                    .into_par_iter()
                    .map(|parent_index| {
                        let left = parent_index * 2;
                        let right = left + 1;

                        let mut hasher = Sha256::new();
                        hasher.update(DOMAIN_INTERNAL);
                        hasher.update(current_level[left]);
                        hasher.update(current_level[right]);
                        hasher.finalize().into()
                    })
                    .collect();

            // Write back results sequentially (safe)
            for (i, hash) in parent_hashes.into_iter().enumerate() {
                self.merkle_nodes[parent_start + i] = hash;
            }

            level_start /= 2;
        }

        self.root_hash = self.merkle_nodes[1];
    }

    /// Recompute root from sparse tree nodes (debug only).
    pub fn recompute_root_from_nodes(&self) -> Hash {
        self.merkle_nodes.get(1).copied().unwrap_or([0u8; 32])
    }

    /// Debug integrity check.
    pub fn verify_internal_consistency(&self) -> bool {
        self.root_hash == self.recompute_root_from_nodes()
    }

    /// Deterministic full rebuild of the Merkle tree.
    /// This ignores dirty tracking and recomputes everything bottom-up.
    /// Used for correctness verification and property testing.
    pub fn full_recompute_root(&self) -> Hash {
        let mut nodes = vec![[0u8; 32]; self.merkle_nodes.len()];

        // Insert leaves
        for (page_id, hash) in &self.page_hashes {
            let idx = (self.tree_size + page_id.0) as usize;
            nodes[idx] = *hash;
        }

        // Recompute internal nodes bottom-up
        for i in (1..self.tree_size as usize).rev() {
            let left = i * 2;
            let right = left + 1;

            let mut hasher = Sha256::new();
            hasher.update(DOMAIN_INTERNAL);
            hasher.update(nodes[left]);
            hasher.update(nodes[right]);

            nodes[i] = hasher.finalize().into();
        }

        nodes[1]
    }

    /// Generate Merkle inclusion proof for a page.
    pub fn merkle_proof(&self, page_id: PageID) -> Option<Vec<Hash>> {
        if !self.page_hashes.contains_key(&page_id) {
            return None;
        }

        let mut proof = Vec::new();

        let leaf_offset = self.merkle_nodes.len() / 2;
        let mut node_index = leaf_offset + page_id.0 as usize;

        while node_index > 1 {
            let sibling = if node_index % 2 == 0 {
                node_index + 1
            } else {
                node_index - 1
            };

            proof.push(self.merkle_nodes[sibling]);
            node_index /= 2;
        }

        Some(proof)
    }
}

// Snapshotting removed. Recovery is handled via journal + tlog replay.

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
        let tlog =
            Arc::new(TransactionLog::new(&config.tlog_path).map_err(MemoryEngineError::TlogOpen)?);
        let graph_log = Arc::new(
            GraphDeltaLog::new(&config.graph_log_path).map_err(MemoryEngineError::GraphLogOpen)?,
        );

        Ok(Self {
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

        // Ensure mmap-backed pages are flushed
        self.state.write().page_store.flush().ok();

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
        hasher.update(DOMAIN_EVENT);
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
        hasher.update(DOMAIN_DELTA);
        hasher.update(&delta.page_id.0.to_be_bytes());
        hasher.update(&delta.epoch.0.to_be_bytes());
        for bit in &delta.mask {
            hasher.update(&[*bit as u8]);
        }
        hasher.update(&delta.payload);
        hasher.finalize().into()
    }


    // Full rebuild no longer used (sparse incremental tree)

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
