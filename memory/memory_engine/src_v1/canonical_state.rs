use std::collections::BTreeMap;

use crate::{
    delta::DeltaError,
    page_store::PageStore,
    primitives::{Hash, PageID},
};
use crate::hash::HashBackend;

pub struct CanonicalState {
    pub(crate) root_hash: Hash,
    pub(crate) page_store: PageStore,
    pub(crate) page_hashes: BTreeMap<PageID, Hash>,
    pub(crate) merkle_nodes: Vec<Hash>,
    pub(crate) max_leaves: u64,
    pub(crate) tree_size: u64,
    pub(crate) dirty_leaves: Vec<u64>,
    pub(crate) backend: Box<dyn HashBackend>,
}

impl CanonicalState {
    pub fn new_empty(backend: Box<dyn HashBackend>) -> Self {
        Self::with_capacity(1024, backend)
    }

    pub fn with_capacity(
        max_leaves: u64,
        backend: Box<dyn HashBackend>,
    ) -> Self {
        let tree_size = max_leaves.next_power_of_two();
        let total_nodes = tree_size * 2;

        // GPU-only tree: start zeroed, GPU will build when needed
        let merkle_nodes = vec![[0u8; 32]; total_nodes as usize];

        let root_hash = merkle_nodes[1];

        Self {
            root_hash,
            page_store: PageStore::in_memory(),
            page_hashes: BTreeMap::new(),
            merkle_nodes,
            max_leaves: tree_size,
            tree_size,
            dirty_leaves: Vec::new(),
            backend,
        }
    }

    pub fn apply_delta(
        &mut self,
        delta: &crate::delta::Delta,
    ) -> Result<(), DeltaError> {
        crate::delta::validate_delta(delta)?;

        let dense = delta.to_dense();

        // ensure capacity first
        if delta.page_id.0 >= self.max_leaves {
            self.rehash_with_new_capacity(delta.page_id.0 + 1);
        }

        // read existing page from PageStore
        let mut page_buf = self
            .page_store
            .read_page(delta.page_id.0)
            .to_vec();

        if dense.len() > page_buf.len() {
            page_buf.resize(dense.len(), 0);
        }

        for (i, &bit) in delta.mask.iter().enumerate() {
            if bit {
                page_buf[i] = dense[i];
            }
        }

        // write mutated page back
        self.page_store
            .write_page(delta.page_id.0, &page_buf);

        // zero-copy hash directly from PageStore
        let page_hash =
            self.backend
                .hash_leaf(self.page_store.read_page(delta.page_id.0));

        self.page_hashes.insert(delta.page_id, page_hash);

        let leaf_index = (self.tree_size + delta.page_id.0) as usize;
        self.merkle_nodes[leaf_index] = page_hash;

        if !self.dirty_leaves.contains(&delta.page_id.0) {
            self.dirty_leaves.push(delta.page_id.0);
        }

        self.backend.rebuild_merkle_tree(
            &mut self.merkle_nodes,
            self.tree_size,
            &self.dirty_leaves,
        );

        self.root_hash = self.merkle_nodes[1];

        self.dirty_leaves.clear();

        Ok(())
    }

    /// Batch apply without intermediate rehash.
    pub fn apply_deltas_batch(
        &mut self,
        deltas: &[crate::delta::Delta],
    ) -> Result<(), DeltaError> {
        for delta in deltas {
            crate::delta::validate_delta(delta)?;

            let dense = delta.to_dense();

            self.page_store.write_page(delta.page_id.0, &dense);

            let page_hash = self.backend.hash_leaf(&dense);

            if delta.page_id.0 >= self.max_leaves {
                self.rehash_with_new_capacity(delta.page_id.0 + 1);
            }

            self.page_hashes.insert(delta.page_id, page_hash);

            let leaf_index = (self.tree_size + delta.page_id.0) as usize;
            self.merkle_nodes[leaf_index] = page_hash;

            if !self.dirty_leaves.contains(&delta.page_id.0) {
                self.dirty_leaves.push(delta.page_id.0);
            }
        }

        self.backend.rebuild_merkle_tree(
            &mut self.merkle_nodes,
            self.tree_size,
            &self.dirty_leaves,
        );

        self.root_hash = self.merkle_nodes[1];

        self.dirty_leaves.clear();

        Ok(())
    }

    pub fn root_hash(&self) -> Hash {
        self.root_hash
    }

    /// Deterministic full rebuild of the Merkle tree.
    /// Used for correctness verification and property testing.
    pub fn full_recompute_root(&self) -> Hash {

        let mut nodes = self.merkle_nodes.clone();

        self.backend.rebuild_merkle_tree(
            &mut nodes,
            self.tree_size,
            &self.dirty_leaves,
        );

        nodes[1]
    }

    fn rehash_with_new_capacity(&mut self, required_leaves: u64) {
        let new_tree_size = required_leaves.next_power_of_two();
        let total_nodes = new_tree_size * 2;

        self.tree_size = new_tree_size;
        self.max_leaves = new_tree_size;

        let new_nodes = vec![[0u8; 32]; total_nodes as usize];

        self.merkle_nodes = new_nodes;

        for (page_id, hash) in self.page_hashes.clone() {
            let idx = (self.tree_size + page_id.0) as usize;
            self.merkle_nodes[idx] = hash;
        }

        self.backend.rebuild_merkle_tree(
            &mut self.merkle_nodes,
            self.tree_size,
            &self.dirty_leaves,
        );

        self.root_hash = self.merkle_nodes[1];

        self.dirty_leaves.clear();
    }
}
