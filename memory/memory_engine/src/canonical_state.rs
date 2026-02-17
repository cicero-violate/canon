
use crate::{
    delta::DeltaError,
    page_store::PageStore,
    primitives::Hash,
};
use crate::hash::HashBackend;

pub struct CanonicalState {
    pub(crate) root_hash: Hash,
    pub(crate) page_store: PageStore,
    pub(crate) merkle_nodes: Vec<Hash>,
    pub(crate) max_leaves: u64,
    pub(crate) tree_size: u64,
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
            merkle_nodes,
            max_leaves: tree_size,
            tree_size,
            backend,
        }
    }

    pub fn apply_delta(
        &mut self,
        delta: &crate::delta::Delta,
    ) -> Result<(), DeltaError> {
        // STRICT: single-delta path is write-only.
        // No per-delta Merkle rebuild.
        crate::delta::validate_delta(delta)?;

        if delta.page_id.0 >= self.max_leaves {
            self.rehash_with_new_capacity(delta.page_id.0 + 1);
        }

        let dense = delta.to_dense();
        self.page_store.write_page(delta.page_id.0, &dense);
        Ok(())
    }

    /// Batch apply without intermediate rehash.
    pub fn apply_deltas_batch(
        &mut self,
        deltas: &[crate::delta::Delta],
    ) -> Result<(), DeltaError> {
        // capacity pre-scan
        let mut required = self.max_leaves;
        for delta in deltas {
            crate::delta::validate_delta(delta)?;
            if delta.page_id.0 >= required {
                required = delta.page_id.0 + 1;
            }
        }

        if required > self.max_leaves {
            self.rehash_with_new_capacity(required);
        }

        // device writes only
        for delta in deltas {
            let dense = delta.to_dense();
            self.page_store.write_page(delta.page_id.0, &dense);
        }

        // single GPU rebuild
        self.backend.rebuild_merkle_tree(
            &mut self.merkle_nodes,
            self.tree_size,
            self.page_store.as_device_ptr(),
        );

        self.root_hash = self.merkle_nodes[1];
        Ok(())
    }

    pub fn root_hash(&self) -> Hash {
        self.root_hash
    }

    /// Deterministic full Merkle recompute (CPU validation path).
    #[cfg(debug_assertions)]
    pub fn full_recompute_root(&self) -> Hash {
        let mut nodes = self.merkle_nodes.clone();
        self.backend.rebuild_merkle_tree(
            &mut nodes,
            self.tree_size,
            self.page_store.as_device_ptr(),
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

        // No incremental page hash restore — full GPU rebuild only

        self.backend.rebuild_merkle_tree(
            &mut self.merkle_nodes,
            self.tree_size,
            self.page_store.as_device_ptr(),
        );

        self.root_hash = self.merkle_nodes[1];
    }
}
