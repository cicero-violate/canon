use crate::primitives::Hash;

pub trait HashBackend: Send + Sync {
    fn hash_leaf(&self, data: &[u8]) -> Hash;

    fn hash_internal(&self, left: Hash, right: Hash) -> Hash;

    /// Hash N parents from 2N children.
    /// children layout: [L0, R0, L1, R1, ...]
    fn hash_internal_layer(&self, out: &mut [Hash], children: &[Hash]);

    /// Fully rebuild dirty Merkle levels on GPU.
    /// Performs multi-level batched hashing in a single pipeline.
    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [Hash],
        tree_size: u64,
        dirty_leaves: &[u64],
    );
}

pub mod gpu;
