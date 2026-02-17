use crate::primitives::Hash;

pub trait HashBackend: Send + Sync {
    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [Hash],
        tree_size: u64,
        pages_ptr: *const u8,
    );
}

pub mod gpu;
