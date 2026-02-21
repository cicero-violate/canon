use crate::primitives::StateHash;
pub trait HashBackend: Send + Sync {
    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [StateHash],
        tree_size: u64,
        pages_ptr: *const u8,
    );
}
pub(crate) mod cpu;
pub mod cuda_ffi;
pub mod gpu;
