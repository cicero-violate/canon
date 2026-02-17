use crate::hash::HashBackend;
use crate::primitives::Hash;

extern "C" {
    fn launch_hash_leaves_and_rebuild(
        tree: *mut u8,
        tree_size: u64,
        pages: *const u8,
    );
}

pub struct GpuBackend;

impl HashBackend for GpuBackend {
    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [Hash],
        tree_size: u64,
        pages_ptr: *const u8,
    ) {
        if tree_size == 0 {
            return;
        }

        unsafe {
            launch_hash_leaves_and_rebuild(
                nodes.as_mut_ptr() as *mut u8,
                tree_size,
                pages_ptr,
            );
        }
    }
}

pub fn create_gpu_backend() -> Box<dyn HashBackend> {
    Box::new(GpuBackend)
}
