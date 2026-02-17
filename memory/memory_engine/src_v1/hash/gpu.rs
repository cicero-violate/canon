use crate::hash::HashBackend;
use crate::primitives::Hash;

use sha2::{Digest, Sha256};

extern "C" {
    fn merkle_rebuild_kernel(
        tree: *mut u8,
        tree_size: u64,
    );
}

pub struct GpuBackend;

impl HashBackend for GpuBackend {
    fn hash_leaf(&self, data: &[u8]) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.finalize().into()
    }

    fn hash_internal(&self, left: Hash, right: Hash) -> Hash {
        let mut hasher = Sha256::new();
        hasher.update(left);
        hasher.update(right);
        hasher.finalize().into()
    }

    fn hash_internal_layer(&self, out: &mut [Hash], children: &[Hash]) {
        for (i, chunk) in children.chunks(2).enumerate() {
            out[i] = self.hash_internal(chunk[0], chunk[1]);
        }
    }

    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [Hash],
        tree_size: u64,
        _dirty_leaves: &[u64],
    ) {
        if tree_size == 0 {
            return;
        }

        unsafe {
            merkle_rebuild_kernel(
                nodes.as_mut_ptr() as *mut u8,
                tree_size,
            );
        }
    }
}

pub fn create_gpu_backend() -> Box<dyn HashBackend> {
    Box::new(GpuBackend)
}
