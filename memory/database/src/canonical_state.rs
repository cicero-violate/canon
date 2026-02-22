use crate::hash::HashBackend;
use crate::{delta::DeltaError, page_store::PageStore, primitives::StateHash};
use std::io::{Read, Write};
use std::path::Path;
pub struct MerkleState {
    pub(crate) root_hash: StateHash,
    pub(crate) page_store: PageStore,
    pub(crate) device_tree_ptr: *mut u8,
    pub(crate) device_tree_bytes: usize,
    pub(crate) max_leaves: u64,
    pub(crate) tree_size: u64,
    pub(crate) backend: Box<dyn HashBackend>,
}
unsafe impl Send for MerkleState {}
unsafe impl Sync for MerkleState {}
impl Drop for MerkleState {
    fn drop(&mut self) {
        if !self.device_tree_ptr.is_null() {
            crate::hash::gpu::GpuBackend::free_tree(self.device_tree_ptr);
        }
    }
}
impl MerkleState {
    pub fn new_empty(backend: Box<dyn HashBackend>) -> Self {
        Self::with_capacity(1024, backend)
    }
    pub fn with_capacity(max_leaves: u64, backend: Box<dyn HashBackend>) -> Self {
        let tree_size = max_leaves.next_power_of_two();
        let total_nodes = tree_size * 2;
        let total_bytes = total_nodes as usize * 32;
        let device_tree_ptr = crate::hash::gpu::GpuBackend::allocate_tree_bytes(total_bytes);
        let root_hash = [0u8; 32];
        Self { root_hash, page_store: PageStore::in_memory(), device_tree_ptr, device_tree_bytes: total_bytes, max_leaves: tree_size, tree_size, backend }
    }
    pub fn apply_delta(&mut self, delta: &crate::delta::Delta) -> Result<(), DeltaError> {
        crate::delta::validate_delta(delta)?;
        if delta.page_id.0 >= self.max_leaves {
            self.rehash_with_new_capacity(delta.page_id.0 + 1);
        }
        let dense = delta.to_dense();
        self.page_store.write_page(delta.page_id.0, &dense);
        let nodes = unsafe { std::slice::from_raw_parts_mut(self.device_tree_ptr as *mut StateHash, (self.tree_size * 2) as usize) };
        self.backend.rebuild_merkle_tree(nodes, self.tree_size, self.page_store.as_device_ptr());
        self.root_hash = nodes[1];
        Ok(())
    }
    /// Batch apply without intermediate rehash.
    pub fn apply_deltas_batch(&mut self, deltas: &[crate::delta::Delta]) -> Result<(), DeltaError> {
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
        for delta in deltas {
            let dense = delta.to_dense();
            self.page_store.write_page(delta.page_id.0, &dense);
        }
        let nodes = unsafe { std::slice::from_raw_parts_mut(self.device_tree_ptr as *mut StateHash, (self.tree_size * 2) as usize) };
        self.backend.rebuild_merkle_tree(nodes, self.tree_size, self.page_store.as_device_ptr());
        self.root_hash = nodes[1];
        Ok(())
    }
    pub fn root_hash(&self) -> StateHash {
        self.root_hash
    }
    pub fn read_page(&self, page_id: u64) -> &[u8] {
        self.page_store.read_page(page_id)
    }
    pub fn checkpoint<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&self.tree_size.to_le_bytes())?;
        file.write_all(&self.root_hash)?;
        let total = self.page_store.capacity_bytes();
        let slice = unsafe { std::slice::from_raw_parts(self.page_store.as_device_ptr(), total) };
        file.write_all(slice)?;
        Ok(())
    }
    pub fn restore_from_checkpoint<P: AsRef<Path>>(path: P, backend: Box<dyn HashBackend>) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let mut size_buf = [0u8; 8];
        file.read_exact(&mut size_buf)?;
        let tree_size = u64::from_le_bytes(size_buf);
        let mut root = [0u8; 32];
        file.read_exact(&mut root)?;
        let mut state = Self::with_capacity(tree_size, backend);
        let total = state.page_store.capacity_bytes();
        let slice = unsafe { std::slice::from_raw_parts_mut(state.page_store.as_device_ptr() as *mut u8, total) };
        file.read_exact(slice)?;
        state.backend.rebuild_merkle_tree(
            unsafe { std::slice::from_raw_parts_mut(state.device_tree_ptr as *mut StateHash, (state.tree_size * 2) as usize) },
            state.tree_size,
            state.page_store.as_device_ptr(),
        );
        state.root_hash = root;
        Ok(state)
    }
    /// Deterministic full Merkle recompute (CPU validation path).
    #[cfg(debug_assertions)]
    pub fn full_recompute_root(&self) -> StateHash {
        let total_nodes = (self.tree_size * 2) as usize;
        let mut nodes = vec![[0u8; 32]; total_nodes];
        self.backend.rebuild_merkle_tree(&mut nodes, self.tree_size, self.page_store.as_device_ptr());
        nodes[1]
    }
    fn rehash_with_new_capacity(&mut self, required_leaves: u64) {
        let new_tree_size = required_leaves.next_power_of_two();
        let total_nodes = new_tree_size * 2;
        let total_bytes = total_nodes as usize * 32;
        self.tree_size = new_tree_size;
        self.max_leaves = new_tree_size;
        crate::hash::gpu::GpuBackend::free_tree(self.device_tree_ptr);
        self.device_tree_ptr = crate::hash::gpu::GpuBackend::allocate_tree_bytes(total_bytes);
        self.device_tree_bytes = total_bytes;
        let nodes = unsafe { std::slice::from_raw_parts_mut(self.device_tree_ptr as *mut StateHash, total_nodes as usize) };
        self.backend.rebuild_merkle_tree(nodes, self.tree_size, self.page_store.as_device_ptr());
        unsafe {
            let root_ptr = self.device_tree_ptr.add(32);
            std::ptr::copy_nonoverlapping(root_ptr, self.root_hash.as_mut_ptr(), 32);
        }
    }
}
