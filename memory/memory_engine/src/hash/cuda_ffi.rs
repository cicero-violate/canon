use core::ffi::c_void;

extern "C" {
    pub fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;

    pub fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    pub fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    pub fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

    pub fn launch_hash_leaves_and_rebuild(tree: *mut u8, tree_size: u64, pages: *const u8);
}
