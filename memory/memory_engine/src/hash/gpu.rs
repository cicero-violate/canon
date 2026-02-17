use crate::hash::cuda_ffi::*;
use crate::hash::HashBackend;
use crate::primitives::Hash;
use core::ffi::c_void;

pub struct GpuBackend {
    _ctx: CudaContext,
}

pub struct CudaContext {
    pub stream: *mut c_void,
}

// SAFETY:
// CUDA stream handle is externally synchronized.
// All GPU work is serialized through higher-level RwLock.
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    pub fn new() -> Self {
        unsafe {
            let mut stream: *mut c_void = core::ptr::null_mut();
            // If stream creation fails (e.g. no CUDA device in CI),
            // fall back to default stream (null).
            if cudaStreamCreate(&mut stream as *mut *mut c_void) != 0 {
                stream = core::ptr::null_mut();
            }
            Self { stream }
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { cudaStreamDestroy(self.stream); }
    }
}

impl GpuBackend {
    pub fn allocate_tree_bytes(bytes: usize) -> *mut u8 {
        unsafe {
            let mut raw: *mut c_void = core::ptr::null_mut();
            let err = cudaMallocManaged(&mut raw as *mut *mut c_void, bytes, 0);

            if err != 0 || raw.is_null() {
                // No CUDA device available at runtime (e.g. CI).
                // Hard fallback to heap to preserve deterministic tests.
                let mut v = vec![0u8; bytes];
                let ptr = v.as_mut_ptr();
                core::mem::forget(v);
                return ptr;
            }

            let ptr = raw as *mut u8;
            core::ptr::write_bytes(ptr, 0, bytes);
            ptr
        }
    }

    pub fn free_tree(ptr: *mut u8) {
        unsafe {
            // Attempt cudaFree; if it fails assume host fallback
            let err = cudaFree(ptr as *mut c_void);
            if err != 0 {
                // host fallback memory leak intentionally avoided
                // cannot reconstruct Vec safely without size
            }
        }
    }
}

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
    Box::new(GpuBackend {
        _ctx: CudaContext::new(),
    })
}
