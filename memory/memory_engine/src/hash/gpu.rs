use crate::hash::{cpu, cuda_ffi::*, HashBackend};
use crate::primitives::Hash;
use core::ffi::c_void;
use std::sync::Once;

pub struct GpuBackend {
    _ctx: Option<CudaContext>,
    cuda: bool,
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
            if cudaStreamCreate(&mut stream as *mut *mut c_void) != 0 {
                debug_cuda("cudaStreamCreate failed; falling back to CPU mode");
                stream = core::ptr::null_mut();
            } else {
                debug_cuda("cudaStreamCreate succeeded");
            }
            Self { stream }
        }
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

impl GpuBackend {
    pub fn allocate_tree_bytes(bytes: usize) -> *mut u8 {
        if !gpu_available() {
            return Self::heap_allocate(bytes);
        }

        unsafe {
            let mut raw: *mut c_void = core::ptr::null_mut();
            let err = cudaMallocManaged(&mut raw as *mut *mut c_void, bytes, 1);

            if err != 0 || raw.is_null() {
                return Self::heap_allocate(bytes);
            }

            let ptr = raw as *mut u8;
            core::ptr::write_bytes(ptr, 0, bytes);
            ptr
        }
    }

    fn heap_allocate(bytes: usize) -> *mut u8 {
        let mut v = vec![0u8; bytes];
        let ptr = v.as_mut_ptr();
        core::mem::forget(v);
        ptr
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
    fn rebuild_merkle_tree(&self, nodes: &mut [Hash], tree_size: u64, pages_ptr: *const u8) {
        if tree_size == 0 {
            return;
        }

        if !self.cuda {
            debug_cuda("GPU unavailable, executing CPU Merkle rebuild");
            cpu::rebuild_merkle_tree(nodes, tree_size, pages_ptr);
            return;
        }

        unsafe {
            launch_hash_leaves_and_rebuild(nodes.as_mut_ptr() as *mut u8, tree_size, pages_ptr);
        }
    }
}

pub fn create_gpu_backend() -> Box<dyn HashBackend> {
    let cuda_supported = gpu_available();
    let (ctx, cuda) = if cuda_supported {
        debug_cuda("CUDA reported as available; attempting to initialize context");
        let ctx = CudaContext::new();
        if ctx.stream.is_null() {
            debug_cuda("CUDA context missing stream; disabling GPU path");
            (None, false)
        } else {
            debug_cuda("CUDA context initialized successfully");
            (Some(ctx), true)
        }
    } else {
        debug_cuda("CUDA detection failed; using CPU fallback");
        (None, false)
    };

    Box::new(GpuBackend { _ctx: ctx, cuda })
}

static CUDA_ONCE: Once = Once::new();
static mut CUDA_PRESENT: bool = false;

fn detect_cuda() -> bool {
    unsafe {
        let mut count = 0i32;
        let status = cudaGetDeviceCount(&mut count as *mut i32);
        debug_cuda(&format!(
            "cudaGetDeviceCount -> status={} count={}",
            status, count
        ));
        status == 0 && count > 0
    }
}

pub fn gpu_available() -> bool {
    unsafe {
        CUDA_ONCE.call_once(|| {
            CUDA_PRESENT = detect_cuda();
            let present = CUDA_PRESENT;
            debug_cuda(&format!("gpu_available cached result: {}", present));
        });
        CUDA_PRESENT
    }
}

fn debug_cuda(message: &str) {
    if std::env::var_os("MEMORY_ENGINE_DEBUG_CUDA").is_some() {
        eprintln!("[memory_engine][cuda] {message}");
    }
}
