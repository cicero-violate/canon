use crate::hash::{cpu, HashBackend};
use crate::primitives::Hash;

// ── CUDA-only types and impls ────────────────────────────────────────────────

#[cfg(feature = "cuda")]
use crate::hash::cuda_ffi::*;
#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Once;

#[cfg(feature = "cuda")]
pub struct CudaContext {
    pub stream: *mut c_void,
}

#[cfg(feature = "cuda")]
unsafe impl Send for CudaContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaContext {}

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.stream.is_null() {
                cudaStreamDestroy(self.stream);
            }
        }
    }
}

// ── GpuBackend ───────────────────────────────────────────────────────────────

pub struct GpuBackend {
    #[cfg(feature = "cuda")]
    _ctx: Option<CudaContext>,
    cuda: bool,
}

impl GpuBackend {
    pub fn allocate_tree_bytes(bytes: usize) -> *mut u8 {
        #[cfg(feature = "cuda")]
        if gpu_available() {
            unsafe {
                let mut raw: *mut c_void = core::ptr::null_mut();
                let err = cudaMallocManaged(&mut raw as *mut *mut c_void, bytes, 1);
                if err == 0 && !raw.is_null() {
                    let ptr = raw as *mut u8;
                    core::ptr::write_bytes(ptr, 0, bytes);
                    return ptr;
                }
            }
        }
        Self::heap_allocate(bytes)
    }

    fn heap_allocate(bytes: usize) -> *mut u8 {
        let mut v = vec![0u8; bytes];
        let ptr = v.as_mut_ptr();
        core::mem::forget(v);
        ptr
    }

    pub fn free_tree(ptr: *mut u8) {
        #[cfg(feature = "cuda")]
        unsafe {
            let err = cudaFree(ptr as *mut c_void);
            if err == 0 {
                return;
            }
        }
        // heap fallback: reconstruct and drop — we don't have size so just leak
        let _ = ptr;
    }
}

impl HashBackend for GpuBackend {
    fn rebuild_merkle_tree(&self, nodes: &mut [Hash], tree_size: u64, pages_ptr: *const u8) {
        if tree_size == 0 {
            return;
        }
        #[cfg(feature = "cuda")]
        if self.cuda {
            unsafe {
                launch_hash_leaves_and_rebuild(
                    nodes.as_mut_ptr() as *mut u8,
                    tree_size,
                    pages_ptr,
                );
            }
            return;
        }
        cpu::rebuild_merkle_tree(nodes, tree_size, pages_ptr);
    }
}

pub fn create_gpu_backend() -> Box<dyn HashBackend> {
    #[cfg(feature = "cuda")]
    {
        if gpu_available() {
            debug_cuda("CUDA reported as available; attempting to initialize context");
            let ctx = CudaContext::new();
            if !ctx.stream.is_null() {
                debug_cuda("CUDA context initialized successfully");
                return Box::new(GpuBackend {
                    _ctx: Some(ctx),
                    cuda: true,
                });
            }
            debug_cuda("CUDA context missing stream; disabling GPU path");
        } else {
            debug_cuda("CUDA detection failed; using CPU fallback");
        }
        return Box::new(GpuBackend {
            _ctx: None,
            cuda: false,
        });
    }
    #[cfg(not(feature = "cuda"))]
    Box::new(GpuBackend { cuda: false })
}

#[cfg(feature = "cuda")]
static CUDA_ONCE: Once = Once::new();
#[cfg(feature = "cuda")]
static mut CUDA_PRESENT: bool = false;

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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

#[cfg(not(feature = "cuda"))]
pub fn gpu_available() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn debug_cuda(message: &str) {
    if std::env::var_os("MEMORY_ENGINE_DEBUG_CUDA").is_some() {
        eprintln!("[memory_engine][cuda] {message}");
    }
}
