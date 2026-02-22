use crate::hash::{cpu, HashBackend};
use crate::primitives::StateHash;
#[cfg(feature = "cuda")]
use crate::hash::cuda_ffi::*;
#[cfg(feature = "cuda")]
use algorithms::cryptography::merkle_tree_gpu::{merkle_build_gpu, HASH_SIZE, PAGE_SIZE};
#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::Once;
pub struct GpuBackend {
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
        let _ = ptr;
    }
}
impl HashBackend for GpuBackend {
    fn rebuild_merkle_tree(
        &self,
        nodes: &mut [StateHash],
        tree_size: u64,
        pages_ptr: *const u8,
    ) {
        if tree_size == 0 {
            return;
        }
        #[cfg(feature = "cuda")]
        if self.cuda {
            let tree_size_usize = tree_size as usize;
            if !tree_size.is_power_of_two() {
                cpu::rebuild_merkle_tree(nodes, tree_size, pages_ptr);
                return;
            }
            let pages_len = tree_size_usize * PAGE_SIZE;
            let pages = unsafe { std::slice::from_raw_parts(pages_ptr, pages_len) };
            let tree = merkle_build_gpu(pages);
            let expected_nodes = 2 * tree_size_usize;
            if nodes.len() < expected_nodes {
                cpu::rebuild_merkle_tree(nodes, tree_size, pages_ptr);
                return;
            }
            for (i, node) in nodes.iter_mut().take(expected_nodes).enumerate() {
                let base = i * HASH_SIZE;
                node.copy_from_slice(&tree[base..base + HASH_SIZE]);
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
            return Box::new(GpuBackend { cuda: true });
        } else {
            debug_cuda("CUDA detection failed; using CPU fallback");
        }
        return Box::new(GpuBackend { cuda: false });
    }
    #[cfg(not(feature = "cuda"))] Box::new(GpuBackend { cuda: false })
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
        debug_cuda(&format!("cudaGetDeviceCount -> status={} count={}", status, count));
        status == 0 && count > 0
    }
}
#[cfg(feature = "cuda")]
pub fn gpu_available() -> bool {
    unsafe {
        CUDA_ONCE
            .call_once(|| {
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
        eprintln!("[database][cuda] {message}");
    }
}
