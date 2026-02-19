//! Unified (zero-copy) memory allocator.
//! Wraps cudaMallocManaged so CSR arrays are accessible from both
//! CPU and GPU without any explicit transfers.
//! Falls back to heap allocation when CUDA is unavailable or the
//! cuda feature is not enabled.

#[cfg(feature = "cuda")]
use core::ffi::c_void;
#[cfg(feature = "cuda")]
use std::ptr::null_mut;

#[cfg(feature = "cuda")]
extern "C" {
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

/// Is a CUDA device present and usable?
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe {
            let mut count = 0i32;
            cudaGetDeviceCount(&mut count as *mut i32) == 0 && count > 0
        }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Synchronize the CUDA device (no-op without cuda feature).
pub fn device_sync() {
    #[cfg(feature = "cuda")]
    unsafe { cudaDeviceSynchronize(); }
}

/// A contiguous buffer in CUDA unified memory (or heap fallback).
/// Zero-copy: CPU reads/writes and GPU kernels share the same physical pages.
pub struct UnifiedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
    on_gpu: bool,
}

unsafe impl<T: Send> Send for UnifiedVec<T> {}
unsafe impl<T: Sync> Sync for UnifiedVec<T> {}

impl<T: Copy + Default> UnifiedVec<T> {
    pub fn with_capacity(cap: usize) -> Self {
        let byte_size = cap * std::mem::size_of::<T>();
        #[cfg(feature = "cuda")]
        if byte_size > 0 && cuda_available() {
            unsafe {
                let mut raw: *mut c_void = null_mut();
                if cudaMallocManaged(&mut raw as *mut *mut c_void, byte_size, 1) == 0
                    && !raw.is_null()
                {
                    let ptr = raw as *mut T;
                    std::ptr::write_bytes(ptr, 0, cap);
                    return Self { ptr, len: 0, cap, on_gpu: true };
                }
            }
        }
        // Heap fallback
        let mut v = vec![T::default(); cap];
        let ptr = v.as_mut_ptr();
        std::mem::forget(v);
        Self { ptr, len: 0, cap, on_gpu: false }
    }

    pub fn push(&mut self, val: T) {
        assert!(self.len < self.cap, "UnifiedVec capacity exceeded");
        unsafe { self.ptr.add(self.len).write(val); }
        self.len += 1;
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const T { self.ptr as *const T }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn on_gpu(&self) -> bool { self.on_gpu }

    /// Fill entire capacity with a value (useful for visited/dist arrays).
    pub fn fill(&mut self, val: T) {
        unsafe {
            for i in 0..self.cap {
                self.ptr.add(i).write(val);
            }
        }
        self.len = self.cap;
    }
}

impl<T> Drop for UnifiedVec<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() { return; }
        if self.on_gpu {
            #[cfg(feature = "cuda")]
            unsafe { cudaFree(self.ptr as *mut c_void); }
        } else {
            unsafe {
                Vec::from_raw_parts(self.ptr, self.len, self.cap);
            }
        }
    }
}
