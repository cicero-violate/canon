#[cfg(feature = "cuda")]
use crate::hash::cuda_ffi::*;
use core::ffi::c_void;
use std::ptr::null_mut;

const PAGE_SIZE: usize = 4096;

// CUDA FFI centralized in hash::cuda_ffi

#[derive(Debug)]
pub struct PageStore {
    ptr: *mut u8,
    capacity: usize,
}

// Unified/device memory is globally accessible.
// Synchronization is enforced at higher layers (RwLock in CanonicalState).
unsafe impl Send for PageStore {}
unsafe impl Sync for PageStore {}

impl PageStore {
    pub fn in_memory() -> Self {
        let capacity = PAGE_SIZE * 1024;

        #[cfg(feature = "cuda")]
        unsafe {
            let mut raw: *mut c_void = null_mut();
            let err = cudaMallocManaged(&mut raw as *mut *mut c_void, capacity, 0);
            if err == 0 && !raw.is_null() {
                let ptr = raw as *mut u8;
                core::ptr::write_bytes(ptr, 0, capacity);
                return Self { ptr, capacity };
            }
        }

        // Deterministic heap fallback when CUDA runtime unavailable
        let mut vec = vec![0u8; capacity];
        let ptr = vec.as_mut_ptr();
        core::mem::forget(vec);
        Self { ptr, capacity }
    }

    pub fn write_page(&mut self, page_id: u64, data: &[u8]) {
        let offset = page_id as usize * PAGE_SIZE;
        let len = data.len().min(PAGE_SIZE);
        if offset + PAGE_SIZE > self.capacity {
            self.grow((page_id as usize + 1) * PAGE_SIZE);
        }
        unsafe {
            let page = std::slice::from_raw_parts_mut(self.ptr.add(offset), PAGE_SIZE);
            page[..len].copy_from_slice(&data[..len]);
        }
    }

    #[allow(dead_code)]
    pub fn read_page(&self, page_id: u64) -> &[u8] {
        let offset = page_id as usize * PAGE_SIZE;
        assert!(offset + PAGE_SIZE <= self.capacity, "capacity exceeded");
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), PAGE_SIZE) }
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    pub fn as_device_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity
    }

    fn grow(&mut self, required: usize) {
        let new_capacity = required.next_power_of_two();
        #[cfg(feature = "cuda")]
        unsafe {
            let mut raw: *mut c_void = null_mut();
            let err = cudaMallocManaged(&mut raw as *mut *mut c_void, new_capacity, 0);
            if err == 0 && !raw.is_null() {
                let new_ptr = raw as *mut u8;
                core::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.capacity);
                cudaFree(self.ptr as *mut c_void);
                self.ptr = new_ptr;
                self.capacity = new_capacity;
                return;
            }
        }

        // Heap fallback grow
        let mut vec = vec![0u8; new_capacity];
        unsafe {
            core::ptr::copy_nonoverlapping(self.ptr, vec.as_mut_ptr(), self.capacity);
        }
        let new_ptr = vec.as_mut_ptr();
        core::mem::forget(vec);
        self.ptr = new_ptr;
        self.capacity = new_capacity;
    }
}

impl Drop for PageStore {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            cudaFree(self.ptr as *mut c_void);
        }
    }
}
