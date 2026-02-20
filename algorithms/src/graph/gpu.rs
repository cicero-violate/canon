//! FFI bridge to graph/bfs.cu via unified libgpu.a
//!
//! Variables:
//!   row_ptr : *const i32  — CSR row pointer, length V+1
//!   col_idx : *const i32  — CSR column indices, length E
//!   V, E    : i32         — vertex and edge counts
//!   source  : i32         — BFS start vertex
//!   level   : *mut i32    — output BFS levels, length V (-1 = unreachable)
//!
//! Equation:
//!   level[v] = d  <=>  shortest path source -> v has length d

#[cfg(feature = "cuda")]
extern "C" {
    pub fn gpu_bfs(
        row_ptr:   *const i32,
        col_idx:   *const i32,
        v:         i32,
        e:         i32,
        source:    i32,
        level_out: *mut i32,
    );
}

/// Safe wrapper: runs GPU BFS, returns level vector.
/// level[i] == -1 means vertex i is unreachable from source.
#[cfg(feature = "cuda")]
pub fn bfs_gpu(row_ptr: &[i32], col_idx: &[i32], source: usize) -> Vec<i32> {
    let v = (row_ptr.len() - 1) as i32;
    let e = col_idx.len() as i32;
    let mut level = vec![-1i32; v as usize];
    unsafe {
        gpu_bfs(
            row_ptr.as_ptr(), col_idx.as_ptr(),
            v, e, source as i32,
            level.as_mut_ptr(),
        );
    }
    level
}
