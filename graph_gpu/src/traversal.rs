//! Graph traversal — BFS with GPU acceleration when available,
//! CPU fallback otherwise. Same API either way.
use crate::csr::{CsrGraph, EdgeKind};
use crate::unified::{cuda_available, device_sync, UnifiedVec};

/// Result of a BFS traversal.
pub struct BfsResult {
    /// Distance from source to each node (-1 = unreachable).
    /// Indexed by internal node index.
    pub dist: Vec<i32>,
    /// Maximum distance reached (diameter of reachable subgraph).
    pub max_dist: i32,
}

impl BfsResult {
    /// Reachable nodes as internal indices.
    pub fn reachable(&self) -> Vec<u32> {
        self.dist
            .iter()
            .enumerate()
            .filter(|(_, &d)| d >= 0)
            .map(|(i, _)| i as u32)
            .collect()
    }
}

/// BFS from `source` (internal index), optionally filtering by edge kind.
/// Uses GPU kernel when CUDA is available, CPU otherwise.
pub fn bfs(graph: &CsrGraph, source: u32, edge_filter: Option<EdgeKind>) -> BfsResult {
    if cuda_available() {
        bfs_gpu(graph, source, edge_filter)
    } else {
        bfs_cpu(graph, source, edge_filter)
    }
}

/// CPU BFS — standard queue-based.
fn bfs_cpu(graph: &CsrGraph, source: u32, edge_filter: Option<EdgeKind>) -> BfsResult {
    let n = graph.n_nodes;
    let mut dist = vec![-1i32; n];
    dist[source as usize] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(source);
    let mut max_dist = 0;

    while let Some(u) = queue.pop_front() {
        let d = dist[u as usize];
        let neighbors = graph.neighbors(u);
        let kinds     = graph.neighbor_kinds(u);
        for (&v, &k) in neighbors.iter().zip(kinds.iter()) {
            if let Some(filter) = edge_filter {
                if k != filter as u8 { continue; }
            }
            if dist[v as usize] == -1 {
                dist[v as usize] = d + 1;
                if d + 1 > max_dist { max_dist = d + 1; }
                queue.push_back(v);
            }
        }
    }
    BfsResult { dist, max_dist }
}

/// GPU BFS — frontier-based level-synchronous parallel BFS.
/// Uses unified memory so no transfers needed.
fn bfs_gpu(graph: &CsrGraph, source: u32, edge_filter: Option<EdgeKind>) -> BfsResult {
    let n = graph.n_nodes;

    // All in unified memory — GPU kernel reads row_offsets/col_indices directly.
    let mut dist_buf     = UnifiedVec::<i32>::with_capacity(n);
    let mut frontier_buf = UnifiedVec::<u32>::with_capacity(n);
    let mut next_buf     = UnifiedVec::<u32>::with_capacity(n);
    let mut next_size    = UnifiedVec::<u32>::with_capacity(1);

    dist_buf.fill(-1i32);
    frontier_buf.fill(0u32);
    next_buf.fill(0u32);
    next_size.fill(0u32);

    // Initialise source
    dist_buf.as_mut_slice()[source as usize] = 0;
    frontier_buf.as_mut_slice()[0] = source;

    let filter_byte: u8 = edge_filter.map(|k| k as u8).unwrap_or(255);
    let mut frontier_size = 1u32;
    let mut current_dist  = 0i32;
    let mut max_dist      = 0i32;

    while frontier_size > 0 {
        next_size.as_mut_slice()[0] = 0;

        unsafe {
            launch_bfs_kernel(
                graph.row_offsets.as_ptr(),
                graph.col_indices.as_ptr(),
                graph.edge_kinds.as_ptr(),
                dist_buf.as_mut_ptr(),
                frontier_buf.as_ptr(),
                frontier_size,
                next_buf.as_mut_ptr(),
                next_size.as_mut_ptr(),
                current_dist,
                filter_byte,
                n as u32,
            );
        }

        device_sync();

        frontier_size = next_size.as_slice()[0];
        if frontier_size > 0 {
            // Swap frontier and next buffers by copying next -> frontier
            let next_sl     = next_buf.as_slice()[..frontier_size as usize].to_vec();
            let frontier_sl = frontier_buf.as_mut_slice();
            frontier_sl[..frontier_size as usize].copy_from_slice(&next_sl);
            current_dist += 1;
            if current_dist > max_dist { max_dist = current_dist; }
        }
    }

    BfsResult {
        dist: dist_buf.as_slice().to_vec(),
        max_dist,
    }
}

extern "C" {
    fn launch_bfs_kernel(
        row_offsets:   *const u32,
        col_indices:   *const u32,
        edge_kinds:    *const u8,
        dist:          *mut i32,
        frontier:      *const u32,
        frontier_size: u32,
        next_frontier: *mut u32,
        next_size:     *mut u32,
        current_dist:  i32,
        edge_filter:   u8,
        n_nodes:       u32,
    );
}
