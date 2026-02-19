//! Graph traversal — BFS with GPU acceleration when available,
//! CPU fallback otherwise. Same API either way.
use crate::csr::{CsrGraph, EdgeKind};
use crate::unified::{cuda_available, device_sync};
#[cfg(feature = "cuda")]
use crate::unified::UnifiedVec;

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

/// Edge-kind filter bitmask for traversals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeMask(u8);

impl EdgeMask {
    pub const ALL: EdgeMask = EdgeMask(0x0F);

    pub fn from_kind(kind: EdgeKind) -> Self {
        EdgeMask(1u8 << (kind as u8))
    }

    pub fn from_kinds(kinds: &[EdgeKind]) -> Self {
        let mut mask = 0u8;
        for kind in kinds {
            mask |= 1u8 << (*kind as u8);
        }
        EdgeMask(mask)
    }

    fn allows(self, kind: u8) -> bool {
        (self.0 & (1u8 << kind)) != 0
    }
}

/// BFS from `source` (internal index), optionally filtering by edge kind.
/// Uses GPU kernel when CUDA is available, CPU otherwise.
pub fn bfs(graph: &CsrGraph, source: u32, edge_filter: Option<EdgeKind>) -> BfsResult {
    let mask = edge_filter.map(EdgeMask::from_kind).unwrap_or(EdgeMask::ALL);
    bfs_mask(graph, source, mask)
}

/// BFS from `source` (internal index), filtering by edge mask.
pub fn bfs_mask(graph: &CsrGraph, source: u32, mask: EdgeMask) -> BfsResult {
    #[cfg(feature = "cuda")]
    if cuda_available() {
        return bfs_gpu_mask(graph, source, mask);
    }
    bfs_cpu_mask(graph, source, mask)
}

/// Result of a DFS traversal.
pub struct DfsResult {
    /// Visit order (preorder) as internal indices.
    pub order: Vec<u32>,
    /// Parent per node (-1 for root/unreached).
    pub parent: Vec<i32>,
}

/// DFS from `source` (internal index), optionally filtering by edge kind.
pub fn dfs(graph: &CsrGraph, source: u32, edge_filter: Option<EdgeKind>) -> DfsResult {
    let mask = edge_filter.map(EdgeMask::from_kind).unwrap_or(EdgeMask::ALL);
    let n = graph.n_nodes;
    let mut visited = vec![false; n];
    let mut parent = vec![-1i32; n];
    let mut order = Vec::new();
    let mut stack = Vec::new();
    stack.push(source);

    while let Some(u) = stack.pop() {
        let u_idx = u as usize;
        if visited[u_idx] {
            continue;
        }
        visited[u_idx] = true;
        order.push(u);

        let neighbors = graph.neighbors(u);
        let kinds = graph.neighbor_kinds(u);
        for (&v, &k) in neighbors.iter().zip(kinds.iter()).rev() {
            if !mask.allows(k) {
                continue;
            }
            let v_idx = v as usize;
            if !visited[v_idx] {
                if parent[v_idx] == -1 {
                    parent[v_idx] = u as i32;
                }
                stack.push(v);
            }
        }
    }

    DfsResult { order, parent }
}

/// Compute immediate dominators for all reachable nodes from `start`.
/// Returns a Vec of `i32` where -1 indicates no dominator (unreachable or root).
pub fn dominator_tree(
    graph: &CsrGraph,
    start: u32,
    edge_filter: Option<EdgeKind>,
) -> Vec<i32> {
    let mask = edge_filter.map(EdgeMask::from_kind).unwrap_or(EdgeMask::ALL);
    let n = graph.n_nodes;
    let reach = bfs_mask(graph, start, mask);
    let reachable: Vec<bool> = reach.dist.iter().map(|&d| d >= 0).collect();

    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        let u_idx = u as u32;
        let neighbors = graph.neighbors(u_idx);
        let kinds = graph.neighbor_kinds(u_idx);
        for (&v, &k) in neighbors.iter().zip(kinds.iter()) {
            if mask.allows(k) {
                preds[v as usize].push(u);
            }
        }
    }

    let mut dom: Vec<Vec<bool>> = vec![vec![true; n]; n];
    for v in 0..n {
        if !reachable[v] {
            let mut only_self = vec![false; n];
            only_self[v] = true;
            dom[v] = only_self;
            continue;
        }
        if v == start as usize {
            let mut only_start = vec![false; n];
            only_start[v] = true;
            dom[v] = only_start;
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for v in 0..n {
            if v == start as usize || !reachable[v] {
                continue;
            }
            if preds[v].is_empty() {
                let mut only_self = vec![false; n];
                only_self[v] = true;
                if dom[v] != only_self {
                    dom[v] = only_self;
                    changed = true;
                }
                continue;
            }
            let mut new_dom = vec![true; n];
            for &p in &preds[v] {
                for i in 0..n {
                    new_dom[i] &= dom[p][i];
                }
            }
            new_dom[v] = true;
            if new_dom != dom[v] {
                dom[v] = new_dom;
                changed = true;
            }
        }
    }

    let mut idom = vec![-1i32; n];
    for v in 0..n {
        if v == start as usize || !reachable[v] {
            continue;
        }
        let candidates: Vec<usize> = dom[v]
            .iter()
            .enumerate()
            .filter(|(i, &d)| d && *i != v)
            .map(|(i, _)| i)
            .collect();
        let mut chosen = None;
        'outer: for d in &candidates {
            for other in &candidates {
                if other == d {
                    continue;
                }
                if dom[*other][*d] {
                    continue 'outer;
                }
            }
            chosen = Some(*d);
            break;
        }
        if let Some(d) = chosen {
            idom[v] = d as i32;
        }
    }

    idom
}

/// CPU BFS — standard queue-based.
fn bfs_cpu_mask(graph: &CsrGraph, source: u32, mask: EdgeMask) -> BfsResult {
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
            if !mask.allows(k) { continue; }
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
#[cfg(feature = "cuda")]
fn bfs_gpu_mask(graph: &CsrGraph, source: u32, mask: EdgeMask) -> BfsResult {
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
                mask.0,
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

#[cfg(feature = "cuda")]
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
