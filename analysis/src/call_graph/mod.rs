//! Call graph construction and reachability.
//!
//! Variables:
//!   N = set of function nodes
//!   E ⊆ N × N = call edges (caller -> callee), kind = "call"
//!   G = (N, E) stored as CSR in GraphSnapshot.csr_cache
//!
//! Equations:
//!   reachable(f) = { g | path f ->* g in G }
//!     computed via bfs_gpu(csr, start_idx) when cuda feature is enabled,
//!     falling back to CPU BFS otherwise.
//!   call_depth(f) = level[idx(f)] from BFS level vector

use algorithms::graph::csr::Csr;
use database::graph_log::{GraphSnapshot, WireNodeId};
use std::collections::HashMap;

pub struct CallGraph {
    pub snapshot: GraphSnapshot,
}

impl CallGraph {
    pub fn from_snapshot(snapshot: GraphSnapshot) -> Self {
        Self { snapshot }
    }

    /// Return all node keys reachable from `start` via call edges.
    /// Builds a call-only CSR (filtering to kind == "call"), then runs BFS.
    pub fn reachable_from(&mut self, start: &str) -> Vec<String> {
        let nodes = &self.snapshot.nodes;
        let edges = &self.snapshot.edges;

        // Dense index: WireNodeId -> usize
        let mut id_to_idx: HashMap<WireNodeId, usize> = HashMap::new();
        let mut idx_to_key: Vec<String> = Vec::new();
        for node in nodes {
            id_to_idx.insert(node.id.clone(), idx_to_key.len());
            idx_to_key.push(node.key.clone());
        }
        let v = idx_to_key.len();
        if v == 0 { return vec![]; }

        // Find start index by key
        let start_id = WireNodeId::from_key(start);
        let Some(&start_idx) = id_to_idx.get(&start_id) else {
            return vec![];
        };

        // Build adjacency list filtered to call edges only
        let mut adj: Vec<Vec<usize>> = vec![vec![]; v];
        for edge in edges {
            if edge.kind == "call" {
                if let (Some(&fi), Some(&ti)) =
                    (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to))
                {
                    adj[fi].push(ti);
                }
            }
        }
        let csr = Csr::from_adj(&adj);
        let levels = Self::bfs_cpu(&csr, start_idx);
        levels
            .into_iter()
            .enumerate()
            .filter(|(_, lvl)| *lvl >= 0)
            .map(|(i, _)| idx_to_key[i].clone())
            .collect()
    }

    /// GPU BFS when cuda feature enabled, returns level vector.
    #[cfg(feature = "cuda")]
    fn bfs_levels(csr: &Csr, start: usize) -> Vec<i32> {
        algorithms::graph::gpu::bfs_gpu(csr, start)
    }

    /// CPU BFS fallback: returns level vector (-1 = unreachable).
    fn bfs_cpu(csr: &Csr, start: usize) -> Vec<i32> {
        let mut level = vec![-1i32; csr.vertex_count()];
        let mut queue = std::collections::VecDeque::new();
        level[start] = 0;
        queue.push_back(start);
        while let Some(v) = queue.pop_front() {
            for &u in csr.neighbours(v) {
                let u = u as usize;
                if level[u] == -1 {
                    level[u] = level[v] + 1;
                    queue.push_back(u);
                }
            }
        }
        level
    }

    /// Call depth of `target` from `start` (-1 = unreachable).
    pub fn call_depth(&mut self, start: &str, target: &str) -> i32 {
        let nodes = &self.snapshot.nodes;
        let edges = &self.snapshot.edges;
        let mut id_to_idx: HashMap<WireNodeId, usize> = HashMap::new();
        let mut idx_to_key: Vec<String> = Vec::new();
        for node in nodes {
            id_to_idx.insert(node.id.clone(), idx_to_key.len());
            idx_to_key.push(node.key.clone());
        }
        let v = idx_to_key.len();
        if v == 0 { return -1; }
        let start_id = WireNodeId::from_key(start);
        let target_id = WireNodeId::from_key(target);
        let (Some(&start_idx), Some(&target_idx)) =
            (id_to_idx.get(&start_id), id_to_idx.get(&target_id))
        else { return -1; };
        let mut adj: Vec<Vec<usize>> = vec![vec![]; v];
        for edge in edges {
            if edge.kind == "call" {
                if let (Some(&fi), Some(&ti)) =
                    (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to))
                {
                    adj[fi].push(ti);
                }
            }
        }
        let csr = Csr::from_adj(&adj);
        let levels = Self::bfs_cpu(&csr, start_idx);
        levels[target_idx]
    }
}
