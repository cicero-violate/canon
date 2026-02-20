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

#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
use algorithms::graph::dijkstra::dijkstra;
#[cfg(feature = "cuda")]
use algorithms::graph::gpu::bfs_gpu;
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
    /// Builds a call-only weighted graph (weight=1), then runs Dijkstra.
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
        if v == 0 {
            return vec![];
        }

        // Find start index by key
        let start_id = WireNodeId::from_key(start);
        let Some(&start_idx) = id_to_idx.get(&start_id) else {
            return vec![];
        };

        // Build weighted adjacency list filtered to call edges only
        let mut adj: Vec<Vec<(usize, u64)>> = vec![vec![]; v];
        for edge in edges {
            if edge.kind == "call" {
                if let (Some(&fi), Some(&ti)) = (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to)) {
                    adj[fi].push((ti, 1));
                }
            }
        }
        if should_use_gpu(v, adj.iter().map(|v| v.len()).sum()) {
            #[cfg(feature = "cuda")]
            {
                let mut adj_u: Vec<Vec<usize>> = vec![Vec::new(); v];
                for (from, tos) in adj.iter().enumerate() {
                    for (to, _) in tos {
                        adj_u[from].push(*to);
                    }
                }
                let csr = Csr::from_adj(&adj_u);
                let levels = bfs_gpu(&csr, start_idx);
                return levels.into_iter().enumerate().filter(|(_, lvl)| *lvl >= 0).map(|(i, _)| idx_to_key[i].clone()).collect();
            }
        }
        let dist = dijkstra(&adj, start_idx);
        dist.into_iter().enumerate().filter(|(_, d)| *d != u64::MAX).map(|(i, _)| idx_to_key[i].clone()).collect()
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
        if v == 0 {
            return -1;
        }
        let start_id = WireNodeId::from_key(start);
        let target_id = WireNodeId::from_key(target);
        let (Some(&start_idx), Some(&target_idx)) = (id_to_idx.get(&start_id), id_to_idx.get(&target_id)) else {
            return -1;
        };
        let mut adj: Vec<Vec<(usize, u64)>> = vec![vec![]; v];
        for edge in edges {
            if edge.kind == "call" {
                if let (Some(&fi), Some(&ti)) = (id_to_idx.get(&edge.from), id_to_idx.get(&edge.to)) {
                    adj[fi].push((ti, 1));
                }
            }
        }
        if should_use_gpu(v, adj.iter().map(|v| v.len()).sum()) {
            #[cfg(feature = "cuda")]
            {
                let mut adj_u: Vec<Vec<usize>> = vec![Vec::new(); v];
                for (from, tos) in adj.iter().enumerate() {
                    for (to, _) in tos {
                        adj_u[from].push(*to);
                    }
                }
                let csr = Csr::from_adj(&adj_u);
                let levels = bfs_gpu(&csr, start_idx);
                return levels[target_idx];
            }
        }
        let dist = dijkstra(&adj, start_idx);
        match dist[target_idx] {
            u64::MAX => -1,
            d => d as i32,
        }
    }
}

fn should_use_gpu(nodes: usize, edges: usize) -> bool {
    nodes.saturating_mul(edges) > 1_000_000
}
