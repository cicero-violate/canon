//! Taint source/sink tracking.
//!
//! Variables:
//!   taint(v) = set of taint labels flowing into v
//!
//! Equations:
//!   v = source(l)    => l ∈ taint(v)
//!   v = f(w)         => taint(w) ⊆ taint(v)   (propagation)
//!   sink(v, policy)  => assert taint(v) ∩ policy.forbidden = ∅

use algorithms::graph::dfs::dfs;
#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
#[cfg(feature = "cuda")]
use algorithms::graph::gpu::bfs_gpu;
use std::collections::{HashMap, HashSet};

pub type VarId = String;
pub type Label = String;

#[derive(Default)]
pub struct TaintState {
    pub taint: HashMap<VarId, HashSet<Label>>,
    pub edges: Vec<(VarId, VarId)>,
}

impl TaintState {
    pub fn new() -> Self { Self::default() }

    pub fn add_source(&mut self, var: VarId, label: Label) {
        self.taint.entry(var).or_default().insert(label);
    }

    /// Propagate: taint(src) flows into dst.
    pub fn propagate(&mut self, src: &str, dst: &str) {
        self.edges.push((src.to_string(), dst.to_string()));
    }

    /// Check sink: returns forbidden labels reaching var.
    pub fn check_sink(&self, var: &str, forbidden: &HashSet<Label>) -> HashSet<Label> {
        if forbidden.is_empty() {
            return HashSet::new();
        }
        let mut nodes: HashSet<String> = HashSet::new();
        for (from, to) in &self.edges {
            nodes.insert(from.clone());
            nodes.insert(to.clone());
        }
        for src in self.taint.keys() {
            nodes.insert(src.clone());
        }
        let mut nodes: Vec<String> = nodes.into_iter().collect();
        nodes.sort();
        let index: HashMap<String, usize> =
            nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let Some(&sink_idx) = index.get(var) else {
            return HashSet::new();
        };
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (from, to) in &self.edges {
            if let (Some(&fi), Some(&ti)) = (index.get(from), index.get(to)) {
                adj[fi].push(ti);
            }
        }
        let mut hits = HashSet::new();
        if should_use_gpu(nodes.len(), self.edges.len()) {
            #[cfg(feature = "cuda")]
            {
                let csr = Csr::from_adj(&adj);
                for (src, labels) in &self.taint {
                    let Some(&src_idx) = index.get(src) else { continue; };
                    let levels = bfs_gpu(&csr, src_idx);
                    if levels[sink_idx] >= 0 {
                        for l in labels {
                            if forbidden.contains(l) {
                                hits.insert(l.clone());
                            }
                        }
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                for (src, labels) in &self.taint {
                    let Some(&src_idx) = index.get(src) else { continue; };
                    let reachable: HashSet<usize> = dfs(&adj, src_idx).into_iter().collect();
                    if reachable.contains(&sink_idx) {
                        for l in labels {
                            if forbidden.contains(l) {
                                hits.insert(l.clone());
                            }
                        }
                    }
                }
            }
        } else {
            for (src, labels) in &self.taint {
                let Some(&src_idx) = index.get(src) else { continue; };
                let reachable: HashSet<usize> = dfs(&adj, src_idx).into_iter().collect();
                if reachable.contains(&sink_idx) {
                    for l in labels {
                        if forbidden.contains(l) {
                            hits.insert(l.clone());
                        }
                    }
                }
            }
        }
        hits
    }
}

fn should_use_gpu(nodes: usize, edges: usize) -> bool {
    nodes.saturating_mul(edges) > 1_000_000
}
