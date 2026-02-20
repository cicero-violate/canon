//! Alias and points-to analysis.
//!
//! Variables:
//!   pt(p) = points-to set of pointer p
//!
//! Equations (Andersen-style inclusion):
//!   p = &x  =>  x ∈ pt(p)
//!   p = q   =>  pt(q) ⊆ pt(p)
//!   p = *q  =>  ∀ r ∈ pt(q): pt(r) ⊆ pt(p)
//!   *p = q  =>  ∀ r ∈ pt(p): pt(q) ⊆ pt(r)

#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
use algorithms::graph::dfs::dfs;
#[cfg(feature = "cuda")]
use algorithms::graph::gpu::bfs_gpu;
use std::collections::{HashMap, HashSet};

pub type Var = String;

#[derive(Default)]
pub struct PointsToGraph {
    pub edges: Vec<(Var, Var)>,
    pub vars: HashSet<Var>,
}

impl PointsToGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// p = &x
    pub fn add_address_of(&mut self, p: &str, x: &str) {
        self.edges.push((p.to_string(), x.to_string()));
        self.vars.insert(p.to_string());
        self.vars.insert(x.to_string());
    }

    /// p = q  =>  pt(q) ⊆ pt(p)
    pub fn add_assign(&mut self, p: &str, q: &str) {
        self.edges.push((p.to_string(), q.to_string()));
        self.vars.insert(p.to_string());
        self.vars.insert(q.to_string());
    }

    /// Returns true if p and q may alias (pt(p) ∩ pt(q) ≠ ∅).
    pub fn may_alias(&self, p: &str, q: &str) -> bool {
        let mut nodes: Vec<String> = self.vars.iter().cloned().collect();
        nodes.sort();
        let index: HashMap<String, usize> = nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let (Some(&pi), Some(&qi)) = (index.get(p), index.get(q)) else {
            return false;
        };
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (from, to) in &self.edges {
            if let (Some(&fi), Some(&ti)) = (index.get(from), index.get(to)) {
                adj[fi].push(ti);
            }
        }
        let (reach_p, reach_q) = if should_use_gpu(nodes.len(), self.edges.len()) {
            #[cfg(feature = "cuda")]
            {
                let csr = Csr::from_adj(&adj);
                let lp = bfs_gpu(&csr, pi);
                let lq = bfs_gpu(&csr, qi);
                let rp: HashSet<usize> = lp.iter().enumerate().filter(|(_, l)| **l >= 0).map(|(i, _)| i).collect();
                let rq: HashSet<usize> = lq.iter().enumerate().filter(|(_, l)| **l >= 0).map(|(i, _)| i).collect();
                (rp, rq)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let rp: HashSet<usize> = dfs(&adj, pi).into_iter().collect();
                let rq: HashSet<usize> = dfs(&adj, qi).into_iter().collect();
                (rp, rq)
            }
        } else {
            let rp: HashSet<usize> = dfs(&adj, pi).into_iter().collect();
            let rq: HashSet<usize> = dfs(&adj, qi).into_iter().collect();
            (rp, rq)
        };
        reach_p.intersection(&reach_q).next().is_some()
    }
}

fn should_use_gpu(nodes: usize, edges: usize) -> bool {
    nodes.saturating_mul(edges) > 1_000_000
}
