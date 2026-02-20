//! Dead code detection: unreachable symbols and branches.
//!
//! Variables:
//!   live(f) = true if f is reachable from any entry point
//!
//! Equations:
//!   live(entry) = true
//!   live(f)     = ∃ g: live(g) ∧ g calls f

#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
use algorithms::graph::dfs::dfs;
#[cfg(feature = "cuda")]
use algorithms::graph::gpu::bfs_gpu;
use std::collections::{HashMap, HashSet};

pub struct DeadCodeAnalysis {
    /// call edges: caller -> callees
    pub calls: HashMap<String, Vec<String>>,
    pub entry_points: HashSet<String>,
}

impl DeadCodeAnalysis {
    pub fn new() -> Self {
        Self { calls: HashMap::new(), entry_points: HashSet::new() }
    }

    pub fn add_call(&mut self, caller: String, callee: String) {
        self.calls.entry(caller).or_default().push(callee);
    }

    pub fn add_entry(&mut self, entry: String) {
        self.entry_points.insert(entry);
    }

    /// Returns set of symbols not reachable from any entry point.
    pub fn dead_symbols(&self) -> HashSet<String> {
        let mut all: HashSet<String> = self.calls.keys().cloned().collect();
        for callees in self.calls.values() {
            for c in callees {
                all.insert(c.clone());
            }
        }
        for e in &self.entry_points {
            all.insert(e.clone());
        }
        let mut nodes: Vec<String> = all.into_iter().collect();
        nodes.sort();
        let index: HashMap<String, usize> = nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (from, tos) in &self.calls {
            let Some(&fi) = index.get(from) else {
                continue;
            };
            for to in tos {
                if let Some(&ti) = index.get(to) {
                    adj[fi].push(ti);
                }
            }
        }
        let mut live: HashSet<usize> = HashSet::new();
        if should_use_gpu(nodes.len(), adj.iter().map(|v| v.len()).sum()) {
            #[cfg(feature = "cuda")]
            {
                let csr = Csr::from_adj(&adj);
                for entry in &self.entry_points {
                    if let Some(&start) = index.get(entry) {
                        let levels = bfs_gpu(&csr, start);
                        for (i, lvl) in levels.iter().enumerate() {
                            if *lvl >= 0 {
                                live.insert(i);
                            }
                        }
                    }
                }
            }
        } else {
            for entry in &self.entry_points {
                if let Some(&start) = index.get(entry) {
                    for v in dfs(&adj, start) {
                        live.insert(v);
                    }
                }
            }
        }
        nodes.into_iter().enumerate().filter(|(i, _)| !live.contains(i)).map(|(_, k)| k).collect()
    }
}

fn should_use_gpu(nodes: usize, edges: usize) -> bool {
    nodes.saturating_mul(edges) > 1_000_000
}

impl Default for DeadCodeAnalysis {
    fn default() -> Self {
        Self::new()
    }
}
