//! Lifetime region analysis.
//!
//! Variables:
//!   region(r) = set of program points where borrow r is live
//!
//! Equations (NLL-style):
//!   region(r) ⊇ { p | r is used at p }
//!   region(r) ⊇ region(r') if r: r' (outlives constraint)

use algorithms::graph::dfs::dfs;
#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
#[cfg(feature = "cuda")]
use algorithms::graph::gpu::bfs_gpu;
use std::collections::{HashMap, HashSet};

pub type Region = String;
pub type ProgramPoint = usize;

#[derive(Default)]
pub struct LifetimeRegions {
    pub regions: HashMap<Region, HashSet<ProgramPoint>>,
    pub outlives: Vec<(Region, Region)>,
}

impl LifetimeRegions {
    pub fn new() -> Self { Self::default() }

    pub fn add_live_point(&mut self, region: Region, point: ProgramPoint) {
        self.regions.entry(region).or_default().insert(point);
    }

    pub fn add_outlives(&mut self, r: Region, r_prime: Region) {
        self.outlives.push((r, r_prime));
    }

    /// Propagate outlives constraints: region(r) ⊇ region(r') if r: r'
    pub fn propagate(&mut self) {
        let mut all: HashSet<String> = self.regions.keys().cloned().collect();
        for (r, r_prime) in &self.outlives {
            all.insert(r.clone());
            all.insert(r_prime.clone());
        }
        let mut nodes: Vec<String> = all.into_iter().collect();
        nodes.sort();
        let index: HashMap<String, usize> =
            nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (r, r_prime) in &self.outlives {
            if let (Some(&ri), Some(&rpi)) = (index.get(r), index.get(r_prime)) {
                adj[ri].push(rpi);
            }
        }
        if should_use_gpu(nodes.len(), self.outlives.len()) {
            #[cfg(feature = "cuda")]
            {
                let csr = Csr::from_adj(&adj);
                for (region, points) in self.regions.clone().into_iter() {
                    let Some(&start) = index.get(&region) else { continue; };
                    let levels = bfs_gpu(&csr, start);
                    for (r_idx, lvl) in levels.iter().enumerate() {
                        if *lvl >= 0 {
                            let r_name = &nodes[r_idx];
                            let entry = self.regions.entry(r_name.clone()).or_default();
                            entry.extend(points.iter().copied());
                        }
                    }
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                for (region, points) in self.regions.clone().into_iter() {
                    let Some(&start) = index.get(&region) else { continue; };
                    for r_idx in dfs(&adj, start) {
                        let r_name = &nodes[r_idx];
                        let entry = self.regions.entry(r_name.clone()).or_default();
                        entry.extend(points.iter().copied());
                    }
                }
            }
        } else {
            for (region, points) in self.regions.clone().into_iter() {
                let Some(&start) = index.get(&region) else { continue; };
                for r_idx in dfs(&adj, start) {
                    let r_name = &nodes[r_idx];
                    let entry = self.regions.entry(r_name.clone()).or_default();
                    entry.extend(points.iter().copied());
                }
            }
        }
    }
}

fn should_use_gpu(nodes: usize, edges: usize) -> bool {
    nodes.saturating_mul(edges) > 1_000_000
}
