//! Control flow graph over MIR basic blocks.
//!
//! Variables:
//!   B   = set of basic blocks
//!   succ: B -> P(B)   successor relation
//!   pred: B -> P(B)   predecessor relation
//!
//! Equations:
//!   entry = B_0
//!   exit  = { b | succ(b) = ∅ }
//!   dom(b) = { b } ∪ ∩{ dom(p) | p ∈ pred(b) }   (dominators)

use algorithms::control_flow::dominators::dominators as doms;
#[cfg(feature = "cuda")]
use algorithms::control_flow::gpu::dominators_gpu;
#[cfg(feature = "cuda")]
use algorithms::graph::csr::Csr;
use std::collections::HashMap;

pub struct CfgNode {
    pub id: usize,
    pub label: String,
}

pub struct Cfg {
    pub nodes: Vec<CfgNode>,
    pub succ: HashMap<usize, Vec<usize>>,
    pub pred: HashMap<usize, Vec<usize>>,
}

impl Cfg {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            succ: HashMap::new(),
            pred: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize) {
        self.succ.entry(from).or_default().push(to);
        self.pred.entry(to).or_default().push(from);
    }

    /// Compute dominator sets using the iterative dataflow algorithm.
    /// dom(entry) = {entry}; dom(b) = {b} ∪ ∩ dom(pred(b))
    pub fn dominators(&self) -> HashMap<usize, std::collections::HashSet<usize>> {
        let node_count = if !self.nodes.is_empty() {
            self.nodes.len()
        } else {
            let max_id = self
                .succ
                .keys()
                .chain(self.pred.keys())
                .copied()
                .max()
                .unwrap_or(0);
            max_id + 1
        };
        let entry = self.nodes.first().map(|n| n.id).unwrap_or(0);
        if should_use_gpu(node_count, &self.pred) {
            #[cfg(feature = "cuda")]
            {
                if let Some(dom) = dominators_gpu_path(node_count, entry, &self.pred) {
                    return dom;
                }
            }
        }
        let dom_vec = doms(node_count, &self.pred, entry);
        dom_vec.into_iter().enumerate().collect()
    }
}

fn should_use_gpu(node_count: usize, pred: &HashMap<usize, Vec<usize>>) -> bool {
    let edges: usize = pred.values().map(|v| v.len()).sum();
    node_count.saturating_mul(edges) > 1_000_000
}

#[cfg(feature = "cuda")]
fn dominators_gpu_path(
    node_count: usize,
    entry: usize,
    pred: &HashMap<usize, Vec<usize>>,
) -> Option<HashMap<usize, std::collections::HashSet<usize>>> {
    // Require dense ids 0..node_count-1
    for id in 0..node_count {
        if !pred.contains_key(&id) && id != entry {
            // allow missing preds for nodes, but ensure id space dense
            continue;
        }
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    for (&n, preds) in pred {
        for &p in preds {
            if p < node_count {
                adj[n].push(p);
            }
        }
    }
    let pred_csr = Csr::from_adj(&adj);
    let dom_bits = dominators_gpu(&pred_csr, entry, node_count);
    let words = (node_count + 63) / 64;
    let mut out: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for n in 0..node_count {
        let mut set = std::collections::HashSet::new();
        for i in 0..node_count {
            let word = dom_bits[n * words + (i >> 6)];
            if (word >> (i & 63)) & 1 == 1 {
                set.insert(i);
            }
        }
        out.insert(n, set);
    }
    Some(out)
}

impl Default for Cfg {
    fn default() -> Self { Self::new() }
}
