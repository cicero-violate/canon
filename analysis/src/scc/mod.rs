//! Strongly connected components in the call graph (Kosaraju).
//!
//! Variables:
//!   index(v)   = discovery order of v
//!   lowlink(v) = lowest index reachable from v's subtree
//!
//! Equations:
//!   lowlink(v) = min({ index(v) }
//!                  ∪ { lowlink(w) | (v,w) ∈ E, w on stack }
//!                  ∪ { lowlink(w) | (v,w) ∈ E, w not yet visited })
//!   v is SCC root iff lowlink(v) = index(v)

use algorithms::graph::scc::kosaraju_scc;
use std::collections::{HashMap, HashSet};

pub fn tarjan_scc(adj: &HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
    let mut all: HashSet<String> = adj.keys().cloned().collect();
    for tos in adj.values() {
        for t in tos {
            all.insert(t.clone());
        }
    }
    let mut nodes: Vec<String> = all.into_iter().collect();
    nodes.sort();
    let index: HashMap<String, usize> =
        nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
    let mut graph = vec![Vec::new(); nodes.len()];
    for (from, tos) in adj {
        let Some(&fi) = index.get(from) else { continue; };
        for to in tos {
            if let Some(&ti) = index.get(to) {
                graph[fi].push(ti);
            }
        }
    }
    let comps = kosaraju_scc(&graph);
    comps
        .into_iter()
        .map(|comp| comp.into_iter().map(|i| nodes[i].clone()).collect())
        .collect()
}
