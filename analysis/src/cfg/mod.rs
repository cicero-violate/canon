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
        let all: std::collections::HashSet<usize> = self.nodes.iter().map(|n| n.id).collect();
        let mut dom: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
        for node in &self.nodes {
            dom.insert(node.id, all.clone());
        }
        if let Some(entry) = self.nodes.first() {
            dom.insert(entry.id, std::iter::once(entry.id).collect());
        }
        let mut changed = true;
        while changed {
            changed = false;
            for node in self.nodes.iter().skip(1) {
                let preds = self.pred.get(&node.id).cloned().unwrap_or_default();
                let new_dom = if preds.is_empty() {
                    all.clone()
                } else {
                    let mut iter = preds.iter().map(|p| dom[p].clone());
                    let first = iter.next().unwrap();
                    iter.fold(first, |acc, s| acc.intersection(&s).cloned().collect())
                };
                let mut new_dom2 = new_dom;
                new_dom2.insert(node.id);
                if dom[&node.id] != new_dom2 {
                    dom.insert(node.id, new_dom2);
                    changed = true;
                }
            }
        }
        dom
    }
}

impl Default for Cfg {
    fn default() -> Self { Self::new() }
}
