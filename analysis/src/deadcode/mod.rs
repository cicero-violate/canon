//! Dead code detection: unreachable symbols and branches.
//!
//! Variables:
//!   live(f) = true if f is reachable from any entry point
//!
//! Equations:
//!   live(entry) = true
//!   live(f)     = ∃ g: live(g) ∧ g calls f

use std::collections::{HashMap, HashSet};

pub struct DeadCodeAnalysis {
    /// call edges: caller -> callees
    pub calls: HashMap<String, Vec<String>>,
    pub entry_points: HashSet<String>,
}

impl DeadCodeAnalysis {
    pub fn new() -> Self {
        Self {
            calls: HashMap::new(),
            entry_points: HashSet::new(),
        }
    }

    pub fn add_call(&mut self, caller: String, callee: String) {
        self.calls.entry(caller).or_default().push(callee);
    }

    pub fn add_entry(&mut self, entry: String) {
        self.entry_points.insert(entry);
    }

    /// Returns set of symbols not reachable from any entry point.
    pub fn dead_symbols(&self) -> HashSet<String> {
        let mut live = HashSet::new();
        let mut queue: std::collections::VecDeque<String> =
            self.entry_points.iter().cloned().collect();
        while let Some(f) = queue.pop_front() {
            if !live.insert(f.clone()) { continue; }
            if let Some(callees) = self.calls.get(&f) {
                for c in callees { queue.push_back(c.clone()); }
            }
        }
        self.calls.keys()
            .chain(self.entry_points.iter())
            .filter(|f| !live.contains(*f))
            .cloned()
            .collect()
    }
}

impl Default for DeadCodeAnalysis {
    fn default() -> Self { Self::new() }
}
