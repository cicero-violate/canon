//! Interprocedural summary computation.
//!
//! Variables:
//!   summary(f) = (pre, post) — precondition/postcondition of f
//!
//! Equations:
//!   summary(f) is computed bottom-up in the call graph SCC order.
//!   For recursive SCCs, iterate to fixpoint.

use algorithms::sorting::merge_sort::merge_sort;
use std::collections::HashMap;

pub type FnId = String;

#[derive(Clone, Debug, Default)]
pub struct FnSummary {
    /// Abstract pre-state (opaque for now, stored as key-value facts)
    pub pre: HashMap<String, String>,
    /// Abstract post-state
    pub post: HashMap<String, String>,
}

#[derive(Default)]
pub struct SummaryStore {
    pub summaries: HashMap<FnId, FnSummary>,
}

impl SummaryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, fn_id: FnId, summary: FnSummary) {
        self.summaries.insert(fn_id, summary);
    }

    pub fn get(&self, fn_id: &str) -> Option<&FnSummary> {
        self.summaries.get(fn_id)
    }

    /// Compute summaries bottom-up given a topological order of SCCs.
    /// Each SCC is a vec of fn ids; `compute` is called per function.
    pub fn compute_bottom_up<F>(&mut self, scc_order: &[Vec<FnId>], mut compute: F)
    where F: FnMut(&FnId, &SummaryStore) -> FnSummary {
        for scc in scc_order {
            let ordered = merge_sort(scc);
            // Iterate to fixpoint for recursive SCCs
            let mut changed = true;
            while changed {
                changed = false;
                for fn_id in &ordered {
                    let new_summary = compute(fn_id, self);
                    let old = self.summaries.get(fn_id);
                    if old.map(|s| s.post != new_summary.post).unwrap_or(true) {
                        self.summaries.insert(fn_id.clone(), new_summary);
                        changed = true;
                    }
                }
            }
        }
    }
}
