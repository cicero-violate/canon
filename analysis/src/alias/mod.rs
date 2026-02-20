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

use std::collections::{HashMap, HashSet};

pub type Var = String;

#[derive(Default)]
pub struct PointsToGraph {
    pub pt: HashMap<Var, HashSet<Var>>,
}

impl PointsToGraph {
    pub fn new() -> Self { Self::default() }

    /// p = &x
    pub fn add_address_of(&mut self, p: &str, x: &str) {
        self.pt.entry(p.to_string()).or_default().insert(x.to_string());
    }

    /// p = q  =>  pt(q) ⊆ pt(p)
    pub fn add_assign(&mut self, p: &str, q: &str) {
        let q_set = self.pt.get(q).cloned().unwrap_or_default();
        self.pt.entry(p.to_string()).or_default().extend(q_set);
    }

    /// Returns true if p and q may alias (pt(p) ∩ pt(q) ≠ ∅).
    pub fn may_alias(&self, p: &str, q: &str) -> bool {
        let pt_p = self.pt.get(p);
        let pt_q = self.pt.get(q);
        match (pt_p, pt_q) {
            (Some(a), Some(b)) => a.intersection(b).next().is_some(),
            _ => false,
        }
    }
}
