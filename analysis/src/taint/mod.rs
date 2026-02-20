//! Taint source/sink tracking.
//!
//! Variables:
//!   taint(v) = set of taint labels flowing into v
//!
//! Equations:
//!   v = source(l)    => l ∈ taint(v)
//!   v = f(w)         => taint(w) ⊆ taint(v)   (propagation)
//!   sink(v, policy)  => assert taint(v) ∩ policy.forbidden = ∅

use std::collections::{HashMap, HashSet};

pub type VarId = String;
pub type Label = String;

#[derive(Default)]
pub struct TaintState {
    pub taint: HashMap<VarId, HashSet<Label>>,
}

impl TaintState {
    pub fn new() -> Self { Self::default() }

    pub fn add_source(&mut self, var: VarId, label: Label) {
        self.taint.entry(var).or_default().insert(label);
    }

    /// Propagate: taint(src) flows into dst.
    pub fn propagate(&mut self, src: &str, dst: &str) {
        let src_labels = self.taint.get(src).cloned().unwrap_or_default();
        self.taint.entry(dst.to_string()).or_default().extend(src_labels);
    }

    /// Check sink: returns forbidden labels reaching var.
    pub fn check_sink(&self, var: &str, forbidden: &HashSet<Label>) -> HashSet<Label> {
        self.taint.get(var)
            .map(|labels| labels.intersection(forbidden).cloned().collect())
            .unwrap_or_default()
    }
}
