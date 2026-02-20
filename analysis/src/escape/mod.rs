//! Escape analysis: determines if a value outlives its defining scope.
//!
//! Variables:
//!   escapes(v) = true if v is reachable outside its defining scope
//!
//! Equations:
//!   escapes(v) = v returned from function
//!              ∨ v stored in heap-allocated structure
//!              ∨ v passed to function that escapes it

use std::collections::HashSet;

pub type VarId = String;

#[derive(Default)]
pub struct EscapeSet {
    pub escaped: HashSet<VarId>,
}

impl EscapeSet {
    pub fn new() -> Self { Self::default() }

    pub fn mark_escaped(&mut self, var: VarId) {
        self.escaped.insert(var);
    }

    pub fn escapes(&self, var: &str) -> bool {
        self.escaped.contains(var)
    }
}
