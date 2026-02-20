//! Escape analysis: determines if a value outlives its defining scope.
//!
//! Variables:
//!   escapes(v) = true if v is reachable outside its defining scope
//!
//! Equations:
//!   escapes(v) = v returned from function
//!              ∨ v stored in heap-allocated structure
//!              ∨ v passed to function that escapes it

use algorithms::searching::linear_search::linear_search;

pub type VarId = String;

#[derive(Default)]
pub struct EscapeSet {
    pub escaped: Vec<VarId>,
}

impl EscapeSet {
    pub fn new() -> Self { Self::default() }

    pub fn mark_escaped(&mut self, var: VarId) {
        if linear_search(&self.escaped, &var).is_none() {
            self.escaped.push(var);
        }
    }

    pub fn escapes(&self, var: &str) -> bool {
        linear_search(&self.escaped, &var.to_string()).is_some()
    }
}
