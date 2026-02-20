//! Type-level analysis: type compatibility, coercion, and variance.
//!
//! Variables:
//!   T = type term
//!   subtype(T1, T2) = T1 <: T2
//!
//! Equations:
//!   T <: T                         (reflexivity)
//!   T1 <: T2, T2 <: T3 => T1 <: T3 (transitivity)
//!   &'a T <: &'b T if 'a: 'b       (lifetime covariance)

use std::collections::{HashMap, HashSet};

pub type TypeId = String;

#[derive(Default)]
pub struct TypeHierarchy {
    /// subtype edges: child -> set of supertypes
    pub supertypes: HashMap<TypeId, HashSet<TypeId>>,
}

impl TypeHierarchy {
    pub fn new() -> Self { Self::default() }

    pub fn add_subtype(&mut self, child: TypeId, parent: TypeId) {
        self.supertypes.entry(child).or_default().insert(parent);
    }

    /// Is `child` a subtype of `parent` (transitive)?
    pub fn is_subtype(&self, child: &str, parent: &str) -> bool {
        if child == parent { return true; }
        let mut visited = HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(child.to_string());
        while let Some(t) = queue.pop_front() {
            if !visited.insert(t.clone()) { continue; }
            if let Some(supers) = self.supertypes.get(&t) {
                for s in supers {
                    if s == parent { return true; }
                    queue.push_back(s.clone());
                }
            }
        }
        false
    }
}
