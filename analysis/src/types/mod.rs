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

use algorithms::graph::dfs::dfs;
use std::collections::{HashMap, HashSet};

pub type TypeId = String;

#[derive(Default)]
pub struct TypeHierarchy {
    /// subtype edges: child -> set of supertypes
    pub supertypes: HashMap<TypeId, HashSet<TypeId>>,
}

impl TypeHierarchy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_subtype(&mut self, child: TypeId, parent: TypeId) {
        self.supertypes.entry(child).or_default().insert(parent);
    }

    /// Is `child` a subtype of `parent` (transitive)?
    pub fn is_subtype(&self, child: &str, parent: &str) -> bool {
        if child == parent {
            return true;
        }
        let mut all: HashSet<String> = self.supertypes.keys().cloned().collect();
        for set in self.supertypes.values() {
            for s in set {
                all.insert(s.clone());
            }
        }
        let mut nodes: Vec<String> = all.into_iter().collect();
        nodes.sort();
        let index: HashMap<String, usize> = nodes.iter().enumerate().map(|(i, k)| (k.clone(), i)).collect();
        let (Some(&start), Some(&target)) = (index.get(child), index.get(parent)) else {
            return false;
        };
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
        for (from, tos) in &self.supertypes {
            let Some(&fi) = index.get(from) else {
                continue;
            };
            for to in tos {
                if let Some(&ti) = index.get(to) {
                    adj[fi].push(ti);
                }
            }
        }
        dfs(&adj, start).into_iter().any(|i| i == target)
    }
}
