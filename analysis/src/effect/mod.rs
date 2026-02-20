//! Side effect tracking.
//!
//! Variables:
//!   effects(f) = set of effect kinds produced by f
//!
//! Effect kinds: Reads, Writes, Allocates, Panics, Io, Unsafe

use algorithms::searching::hash_lookup::hash_lookup;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Effect {
    Reads(String),
    Writes(String),
    Allocates,
    Panics,
    Io,
    Unsafe,
}

#[derive(Default)]
pub struct EffectMap {
    pub effects: HashMap<String, HashSet<Effect>>,
}

impl EffectMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, fn_id: &str, effect: Effect) {
        self.effects.entry(fn_id.to_string()).or_default().insert(effect);
    }

    pub fn get(&self, fn_id: &str) -> HashSet<&Effect> {
        hash_lookup(&self.effects, &fn_id.to_string()).map(|s| s.iter().collect()).unwrap_or_default()
    }

    pub fn is_pure(&self, fn_id: &str) -> bool {
        self.effects.get(fn_id).map(|s| s.is_empty()).unwrap_or(true)
    }
}
