//! Lifetime region analysis.
//!
//! Variables:
//!   region(r) = set of program points where borrow r is live
//!
//! Equations (NLL-style):
//!   region(r) ⊇ { p | r is used at p }
//!   region(r) ⊇ region(r') if r: r' (outlives constraint)

use std::collections::{HashMap, HashSet};

pub type Region = String;
pub type ProgramPoint = usize;

#[derive(Default)]
pub struct LifetimeRegions {
    pub regions: HashMap<Region, HashSet<ProgramPoint>>,
    pub outlives: Vec<(Region, Region)>,
}

impl LifetimeRegions {
    pub fn new() -> Self { Self::default() }

    pub fn add_live_point(&mut self, region: Region, point: ProgramPoint) {
        self.regions.entry(region).or_default().insert(point);
    }

    pub fn add_outlives(&mut self, r: Region, r_prime: Region) {
        self.outlives.push((r, r_prime));
    }

    /// Propagate outlives constraints: region(r) ⊇ region(r') if r: r'
    pub fn propagate(&mut self) {
        let mut changed = true;
        while changed {
            changed = false;
            for (r, r_prime) in &self.outlives.clone() {
                let rp_set = self.regions.get(r_prime).cloned().unwrap_or_default();
                let r_set = self.regions.entry(r.clone()).or_default();
                let before = r_set.len();
                r_set.extend(rp_set);
                if r_set.len() != before { changed = true; }
            }
        }
    }
}
