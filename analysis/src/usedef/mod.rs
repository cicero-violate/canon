//! Use-before-define and def-use chains.
//!
//! Variables:
//!   defs(v) = { d | d defines variable v }
//!   uses(v) = { u | u uses variable v }
//!
//! Equations:
//!   def_use(d) = { u ∈ uses(v) | d reaches u }
//!   use_def(u) = { d ∈ defs(v) | d reaches u }

use std::collections::HashMap;

pub type VarId = String;
pub type SiteId = String;

#[derive(Default)]
pub struct UseDefChains {
    /// def site -> list of use sites it reaches
    pub def_use: HashMap<SiteId, Vec<SiteId>>,
    /// use site -> list of def sites that reach it
    pub use_def: HashMap<SiteId, Vec<SiteId>>,
}

impl UseDefChains {
    pub fn new() -> Self { Self::default() }

    pub fn add_chain(&mut self, def_site: SiteId, use_site: SiteId) {
        self.def_use.entry(def_site.clone()).or_default().push(use_site.clone());
        self.use_def.entry(use_site).or_default().push(def_site);
    }

    /// Variables used before any reaching definition (potential UB).
    pub fn used_before_defined(&self) -> Vec<SiteId> {
        self.use_def
            .iter()
            .filter(|(_, defs)| defs.is_empty())
            .map(|(u, _)| u.clone())
            .collect()
    }
}
