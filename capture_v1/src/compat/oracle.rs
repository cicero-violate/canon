use syn::Signature;

/// Bridge trait implemented by the rustc integration layer.
pub trait StructuralEditOracle {
    fn impact_of(&self, symbol_id: &str) -> Vec<String>;
    fn satisfies_bounds(&self, id: &str, new_sig: &Signature) -> bool;
    fn is_macro_generated(&self, symbol_id: &str) -> bool;
    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String>;
}

/// Fallback oracle for offline usage (no rustc integration).
#[derive(Debug, Clone, Default)]
pub struct NullOracle;

impl StructuralEditOracle for NullOracle {
    fn impact_of(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }

    fn satisfies_bounds(&self, _id: &str, _new_sig: &Signature) -> bool {
        true
    }

    fn is_macro_generated(&self, _symbol_id: &str) -> bool {
        false
    }

    fn cross_crate_users(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
}
