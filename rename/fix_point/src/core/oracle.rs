use syn::Signature;


pub trait StructuralEditOracle {
    fn impact_of(&self, symbol_id: &str) -> Vec<String>;
    fn satisfies_bounds(&self, id: &str, new_sig: &Signature) -> bool;
    fn is_macro_generated(&self, symbol_id: &str) -> bool;
    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String>;
}
