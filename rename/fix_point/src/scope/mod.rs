mod frame;

use crate::scope::frame::ScopeFrame;


/// Replaces the ad-hoc LocalTypeContext with a proper scope hierarchy
use super::core::SymbolIndex;


pub struct ScopeBinder {
    /// Stack of scopes
    scopes: Vec<ScopeFrame>,
    /// Current active scope index
    current_scope: usize,
    /// Symbol table reference for method lookups
    symbol_table_ref: *const SymbolIndex,
}
