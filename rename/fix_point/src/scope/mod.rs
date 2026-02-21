mod frame;

pub struct ScopeBinder {
    /// Stack of scopes
    scopes: Vec<ScopeFrame>,
    /// Current active scope index
    current_scope: usize,
    /// Symbol table reference for method lookups
    symbol_table_ref: *const SymbolIndex,
}
