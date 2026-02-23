//! Scoped binder for tracking variable types through nested scopes
mod frame;
use crate::scope::frame::ScopeFrame;
/// Replaces the ad-hoc LocalTypeContext with a proper scope hierarchy
use super::core::SymbolIndex;
/// Scoped binder that tracks variable types through nested scopes
/// Supports pattern destructuring, closure captures, and method return types
pub struct ScopeBinder {
    /// Stack of scopes
    scopes: Vec<ScopeFrame>,
    /// Current active scope index
    current_scope: usize,
    /// Symbol table reference for method lookups
    symbol_table_ref: *const SymbolIndex,
}
impl ScopeBinder {
    pub fn new(symbol_table: &SymbolIndex) -> Self {
        let root = ScopeFrame::new(None);
        Self {
            scopes: vec![root],
            current_scope: 0,
            symbol_table_ref: symbol_table as *const SymbolIndex,
        }
    }
    /// Enter a new scope (for blocks, functions, closures)
    pub fn push_scope(&mut self) {
        let parent = self.current_scope;
        let scope = ScopeFrame::new(Some(parent));
        self.scopes.push(scope);
        self.current_scope = self.scopes.len() - 1;
    }
    /// Exit the current scope
    pub fn pop_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }
    /// Bind a variable name to a type in the current scope
    pub fn bind(&mut self, name: String, type_path: String) {
        self.scopes[self.current_scope].bindings.insert(name, type_path);
    }
    /// Resolve a variable name, searching up the scope chain
    pub fn resolve(&self, name: &str) -> Option<&String> {
        let mut scope_idx = self.current_scope;
        loop {
            let scope = &self.scopes[scope_idx];
            if let Some(ty) = scope.bindings.get(name) {
                return Some(ty);
            }
            match scope.parent {
                Some(parent) => scope_idx = parent,
                None => return None,
            }
        }
    }
    /// Resolve a method call to a symbol ID
    pub fn resolve_method(
        &self,
        receiver_type: &str,
        method_name: &str,
    ) -> Option<String> {
        let symbol_table = unsafe { &*self.symbol_table_ref };
        let direct_id = format!("{}::{}", receiver_type, method_name);
        if symbol_table.symbols.contains_key(&direct_id) {
            return Some(direct_id);
        }
        for (id, entry) in &symbol_table.symbols {
            if entry.kind == "trait_method" && entry.name == method_name {
                return Some(id.clone());
            }
        }
        None
    }
    /// Get the return type of a method by looking up its signature
    pub fn get_method_return_type(
        &self,
        receiver_type: &str,
        method_name: &str,
    ) -> Option<String> {
        let symbol_table = unsafe { &*self.symbol_table_ref };
        let method_id = self.resolve_method(receiver_type, method_name)?;
        let method_entry = symbol_table.symbols.get(&method_id)?;
        if let Some(signature) = method_entry
            .attributes
            .iter()
            .find(|attr| attr.starts_with("return_type:"))
        {
            let return_type = signature.trim_start_matches("return_type:").trim();
            return Some(return_type.to_string());
        }
        None
    }
}
