use syn::{
    Arm, Expr, ExprClosure, ExprForLoop, ExprMacro, ExprMethodCall, ImplItemFn, ItemImpl,
    Local, Pat,
};


use syn::ItemMacro;


use crate::macros::{extract_macro_rules_identifiers, MacroIdentifierCollector};


use crate::attributes::extract_symbols_from_attributes;


use syn::visit::{self, Visit};


use std::path::Path;


use crate::resolve::Resolver;


use super::{OccurrenceVisitor, ImplCtx, path_to_symbol, resolve_relative_prefix};


pub fn normalize_use_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    if prefix.first().map(|s| s.as_str()) == Some("crate") {
        return prefix.to_vec();
    }
    if prefix.first().map(|s| s.as_str()) == Some("self")
        || prefix.first().map(|s| s.as_str()) == Some("super")
    {
        return resolve_relative_prefix(prefix, module_path);
    }
    let mut out: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    out.extend(prefix.iter().cloned());
    out
}


pub fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}
