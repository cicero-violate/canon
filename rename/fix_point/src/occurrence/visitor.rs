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
