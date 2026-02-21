mod visitor;

pub(crate) use visitor::path_to_string;


use super::core::{span_to_range, SymbolIndex, SymbolOccurrence};


use super::alias::AliasGraph;


use super::resolve::Resolver;


use crate::pattern::extract_type_from_pattern;


use crate::pattern::binding::PatternBindingCollector;


use super::scope::ScopeBinder;


use algorithms::string_algorithms::kmp::kmp_search;


use proc_macro2::Span;


use std::collections::HashMap;


use std::path::Path;


use syn::visit::{self, Visit};


use syn::{
    Arm, Expr, ExprClosure, ExprForLoop, ExprMacro, ExprMethodCall, ImplItemFn, ItemImpl,
    Local, Pat,
};


#[derive(Clone)]
struct ImplCtx {
    type_name: String,
}


/// Enhanced occurrence visitor with full pattern and attribute support
pub struct OccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    alias_graph: &'a AliasGraph,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    scoped_binder: ScopeBinder,
    current_impl: Option<ImplCtx>,
    current_struct: Option<String>,
}


impl<'a> OccurrenceVisitor<'a> {
    pub fn new(
        module_path: &'a str,
        file: &'a Path,
        symbol_table: &'a SymbolIndex,
        use_map: &'a HashMap<String, String>,
        alias_graph: &'a AliasGraph,
        occurrences: &'a mut Vec<SymbolOccurrence>,
    ) -> Self {
        Self {
            module_path,
            file,
            symbol_table,
            use_map,
            alias_graph,
            occurrences,
            scoped_binder: ScopeBinder::new(symbol_table),
            current_impl: None,
            current_struct: None,
        }
    }
    fn add_occurrence(&mut self, id: String, kind: &str, span: Span) {
        self.occurrences
            .push(SymbolOccurrence {
                id,
                file: self.file.to_string_lossy().to_string(),
                kind: kind.to_string(),
                span: span_to_range(span),
            });
    }
    /// Process pattern bindings and add to scope
    fn process_pattern_bindings(&mut self, pat: &Pat) {
        let bindings = PatternBindingCollector::collect_from_pattern(pat);
        let type_hint = extract_type_from_pattern(pat);
        for (var_name, _) in bindings {
            if let Some(ref ty) = type_hint {
                self.scoped_binder.bind(var_name, ty.clone());
            }
        }
    }
    /// Infer type from an expression (simplified)
    fn infer_expr_type(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(expr_path) => {
                if let Some(ident) = expr_path.path.get_ident() {
                    let var_name = ident.to_string();
                    return self.scoped_binder.resolve(&var_name).cloned();
                }
                None
            }
            Expr::MethodCall(method_call) => {
                if let Some(receiver_type) = self.infer_expr_type(&method_call.receiver)
                {
                    let method_name = method_call.method.to_string();
                    return self
                        .scoped_binder
                        .get_method_return_type(&receiver_type, &method_name);
                }
                None
            }
            Expr::Struct(expr_struct) => {
                if let Some(symbol) = path_to_symbol(
                    &expr_struct.path,
                    self.module_path,
                    self.alias_graph,
                    self.symbol_table,
                ) {
                    Some(symbol)
                } else {
                    Some(path_to_string(&expr_struct.path))
                }
            }
            Expr::Call(call_expr) => {
                if let Expr::Path(ref path_expr) = *call_expr.func {
                    if let Some(func_symbol) = path_to_symbol(
                        &path_expr.path,
                        self.module_path,
                        self.alias_graph,
                        self.symbol_table,
                    ) {
                        if let Some(last_segment) = func_symbol.rsplit("::").next() {
                            if last_segment == "new" || last_segment == "default" {
                                let parts: Vec<&str> = func_symbol
                                    .rsplitn(2, "::")
                                    .collect();
                                if parts.len() == 2 {
                                    return Some(parts[1].to_string());
                                }
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }
}


pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
}


fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let resolver = Resolver::new(module_path, alias_graph, symbol_table);
    resolver.resolve_path_segments(&segments)
}


fn resolve_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    let mut module_parts: Vec<String> = module_path
        .split("::")
        .map(|s| s.to_string())
        .collect();
    let mut idx = 0usize;
    while idx < prefix.len() && prefix[idx] == "super" {
        if module_parts.len() > 1 {
            module_parts.pop();
        }
        idx += 1;
    }
    if idx < prefix.len() && prefix[idx] == "self" {
        idx += 1;
    }
    module_parts.extend(prefix[idx..].iter().cloned());
    module_parts
}
