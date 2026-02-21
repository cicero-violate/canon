//! Enhanced occurrence tracking with pattern destructuring and attribute support
use super::attributes::extract_symbols_from_attributes;
use super::core::{span_to_range, SymbolIndex, SymbolOccurrence};
use super::alias::AliasGraph;
use super::resolve::Resolver;
use super::macros::{extract_macro_rules_identifiers, MacroIdentifierCollector};
use super::pattern::{extract_type_from_pattern, PatternBindingCollector};
use super::scope::ScopeBinder;
use algorithms::string_algorithms::kmp::kmp_search;
use proc_macro2::Span;
use std::collections::HashMap;
use std::path::Path;
use syn::visit::{self, Visit};
use syn::ItemMacro;
use syn::{
    Arm, Expr, ExprClosure, ExprForLoop, ExprMacro, ExprMethodCall, ImplItemFn, ItemImpl,
    Local, Pat,
};
/// KMP pre-filter: returns true if `symbol` appears anywhere in `source`.
/// Avoids running the full AST visitor on files that cannot contain the symbol.
pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
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
    current_impl: Option<OccurrenceImplContext>,
    current_struct: Option<String>,
}
#[derive(Clone)]
struct OccurrenceImplContext {
    type_name: String,
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
impl<'a> Visit<'a> for OccurrenceVisitor<'a> {
    fn visit_item_mod(&mut self, node: &'a syn::ItemMod) {
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }
    }
    fn visit_item_struct(&mut self, node: &'a syn::ItemStruct) {
        let struct_name = format!("{}::{}", self.module_path, node.ident);
        self.current_struct = Some(struct_name.clone());
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }
        visit::visit_item_struct(self, node);
        self.current_struct = None;
    }
    fn visit_item_enum(&mut self, node: &'a syn::ItemEnum) {
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }
        visit::visit_item_enum(self, node);
    }
    fn visit_item_fn(&mut self, node: &'a syn::ItemFn) {
        self.scoped_binder.push_scope();
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }
        for input in &node.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.process_pattern_bindings(&pat_type.pat);
            }
        }
        visit::visit_item_fn(self, node);
        self.scoped_binder.pop_scope();
    }
    fn visit_item_impl(&mut self, node: &'a ItemImpl) {
        let prev_impl = self.current_impl.clone();
        let type_name = match &*node.self_ty {
            syn::Type::Path(type_path) => {
                path_to_symbol(
                        &type_path.path,
                        self.module_path,
                        self.alias_graph,
                        self.symbol_table,
                    )
                    .or_else(|| Some(path_to_string(&type_path.path)))
            }
            _ => None,
        };
        if let Some(type_name) = type_name {
            self.current_impl = Some(OccurrenceImplContext { type_name });
        } else {
            self.current_impl = None;
        }
        visit::visit_item_impl(self, node);
        self.current_impl = prev_impl;
    }
    fn visit_impl_item_fn(&mut self, node: &'a ImplItemFn) {
        self.scoped_binder.push_scope();
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }
        if let Some(ctx) = &self.current_impl {
            let has_receiver = node
                .sig
                .inputs
                .iter()
                .any(|arg| matches!(arg, syn::FnArg::Receiver(_)));
            if has_receiver {
                self.scoped_binder.bind("self".to_string(), ctx.type_name.clone());
            }
        }
        for input in &node.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                self.process_pattern_bindings(&pat_type.pat);
            }
        }
        visit::visit_impl_item_fn(self, node);
        self.scoped_binder.pop_scope();
    }
    fn visit_local(&mut self, node: &'a Local) {
        self.process_pattern_bindings(&node.pat);
        if let Some(init) = &node.init {
            if let Some(inferred_type) = self.infer_expr_type(&init.expr) {
                if let Pat::Ident(pat_ident) = &node.pat {
                    let var_name = pat_ident.ident.to_string();
                    self.scoped_binder.bind(var_name, inferred_type);
                }
            }
        }
        visit::visit_local(self, node);
    }
    fn visit_arm(&mut self, node: &'a Arm) {
        self.scoped_binder.push_scope();
        self.process_pattern_bindings(&node.pat);
        visit::visit_arm(self, node);
        self.scoped_binder.pop_scope();
    }
    fn visit_expr_for_loop(&mut self, node: &'a ExprForLoop) {
        self.scoped_binder.push_scope();
        self.process_pattern_bindings(&node.pat);
        visit::visit_expr_for_loop(self, node);
        self.scoped_binder.pop_scope();
    }
    fn visit_expr_closure(&mut self, node: &'a ExprClosure) {
        self.scoped_binder.push_scope();
        for input in &node.inputs {
            self.process_pattern_bindings(input);
        }
        visit::visit_expr_closure(self, node);
        self.scoped_binder.pop_scope();
    }
    fn visit_expr_method_call(&mut self, node: &'a ExprMethodCall) {
        if let Some(receiver_type) = self.infer_expr_type(&node.receiver) {
            let method_name = node.method.to_string();
            if let Some(method_id) = self
                .scoped_binder
                .resolve_method(&receiver_type, &method_name)
            {
                self.add_occurrence(method_id, "method_call", node.method.span());
            }
        }
        visit::visit_expr_method_call(self, node);
    }
    fn visit_expr_macro(&mut self, node: &'a ExprMacro) {
        let mut collector = MacroIdentifierCollector::new();
        collector.process_macro_invocation(&node.mac);
        for (ident, span) in collector.identifiers {
            if let Some(id) = self.resolve_symbol(&ident) {
                self.add_occurrence(id, "macro_arg", span);
            }
        }
        visit::visit_expr_macro(self, node);
    }
    fn visit_path(&mut self, path: &'a syn::Path) {
        let segments: Vec<String> = path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect();
        if segments.is_empty() {
            return;
        }
        let resolver = Resolver::new(
            self.module_path,
            self.alias_graph,
            self.symbol_table,
        );
        for (idx, seg) in path.segments.iter().enumerate() {
            let prefix = &segments[..=idx];
            if let Some(symbol) = resolver.resolve_path_segments(prefix) {
                self.add_occurrence(symbol, "path", seg.ident.span());
            }
        }
        visit::visit_path(self, path);
    }
    fn visit_item_use(&mut self, node: &'a syn::ItemUse) {
        let mut prefix = Vec::new();
        if node.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        self.record_use_tree(&node.tree, &mut prefix);
        visit::visit_item_use(self, node);
    }
    fn visit_item_macro(&mut self, node: &'a ItemMacro) {
        if node.mac.path.is_ident("macro_rules") {
            for (ident, span) in extract_macro_rules_identifiers(node) {
                if let Some(id) = self.resolve_symbol(&ident) {
                    self.add_occurrence(id, "macro_body", span);
                }
            }
        } else {
            let mut collector = MacroIdentifierCollector::new();
            collector.process_macro_invocation(&node.mac);
            for (ident, span) in collector.identifiers {
                if let Some(id) = self.resolve_symbol(&ident) {
                    self.add_occurrence(id, "macro_arg", span);
                }
            }
        }
        visit::visit_item_macro(self, node);
    }
}
fn normalize_use_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
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
fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
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
impl<'a> OccurrenceVisitor<'a> {
    /// Iterative file-level walker — visits all items across nested inline mods
    /// using a heap stack, avoiding OS stack overflow from deep mod nesting.
    pub fn visit_file_items(&mut self, ast: &'a syn::File) {
        use syn::visit::Visit;
        let mut stack: Vec<&'a [syn::Item]> = vec![& ast.items];
        while let Some(items) = stack.pop() {
            for item in items {
                if let syn::Item::Mod(m) = item {
                    if let Some((_, inline_items)) = &m.content {
                        stack.push(inline_items.as_slice());
                    }
                    self.visit_item_mod(m);
                } else {
                    self.visit_item(item);
                }
            }
        }
    }
    fn resolve_symbol(&self, symbol: &str) -> Option<String> {
        let segments: Vec<String> = symbol.split("::").map(|s| s.to_string()).collect();
        let resolver = Resolver::new(
            self.module_path,
            self.alias_graph,
            self.symbol_table,
        );
        resolver.resolve_path_segments(&segments)
    }
    fn record_use_tree(&mut self, tree: &syn::UseTree, prefix: &mut Vec<String>) {
        match tree {
            syn::UseTree::Path(p) => {
                let mut next_prefix = prefix.clone();
                next_prefix.push(p.ident.to_string());
                let full = normalize_use_prefix(&next_prefix, self.module_path);
                let resolver = Resolver::new(
                    self.module_path,
                    self.alias_graph,
                    self.symbol_table,
                );
                if let Some(resolved) = resolver.resolve_path_segments(&full) {
                    self.add_occurrence(resolved, "use_path", p.ident.span());
                }
                prefix.push(p.ident.to_string());
                self.record_use_tree(&p.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                let mut next_prefix = prefix.clone();
                next_prefix.push(name.ident.to_string());
                let full = normalize_use_prefix(&next_prefix, self.module_path);
                let resolver = Resolver::new(
                    self.module_path,
                    self.alias_graph,
                    self.symbol_table,
                );
                if let Some(resolved) = resolver.resolve_path_segments(&full) {
                    self.add_occurrence(resolved, "use", name.ident.span());
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut next_prefix = prefix.clone();
                next_prefix.push(rename.ident.to_string());
                let full = normalize_use_prefix(&next_prefix, self.module_path);
                let resolver = Resolver::new(
                    self.module_path,
                    self.alias_graph,
                    self.symbol_table,
                );
                if let Some(resolved) = resolver.resolve_path_segments(&full) {
                    self.add_occurrence(resolved, "use", rename.ident.span());
                }
            }
            syn::UseTree::Group(group) => {
                for item in &group.items {
                    self.record_use_tree(item, prefix);
                }
            }
            syn::UseTree::Glob(_) => {}
        }
    }
}
