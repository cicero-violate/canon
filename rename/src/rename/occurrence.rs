//! Enhanced occurrence tracking with pattern destructuring and attribute support

use proc_macro2::Span;
use std::path::Path;
use syn::visit::{self, Visit};
use syn::{Arm, Expr, ExprClosure, ExprForLoop, ExprMethodCall, ImplItemFn, ItemImpl, Local, Pat};

use super::attributes::extract_symbols_from_attributes;
use super::core::{span_to_range, OccurrenceEntry, SymbolTable};
use super::macros::{extract_macro_rules_identifiers, MacroIdentifierCollector};
use super::pattern::{extract_type_from_pattern, PatternBindingCollector};
use super::scope::ScopedBinder;
use std::collections::HashMap;
use syn::ItemMacro;

/// Enhanced occurrence visitor with full pattern and attribute support
pub struct EnhancedOccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolTable,
    use_map: &'a HashMap<String, String>,
    occurrences: &'a mut Vec<OccurrenceEntry>,
    scoped_binder: ScopedBinder,
    current_impl: Option<ImplContext>,
    current_struct: Option<String>,
}

#[derive(Clone)]
struct ImplContext {
    type_name: String,
    trait_name: Option<String>,
}

impl<'a> EnhancedOccurrenceVisitor<'a> {
    pub fn new(
        module_path: &'a str,
        file: &'a Path,
        symbol_table: &'a SymbolTable,
        use_map: &'a HashMap<String, String>,
        occurrences: &'a mut Vec<OccurrenceEntry>,
    ) -> Self {
        Self {
            module_path,
            file,
            symbol_table,
            use_map,
            occurrences,
            scoped_binder: ScopedBinder::new(symbol_table),
            current_impl: None,
            current_struct: None,
        }
    }

    fn add_occurrence(&mut self, id: String, kind: &str, span: Span) {
        self.occurrences.push(OccurrenceEntry {
            id,
            file: self.file.to_string_lossy().to_string(),
            kind: kind.to_string(),
            span: span_to_range(span),
        });
    }

    /// Process pattern bindings and add to scope
    fn process_pattern_bindings(&mut self, pat: &Pat) {
        let bindings = PatternBindingCollector::collect_from_pattern(pat);

        // Try to extract type hint from pattern
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
                // Try to infer return type of method
                if let Some(receiver_type) = self.infer_expr_type(&method_call.receiver) {
                    let method_name = method_call.method.to_string();
                    return self
                        .scoped_binder
                        .get_method_return_type(&receiver_type, &method_name);
                }
                None
            }
            Expr::Struct(expr_struct) => {
                // Struct construction - try to resolve to fully qualified path
                if let Some(symbol) = path_to_symbol(
                    &expr_struct.path,
                    self.module_path,
                    self.use_map,
                    self.symbol_table,
                ) {
                    Some(symbol)
                } else {
                    Some(path_to_string(&expr_struct.path))
                }
            }
            Expr::Call(call_expr) => {
                // Function/associated function call
                // Try to infer return type from the function being called
                if let Expr::Path(ref path_expr) = *call_expr.func {
                    // Resolve the function path to get fully qualified name
                    if let Some(func_symbol) = path_to_symbol(
                        &path_expr.path,
                        self.module_path,
                        self.use_map,
                        self.symbol_table,
                    ) {
                        // Heuristic: Constructor functions (Type::new, Type::default, etc)
                        // typically return the type
                        if let Some(last_segment) = func_symbol.rsplit("::").next() {
                            if last_segment == "new" || last_segment == "default" {
                                // Return the type (everything before ::function_name)
                                let parts: Vec<&str> = func_symbol.rsplitn(2, "::").collect();
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

impl<'a> Visit<'a> for EnhancedOccurrenceVisitor<'a> {
    fn visit_item_struct(&mut self, node: &'a syn::ItemStruct) {
        // Track we're in a struct for self references
        let struct_name = format!("{}::{}", self.module_path, node.ident);
        self.current_struct = Some(struct_name.clone());

        // Visit attributes
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }

        visit::visit_item_struct(self, node);
        self.current_struct = None;
    }

    fn visit_item_enum(&mut self, node: &'a syn::ItemEnum) {
        // Visit attributes
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }

        visit::visit_item_enum(self, node);
    }

    fn visit_item_fn(&mut self, node: &'a syn::ItemFn) {
        // Enter new scope for function
        self.scoped_binder.push_scope();

        // Visit attributes
        for (symbol, span) in extract_symbols_from_attributes(&node.attrs) {
            if let Some(id) = self.resolve_symbol(&symbol) {
                self.add_occurrence(id, "attribute", span);
            }
        }

        // Bind function parameters
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
            syn::Type::Path(type_path) => path_to_symbol(
                &type_path.path,
                self.module_path,
                self.use_map,
                self.symbol_table,
            )
            .or_else(|| Some(path_to_string(&type_path.path))),
            _ => None,
        };
        let trait_name = node.trait_.as_ref().and_then(|(_, path, _)| {
            path_to_symbol(path, self.module_path, self.use_map, self.symbol_table)
                .or_else(|| Some(path_to_string(path)))
        });
        if let Some(type_name) = type_name {
            self.current_impl = Some(ImplContext {
                type_name,
                trait_name,
            });
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
                self.scoped_binder
                    .bind("self".to_string(), ctx.type_name.clone());
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
        // Handle let bindings with pattern destructuring
        self.process_pattern_bindings(&node.pat);

        // If there's an initializer, try to infer its type and update bindings
        if let Some(init) = &node.init {
            if let Some(inferred_type) = self.infer_expr_type(&init.expr) {
                // Update binding with inferred type
                // (PatternBindingCollector already added the binding)
                if let Pat::Ident(pat_ident) = &node.pat {
                    let var_name = pat_ident.ident.to_string();
                    self.scoped_binder.bind(var_name, inferred_type);
                }
            }
        }

        visit::visit_local(self, node);
    }

    fn visit_arm(&mut self, node: &'a Arm) {
        // Enter new scope for match arm
        self.scoped_binder.push_scope();

        // Process pattern bindings in the match pattern
        self.process_pattern_bindings(&node.pat);

        visit::visit_arm(self, node);
        self.scoped_binder.pop_scope();
    }

    fn visit_expr_for_loop(&mut self, node: &'a ExprForLoop) {
        // Enter new scope for loop
        self.scoped_binder.push_scope();

        // Process pattern bindings in the for pattern
        self.process_pattern_bindings(&node.pat);

        visit::visit_expr_for_loop(self, node);
        self.scoped_binder.pop_scope();
    }

    fn visit_expr_closure(&mut self, node: &'a ExprClosure) {
        // Enter new scope for closure
        self.scoped_binder.push_scope();

        // Process closure parameter patterns
        for input in &node.inputs {
            self.process_pattern_bindings(input);
        }

        visit::visit_expr_closure(self, node);
        self.scoped_binder.pop_scope();
    }

    fn visit_expr_method_call(&mut self, node: &'a ExprMethodCall) {
        // Try to resolve the method call
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

    fn visit_path(&mut self, path: &'a syn::Path) {
        // Resolve path to symbol
        if let Some(symbol) =
            path_to_symbol(path, self.module_path, self.use_map, self.symbol_table)
        {
            if let Some(ident) = path.segments.last() {
                self.add_occurrence(symbol, "path", ident.ident.span());
            }
        }

        visit::visit_path(self, path);
    }

    fn visit_item_use(&mut self, node: &'a syn::ItemUse) {
        // Track use statements for renaming imported symbols
        let mut prefix = Vec::new();
        if node.leading_colon.is_some() {
            prefix.push("crate".to_string());
        }
        self.record_use_tree(&node.tree, &mut prefix);
        visit::visit_item_use(self, node);
    }

    fn visit_item_macro(&mut self, node: &'a ItemMacro) {
        // Handle macro_rules! definitions
        if node.mac.path.is_ident("macro_rules") {
            // Extract identifiers from macro body
            for (ident, span) in extract_macro_rules_identifiers(node) {
                if let Some(id) = self.resolve_symbol(&ident) {
                    self.add_occurrence(id, "macro_body", span);
                }
            }
        } else {
            // Handle macro invocations
            // Extract identifiers from macro arguments
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

// Helper functions

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
    let mut module_parts: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
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
    path.segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    use_map: &HashMap<String, String>,
    symbol_table: &SymbolTable,
) -> Option<String> {
    let path_str = path_to_string(path);

    // Try direct lookup
    let full_path = format!("{}::{}", module_path, path_str);
    if symbol_table.symbols.contains_key(&full_path) {
        return Some(full_path);
    }

    // Try use map
    if let Some(resolved) = use_map.get(&path_str) {
        return Some(resolved.clone());
    }

    None
}

impl<'a> EnhancedOccurrenceVisitor<'a> {
    fn resolve_symbol(&self, symbol: &str) -> Option<String> {
        // Try direct lookup
        let full_path = format!("{}::{}", self.module_path, symbol);
        if self.symbol_table.symbols.contains_key(&full_path) {
            return Some(full_path);
        }

        // Try use map
        if let Some(resolved) = self.use_map.get(symbol) {
            return Some(resolved.clone());
        }

        None
    }

    fn record_use_tree(&mut self, tree: &syn::UseTree, prefix: &mut Vec<String>) {
        match tree {
            syn::UseTree::Path(p) => {
                // Track the path segment itself as an occurrence
                let mut full = normalize_use_prefix(prefix, self.module_path);
                full.push(p.ident.to_string());
                let id = full.join("::");
                if self.symbol_table.symbols.contains_key(&id) {
                    self.add_occurrence(id, "use_path", p.ident.span());
                }

                prefix.push(p.ident.to_string());
                self.record_use_tree(&p.tree, prefix);
                prefix.pop();
            }
            syn::UseTree::Name(name) => {
                let mut full = normalize_use_prefix(prefix, self.module_path);
                full.push(name.ident.to_string());
                let id = full.join("::");
                if self.symbol_table.symbols.contains_key(&id) {
                    self.add_occurrence(id, "use", name.ident.span());
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut full = normalize_use_prefix(prefix, self.module_path);
                full.push(rename.ident.to_string());
                let id = full.join("::");
                if self.symbol_table.symbols.contains_key(&id) {
                    self.add_occurrence(id, "use", rename.rename.span());
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
