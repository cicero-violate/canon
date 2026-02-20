//! Pattern visitor for extracting bindings from destructuring patterns

use syn::visit::Visit;
use syn::{Pat, PatIdent, PatSlice, PatStruct, PatTuple, PatTupleStruct};

/// Visitor that extracts variable bindings from patterns
pub struct PatternBindingCollector {
    /// Collected bindings (variable name, optional type hint)
    pub bindings: Vec<(String, Option<String>)>,
}

impl PatternBindingCollector {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
    }

    /// Collect bindings from a pattern
    pub fn collect_from_pattern(pat: &Pat) -> Vec<(String, Option<String>)> {
        let mut collector = Self::new();
        collector.visit_pat(pat);
        collector.bindings
    }
}

impl<'ast> Visit<'ast> for PatternBindingCollector {
    fn visit_pat_ident(&mut self, node: &'ast PatIdent) {
        // Extract identifier binding
        let name = node.ident.to_string();

        // Skip _ patterns (wildcard)
        if name != "_" {
            self.bindings.push((name, None));
        }

        // Continue visiting sub-patterns (e.g., in ref patterns)
        if let Some((_, subpat)) = &node.subpat {
            self.visit_pat(subpat);
        }
    }

    fn visit_pat_tuple(&mut self, node: &'ast PatTuple) {
        // Visit all elements in tuple pattern
        for elem in &node.elems {
            self.visit_pat(elem);
        }
    }

    fn visit_pat_struct(&mut self, node: &'ast PatStruct) {
        // Visit all field patterns in struct pattern
        for field_pat in &node.fields {
            self.visit_pat(&field_pat.pat);
        }

        // Handle rest pattern (..)
        // Rest patterns don't bind variables
    }

    fn visit_pat_tuple_struct(&mut self, node: &'ast PatTupleStruct) {
        // Visit all elements in tuple struct pattern
        for elem in &node.elems {
            self.visit_pat(elem);
        }
    }

    fn visit_pat_slice(&mut self, node: &'ast PatSlice) {
        // Visit all elements in slice pattern
        for elem in &node.elems {
            self.visit_pat(elem);
        }
    }
}

/// Extract type hint from a pattern if available
pub fn extract_type_from_pattern(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Type(pat_type) => {
            // Pattern with explicit type annotation: x: Foo
            Some(type_to_string(&pat_type.ty))
        }
        _ => None,
    }
}

/// Convert a syn::Type to a string representation
fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => path_to_string(&type_path.path),
        syn::Type::Reference(type_ref) => {
            format!("&{}", type_to_string(&type_ref.elem))
        }
        syn::Type::Tuple(type_tuple) => {
            let elems: Vec<String> = type_tuple.elems.iter().map(type_to_string).collect();
            format!("({})", elems.join(", "))
        }
        syn::Type::Slice(type_slice) => {
            format!("[{}]", type_to_string(&type_slice.elem))
        }
        _ => "Unknown".to_string(),
    }
}

/// Convert a syn::Path to a string
fn path_to_string(path: &syn::Path) -> String {
    path.segments.iter().map(|seg| seg.ident.to_string()).collect::<Vec<_>>().join("::")
}
