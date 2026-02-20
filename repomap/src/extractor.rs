use crate::symbol::Symbol;
use tree_sitter::{Node, Parser};

/// Extract a text slice from source bytes given a node.
fn node_text<'a>(node: Node, src: &'a [u8]) -> &'a str {
    node.utf8_text(src).unwrap_or("")
}

/// Find a direct named child with a given field name and return its text.
fn field_text<'a>(node: Node, field: &str, src: &'a [u8]) -> Option<&'a str> {
    node.child_by_field_name(field).map(|n| node_text(n, src))
}

/// Extract the bare function signature: `fn name(params) -> ret`
/// without the body block.
fn fn_signature(node: Node, src: &[u8]) -> String {
    let name = field_text(node, "name", src).unwrap_or("?");

    // parameters node
    let params = node.child_by_field_name("parameters").map(|n| node_text(n, src)).unwrap_or("()");

    // optional return type
    let ret = node.child_by_field_name("return_type").map(|n| format!(" -> {}", node_text(n, src))).unwrap_or_default();

    format!("fn {}{}{}", name, params, ret)
}

/// Collect method signatures inside a trait or impl body.
fn collect_methods(body: Node, src: &[u8]) -> Vec<String> {
    let mut methods = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "function_item" {
            methods.push(fn_signature(child, src));
        }
    }
    methods
}

/// Collect struct field names from a field_declaration_list node.
fn collect_struct_fields(body: Node, src: &[u8]) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "field_declaration" {
            if let Some(name_node) = child.child_by_field_name("name") {
                fields.push(node_text(name_node, src).to_string());
            }
        }
    }
    fields
}

/// Collect enum variant names from an enum_variant_list node.
fn collect_enum_variants(body: Node, src: &[u8]) -> Vec<String> {
    let mut variants = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "enum_variant" {
            if let Some(name_node) = child.child_by_field_name("name") {
                variants.push(node_text(name_node, src).to_string());
            }
        }
    }
    variants
}

/// Walk the top-level nodes of a source file and extract symbols.
fn extract_top_level(root: Node, src: &[u8]) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    let mut cursor = root.walk();

    for node in root.children(&mut cursor) {
        let line = node.start_position().row + 1; // 1-indexed

        match node.kind() {
            "struct_item" => {
                let name = field_text(node, "name", src).unwrap_or("?").to_string();
                let fields = node.child_by_field_name("body").map(|b| collect_struct_fields(b, src)).unwrap_or_default();
                symbols.push(Symbol::Struct { name, fields, line });
            }

            "enum_item" => {
                let name = field_text(node, "name", src).unwrap_or("?").to_string();
                let variants = node.child_by_field_name("body").map(|b| collect_enum_variants(b, src)).unwrap_or_default();
                symbols.push(Symbol::Enum { name, variants, line });
            }

            "trait_item" => {
                let name = field_text(node, "name", src).unwrap_or("?").to_string();
                let methods = node.child_by_field_name("body").map(|b| collect_methods(b, src)).unwrap_or_default();
                symbols.push(Symbol::Trait { name, methods, line });
            }

            "function_item" => {
                let name = field_text(node, "name", src).unwrap_or("?").to_string();
                let signature = fn_signature(node, src);
                symbols.push(Symbol::Function { name, signature, line });
            }

            "impl_item" => {
                // `impl Trait for Type` or plain `impl Type`
                let type_name = field_text(node, "type", src).unwrap_or("?").to_string();
                let trait_name = field_text(node, "trait", src).map(|s| s.to_string());
                let methods = node.child_by_field_name("body").map(|b| collect_methods(b, src)).unwrap_or_default();
                symbols.push(Symbol::Impl { type_name, trait_name, methods, line });
            }

            "type_item" => {
                let name = field_text(node, "name", src).unwrap_or("?").to_string();
                symbols.push(Symbol::TypeAlias { name, line });
            }

            _ => {}
        }
    }

    symbols
}

/// Parse a single Rust source string and return its symbols.
pub fn extract_symbols(src: &str) -> Vec<Symbol> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into()).expect("failed to load Rust grammar");

    let tree = parser.parse(src, None).expect("parse failed");
    extract_top_level(tree.root_node(), src.as_bytes())
}
