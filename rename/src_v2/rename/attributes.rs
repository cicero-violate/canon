//! Visitor for extracting symbol references from attributes and doc comments

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{Attribute, Meta};

/// Extract potential symbol references from attributes
pub fn extract_symbols_from_attributes(attrs: &[Attribute]) -> Vec<(String, Span)> {
    let mut symbols = Vec::new();

    for attr in attrs {
        match &attr.meta {
            Meta::Path(path) => {
                // Simple path attribute: #[derive(Clone)]
                if let Some(ident) = path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                } else {
                    // Multi-segment path: #[derive(serde::Serialize)]
                    for segment in &path.segments {
                        symbols.push((segment.ident.to_string(), segment.ident.span()));
                    }
                }
            }
            Meta::List(meta_list) => {
                // List attribute: #[derive(Clone, Debug)]
                // Parse the path
                if let Some(ident) = meta_list.path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                }

                // Parse tokens inside the list
                // This is simplified - full implementation would parse nested tokens
                let tokens_str = meta_list.tokens.to_string();
                extract_identifiers_from_token_string(
                    &tokens_str,
                    &mut symbols,
                    meta_list.tokens.span(),
                );
            }
            Meta::NameValue(meta_name_value) => {
                // Name-value attribute: #[doc = "..."]
                if let Some(ident) = meta_name_value.path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                }

                // Extract references from doc comments
                if meta_name_value.path.is_ident("doc") {
                    if let syn::Expr::Lit(expr_lit) = &meta_name_value.value {
                        if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                            let doc_text = lit_str.value();
                            extract_doc_references(&doc_text, &mut symbols, lit_str.span());
                        }
                    }
                }
            }
        }
    }

    symbols
}

/// Extract identifiers from a token string (simplified heuristic)
fn extract_identifiers_from_token_string(
    tokens: &str,
    symbols: &mut Vec<(String, Span)>,
    span: Span,
) {
    // Split by common delimiters and extract valid identifiers
    for word in tokens.split(&[' ', ',', '(', ')', '[', ']', '{', '}', '<', '>', ':', ';']) {
        let trimmed = word.trim();
        if !trimmed.is_empty() && is_valid_ident_char(trimmed) {
            symbols.push((trimmed.to_string(), span));
        }
    }
}

/// Check if a string could be a valid identifier
fn is_valid_ident_char(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let first = s.chars().next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }

    s.chars()
        .all(|c| c.is_alphanumeric() || c == '_' || c == ':')
}

/// Extract symbol references from doc comment text
/// Looks for markdown links like [`Foo`], [`Foo::bar`], [link text](Foo::bar)
fn extract_doc_references(doc_text: &str, symbols: &mut Vec<(String, Span)>, span: Span) {
    // Find backtick-quoted identifiers: `Foo`, `Foo::bar`
    let mut chars = doc_text.chars().peekable();
    let mut in_backtick = false;
    let mut current_ref = String::new();

    while let Some(ch) = chars.next() {
        if ch == '`' {
            if in_backtick {
                // End of backtick section
                if !current_ref.is_empty() && is_likely_symbol_ref(&current_ref) {
                    symbols.push((current_ref.clone(), span));
                }
                current_ref.clear();
                in_backtick = false;
            } else {
                // Start of backtick section
                in_backtick = true;
            }
        } else if in_backtick {
            current_ref.push(ch);
        }
    }

    // Also look for bracket references: [Foo], [Foo::bar]
    extract_bracket_references(doc_text, symbols, span);
}

/// Extract references from [bracket] syntax
fn extract_bracket_references(doc_text: &str, symbols: &mut Vec<(String, Span)>, span: Span) {
    let mut chars = doc_text.chars().peekable();
    let mut in_bracket = false;
    let mut current_ref = String::new();

    while let Some(ch) = chars.next() {
        if ch == '[' && chars.peek() != Some(&'`') {
            in_bracket = true;
            current_ref.clear();
        } else if ch == ']' && in_bracket {
            if !current_ref.is_empty() && is_likely_symbol_ref(&current_ref) {
                symbols.push((current_ref.clone(), span));
            }
            current_ref.clear();
            in_bracket = false;
        } else if in_bracket && ch != '`' {
            current_ref.push(ch);
        }
    }
}

/// Check if a string is likely a symbol reference (not prose)
fn is_likely_symbol_ref(s: &str) -> bool {
    let trimmed = s.trim();

    // Must start with uppercase or contain ::
    if trimmed.is_empty() {
        return false;
    }

    let first_char = trimmed.chars().next().unwrap();
    let looks_like_type = first_char.is_uppercase();
    let has_path_sep = trimmed.contains("::");

    // Exclude common words that appear in docs
    let common_words = ["the", "this", "that", "these", "those", "a", "an"];
    let is_common_word = common_words.contains(&trimmed.to_lowercase().as_str());

    (looks_like_type || has_path_sep) && !is_common_word
}
