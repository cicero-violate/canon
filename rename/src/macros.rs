//! Macro handling for identifier extraction and tracking
use proc_macro2::{Span, TokenStream, TokenTree};
use syn::ItemMacro;
/// Extract literal identifiers from macro_rules! bodies
/// This is a best-effort heuristic approach since full macro expansion is complex
pub fn extract_macro_rules_identifiers(item_macro: &ItemMacro) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    if !item_macro.mac.path.is_ident("macro_rules") {
        return identifiers;
    }
    extract_identifiers_from_tokens(&item_macro.mac.tokens, &mut identifiers);
    identifiers
}
/// Recursively extract identifiers from token stream
fn extract_identifiers_from_tokens(tokens: &TokenStream, identifiers: &mut Vec<(String, Span)>) {
    for token in tokens.clone() {
        match token {
            TokenTree::Ident(ident) => {
                let ident_str = ident.to_string();
                if !is_macro_keyword(&ident_str) && !is_metavariable(&ident_str) {
                    identifiers.push((ident_str, ident.span()));
                }
            }
            TokenTree::Group(group) => {
                extract_identifiers_from_tokens(&group.stream(), identifiers);
            }
            _ => {}
        }
    }
}
/// Check if an identifier is a macro-related keyword
fn is_macro_keyword(ident: &str) -> bool {
    matches!(ident, "tt" | "ident" | "path" | "expr" | "ty" | "pat" | "stmt" | "block" | "item" | "meta" | "vis" | "lifetime" | "literal")
}
/// Check if an identifier is a metavariable (starts with $)
fn is_metavariable(ident: &str) -> bool {
    ident.starts_with('$')
}
/// Extract identifiers from derive attribute arguments
pub fn extract_derive_idents(meta_list: &syn::MetaList) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    extract_identifiers_from_tokens(&meta_list.tokens, &mut identifiers);
    identifiers
}
/// Extract identifiers from procedural macro attributes
/// For example: #[my_macro(foo = "bar", baz)]
pub fn extract_proc_macro_idents(meta_list: &syn::MetaList) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    if let Some(ident) = meta_list.path.get_ident() {
        identifiers.push((ident.to_string(), ident.span()));
    }
    extract_identifiers_from_tokens(&meta_list.tokens, &mut identifiers);
    identifiers
}
/// Analyze a macro invocation to predict generated identifiers
/// This is highly heuristic and works best with common patterns
pub struct MacroInvocationAnalyzer {
    /// Known macro patterns that generate predictable identifiers
    patterns: Vec<MacroPattern>,
}
struct MacroPattern {
    macro_name: String,
    /// Function to predict generated identifiers based on arguments
    predict: fn(&TokenStream) -> Vec<String>,
}
impl MacroInvocationAnalyzer {
    pub fn new() -> Self {
        Self { patterns: vec![] }
    }
    pub fn predict_generated_idents(&self, macro_name: &str, tokens: &TokenStream) -> Vec<String> {
        for pattern in &self.patterns {
            if pattern.macro_name == macro_name {
                return (pattern.predict)(tokens);
            }
        }
        Vec::new()
    }
}
/// Check if a macro invocation is likely unsupported for renaming
pub fn is_unsupported_macro(macro_path: &syn::Path) -> bool {
    if let Some(ident) = macro_path.get_ident() {
        let name = ident.to_string();
        matches!(name.as_str(), "include" | "include_str" | "include_bytes" | "concat" | "stringify" | "file" | "line" | "column" | "module_path")
    } else {
        false
    }
}
/// Report for macro handling limitations
#[derive(Debug, Clone)]
pub struct MacroHandlingReport {
    pub supported_macros: usize,
    pub unsupported_macros: Vec<String>,
    pub extracted_identifiers: usize,
    pub flagged_for_review: Vec<String>,
}
impl MacroHandlingReport {
    pub fn new() -> Self {
        Self { supported_macros: 0, unsupported_macros: Vec::new(), extracted_identifiers: 0, flagged_for_review: Vec::new() }
    }
    pub fn add_unsupported(&mut self, macro_name: String) {
        self.unsupported_macros.push(macro_name);
    }
    pub fn add_flagged_reason(&mut self, reason: String) {
        self.flagged_for_review.push(reason);
    }
}
/// Visitor for collecting macro-related identifiers
pub struct MacroIdentifierCollector {
    pub identifiers: Vec<(String, Span)>,
    pub report: MacroHandlingReport,
}
impl MacroIdentifierCollector {
    pub fn new() -> Self {
        Self { identifiers: Vec::new(), report: MacroHandlingReport::new() }
    }
    /// Process a macro_rules! definition
    pub fn process_macro_rules_def(&mut self, item_macro: &ItemMacro) {
        let extracted = extract_macro_rules_identifiers(item_macro);
        self.report.extracted_identifiers += extracted.len();
        self.identifiers.extend(extracted);
    }
    /// Process a macro invocation
    pub fn process_macro_invocation(&mut self, mac: &syn::Macro) {
        if is_unsupported_macro(&mac.path) {
            if let Some(ident) = mac.path.get_ident() {
                self.report.add_unsupported(ident.to_string());
            }
        } else {
            let mut extracted = Vec::new();
            extract_identifiers_from_tokens(&mac.tokens, &mut extracted);
            self.report.extracted_identifiers += extracted.len();
            self.identifiers.extend(extracted);
        }
    }
}
