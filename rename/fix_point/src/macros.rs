use proc_macro2::{Span, TokenStream, TokenTree};


use syn::ItemMacro;


#[derive(Debug, Clone)]
pub struct MacroHandlingReport {
    pub supported_macros: usize,
    pub unsupported_macros: Vec<String>,
    pub extracted_identifiers: usize,
    pub flagged_for_review: Vec<String>,
}


pub struct MacroIdentifierCollector {
    pub identifiers: Vec<(String, Span)>,
    pub report: MacroHandlingReport,
}


pub struct MacroInvocationAnalyzer {
    /// Known macro patterns that generate predictable identifiers
    patterns: Vec<MacroPattern>,
}


struct MacroPattern {
    macro_name: String,
    /// Function to predict generated identifiers based on arguments
    predict: fn(&TokenStream) -> Vec<String>,
}
