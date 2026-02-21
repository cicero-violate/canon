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


pub fn extract_derive_idents(meta_list: &syn::MetaList) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    extract_identifiers_from_tokens(&meta_list.tokens, &mut identifiers);
    identifiers
}


fn extract_identifiers_from_tokens(
    tokens: &TokenStream,
    identifiers: &mut Vec<(String, Span)>,
) {
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


pub fn extract_macro_rules_identifiers(item_macro: &ItemMacro) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    if !item_macro.mac.path.is_ident("macro_rules") {
        return identifiers;
    }
    extract_identifiers_from_tokens(&item_macro.mac.tokens, &mut identifiers);
    identifiers
}


pub fn extract_proc_macro_idents(meta_list: &syn::MetaList) -> Vec<(String, Span)> {
    let mut identifiers = Vec::new();
    if let Some(ident) = meta_list.path.get_ident() {
        identifiers.push((ident.to_string(), ident.span()));
    }
    extract_identifiers_from_tokens(&meta_list.tokens, &mut identifiers);
    identifiers
}


fn is_macro_keyword(ident: &str) -> bool {
    matches!(
        ident, "tt" | "ident" | "path" | "expr" | "ty" | "pat" | "stmt" | "block" |
        "item" | "meta" | "vis" | "lifetime" | "literal"
    )
}


fn is_metavariable(ident: &str) -> bool {
    ident.starts_with('$')
}


pub fn is_unsupported_macro(macro_path: &syn::Path) -> bool {
    if let Some(ident) = macro_path.get_ident() {
        let name = ident.to_string();
        matches!(
            name.as_str(), "include" | "include_str" | "include_bytes" | "concat" |
            "stringify" | "file" | "line" | "column" | "module_path"
        )
    } else {
        false
    }
}
