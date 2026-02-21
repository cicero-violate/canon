fn extract_bracket_references(
    doc_text: &str,
    symbols: &mut Vec<(String, Span)>,
    span: Span,
) {
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


fn extract_doc_references(
    doc_text: &str,
    symbols: &mut Vec<(String, Span)>,
    span: Span,
) {
    let mut chars = doc_text.chars().peekable();
    let mut in_backtick = false;
    let mut current_ref = String::new();
    while let Some(ch) = chars.next() {
        if ch == '`' {
            if in_backtick {
                if !current_ref.is_empty() && is_likely_symbol_ref(&current_ref) {
                    symbols.push((current_ref.clone(), span));
                }
                current_ref.clear();
                in_backtick = false;
            } else {
                in_backtick = true;
            }
        } else if in_backtick {
            current_ref.push(ch);
        }
    }
    extract_bracket_references(doc_text, symbols, span);
}


fn extract_identifiers_from_token_string(
    tokens: &str,
    symbols: &mut Vec<(String, Span)>,
    span: Span,
) {
    for word in tokens
        .split(&[' ', ',', '(', ')', '[', ']', '{', '}', '<', '>', ':', ';'])
    {
        let trimmed = word.trim();
        if !trimmed.is_empty() && is_valid_ident_char(trimmed) {
            symbols.push((trimmed.to_string(), span));
        }
    }
}


pub fn extract_symbols_from_attributes(attrs: &[Attribute]) -> Vec<(String, Span)> {
    let mut symbols = Vec::new();
    for attr in attrs {
        match &attr.meta {
            Meta::Path(path) => {
                if let Some(ident) = path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                } else {
                    for segment in &path.segments {
                        symbols.push((segment.ident.to_string(), segment.ident.span()));
                    }
                }
            }
            Meta::List(meta_list) => {
                if let Some(ident) = meta_list.path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                }
                let tokens_str = meta_list.tokens.to_string();
                extract_identifiers_from_token_string(
                    &tokens_str,
                    &mut symbols,
                    meta_list.tokens.span(),
                );
            }
            Meta::NameValue(meta_name_value) => {
                if meta_name_value.path.is_ident("doc") {
                    continue;
                }
                if let Some(ident) = meta_name_value.path.get_ident() {
                    symbols.push((ident.to_string(), ident.span()));
                }
            }
        }
    }
    symbols
}


fn is_likely_symbol_ref(s: &str) -> bool {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return false;
    }
    let first_char = trimmed.chars().next().unwrap();
    let looks_like_type = first_char.is_uppercase();
    let has_path_sep = trimmed.contains("::");
    let common_words = ["the", "this", "that", "these", "those", "a", "an"];
    let is_common_word = common_words.contains(&trimmed.to_lowercase().as_str());
    (looks_like_type || has_path_sep) && !is_common_word
}


fn is_valid_ident_char(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == ':')
}
