pub struct StructuredEditOptions {
    doc_literals: bool,
    attr_literals: bool,
    use_statements: bool,
}


pub fn are_structured_edits_enabled() -> bool {
    structured_edit_config().is_enabled()
}


fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => {
            let v = value.to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "disable")
        }
        Err(_) => default,
    }
}


pub fn structured_edit_config() -> StructuredEditOptions {
    let base_enabled = env_flag("SEMANTIC_LINT_STRUCTURED_EDITS", true);
    if !base_enabled {
        return StructuredEditOptions::disabled();
    }
    let doc_literals = env_flag("SEMANTIC_LINT_STRUCTURED_DOCS", true);
    let attr_literals = env_flag("SEMANTIC_LINT_STRUCTURED_ATTRS", true);
    let use_statements = env_flag("SEMANTIC_LINT_STRUCTURED_USES", true);
    StructuredEditOptions::new(
        base_enabled && doc_literals,
        base_enabled && attr_literals,
        base_enabled && use_statements,
    )
}
