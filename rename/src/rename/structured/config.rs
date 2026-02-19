#[derive(Clone, Debug)]
pub struct StructuredEditOptions {
    doc_literals: bool,
    attr_literals: bool,
    use_statements: bool,
}
impl StructuredEditOptions {
    pub fn new(doc_literals: bool, attr_literals: bool, use_statements: bool) -> Self {
        Self {
            doc_literals,
            attr_literals,
            use_statements,
        }
    }
    pub fn disabled() -> Self {
        Self::new(false, false, false)
    }
    pub fn are_all_passes_enabled() -> Self {
        Self::new(true, true, true)
    }
    pub fn is_enabled(&self) -> bool {
        self.doc_literals || self.attr_literals || self.use_statements
    }
    pub fn doc_literals_enabled(&self) -> bool {
        self.doc_literals
    }
    pub fn attr_literals_enabled(&self) -> bool {
        self.attr_literals
    }
    pub fn doc_or_attr_enabled(&self) -> bool {
        self.doc_literals || self.attr_literals
    }
    pub fn use_statements_enabled(&self) -> bool {
        self.use_statements
    }
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if self.doc_literals {
            parts.push("docs");
        }
        if self.attr_literals {
            parts.push("attrs");
        }
        if self.use_statements {
            parts.push("uses");
        }
        if parts.is_empty() { "none".to_string() } else { parts.join("+") }
    }
}
pub fn are_structured_edits_enabled() -> bool {
    structured_edit_config().is_enabled()
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
fn env_flag(key: &str, default: bool) -> bool {
    match std::env::var(key) {
        Ok(value) => {
            let v = value.to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "disable")
        }
        Err(_) => default,
    }
}
