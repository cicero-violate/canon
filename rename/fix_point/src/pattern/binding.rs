pub struct PatternBindingCollector {
    /// Collected bindings (variable name, optional type hint)
    pub bindings: Vec<(String, Option<String>)>,
}
