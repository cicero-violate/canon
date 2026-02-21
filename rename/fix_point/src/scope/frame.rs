impl ScopeFrame {
    pub(crate) fn new(parent: Option<usize>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
        }
    }
}
