impl ScopeFrame {
    pub(crate) fn new(parent: Option<usize>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
        }
    }
}


impl ScopeFrame {
    pub(crate) fn new(parent: Option<usize>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
        }
    }
}


impl ScopeFrame {
    pub(crate) fn new(parent: Option<usize>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
        }
    }
}


impl ScopeFrame {
    pub(crate) fn new(parent: Option<usize>) -> Self {
        Self {
            bindings: HashMap::new(),
            parent,
        }
    }
}


/// A scope in the binding hierarchy
#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}


/// A scope in the binding hierarchy
#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}


/// A scope in the binding hierarchy
#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}


/// A scope in the binding hierarchy
#[derive(Debug, Clone)]
pub struct ScopeFrame {
    /// Bindings local to this scope (variable name -> type)
    pub(crate) bindings: HashMap<String, String>,
    /// Parent scope index (None for root scope)
    pub(crate) parent: Option<usize>,
}
