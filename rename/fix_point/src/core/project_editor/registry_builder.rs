#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


#[derive(Clone)]
struct ImplContext {
    struct_path: String,
    trait_path: Option<String>,
}


struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}


struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}


type SpanLookup = std::collections::HashMap<
    PathBuf,
    std::collections::HashMap<String, SpanOverride>,
>;


struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
    source: Arc<String>,
    span_lookup: Option<&'a SpanLookup>,
    canonical_file: PathBuf,
}


type SpanLookup = std::collections::HashMap<
    PathBuf,
    std::collections::HashMap<String, SpanOverride>,
>;


struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
    source: Arc<String>,
    span_lookup: Option<&'a SpanLookup>,
    canonical_file: PathBuf,
}


struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}


struct SpanOverride {
    pub span: SpanRange,
    pub byte_range: Option<(usize, usize)>,
}


type SpanLookup = std::collections::HashMap<
    PathBuf,
    std::collections::HashMap<String, SpanOverride>,
>;


struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
    source: Arc<String>,
    span_lookup: Option<&'a SpanLookup>,
    canonical_file: PathBuf,
}


type SpanLookup = std::collections::HashMap<
    PathBuf,
    std::collections::HashMap<String, SpanOverride>,
>;


struct NodeRegistryBuilder<'a> {
    project_root: &'a Path,
    file: &'a Path,
    registry: &'a mut NodeRegistry,
    module_path: String,
    item_index: usize,
    parent_path: Vec<usize>,
    current_impl: Option<ImplContext>,
    source: Arc<String>,
    span_lookup: Option<&'a SpanLookup>,
    canonical_file: PathBuf,
}
