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


struct ItemCollector<'a> {
    file: &'a Path,
    symbols: Vec<SymbolRecord>,
    alias_graph: &'a mut crate::alias::AliasGraph,
}


fn split_prefix(prefix: &str) -> Vec<String> {
    if prefix.is_empty() {
        Vec::new()
    } else {
        prefix.split("::").map(|s| s.to_string()).collect()
    }
}
