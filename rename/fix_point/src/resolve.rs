pub struct Resolver<'a> {
    pub module_path: &'a str,
    pub alias_graph: &'a AliasGraph,
    pub symbol_table: &'a SymbolIndex,
    cache: RefCell<HashMap<String, Option<String>>>,
}


pub struct ResolverContext {
    pub module_path: String,
    pub alias_graph: Arc<AliasGraph>,
    pub symbol_table: Arc<SymbolIndex>,
}


fn normalize_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    if prefix.first().map(|s| s.as_str()) == Some("crate") {
        return prefix.to_vec();
    }
    if prefix.first().map(|s| s.as_str()) == Some("self")
        || prefix.first().map(|s| s.as_str()) == Some("super")
    {
        return resolve_relative_prefix(prefix, module_path);
    }
    let mut out: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    out.extend(prefix.iter().cloned());
    out
}


fn resolve_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    let mut module_parts: Vec<String> = module_path
        .split("::")
        .map(|s| s.to_string())
        .collect();
    let mut idx = 0usize;
    while idx < prefix.len() && prefix[idx] == "super" {
        if module_parts.len() > 1 {
            module_parts.pop();
        }
        idx += 1;
    }
    if idx < prefix.len() && prefix[idx] == "self" {
        idx += 1;
    }
    module_parts.extend(prefix[idx..].iter().cloned());
    module_parts
}


fn split_module_and_name(path: &str) -> Option<(&str, &str)> {
    let (mod_path, name) = path.rsplit_once("::")?;
    Some((mod_path, name))
}
