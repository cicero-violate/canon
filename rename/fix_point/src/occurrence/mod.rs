mod visitor;

pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
}


/// Enhanced occurrence visitor with full pattern and attribute support
pub struct OccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    alias_graph: &'a AliasGraph,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    scoped_binder: ScopeBinder,
    current_impl: Option<ImplCtx>,
    current_struct: Option<String>,
}


pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
}


/// Enhanced occurrence visitor with full pattern and attribute support
pub struct OccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    alias_graph: &'a AliasGraph,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    scoped_binder: ScopeBinder,
    current_impl: Option<ImplCtx>,
    current_struct: Option<String>,
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


pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
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


/// Enhanced occurrence visitor with full pattern and attribute support
pub struct OccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    alias_graph: &'a AliasGraph,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    scoped_binder: ScopeBinder,
    current_impl: Option<ImplCtx>,
    current_struct: Option<String>,
}


pub fn file_contains_symbol(source: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return false;
    }
    !kmp_search(source, symbol).is_empty()
}


fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let resolver = Resolver::new(module_path, alias_graph, symbol_table);
    resolver.resolve_path_segments(&segments)
}


/// Enhanced occurrence visitor with full pattern and attribute support
pub struct OccurrenceVisitor<'a> {
    module_path: &'a str,
    file: &'a Path,
    symbol_table: &'a SymbolIndex,
    use_map: &'a HashMap<String, String>,
    alias_graph: &'a AliasGraph,
    occurrences: &'a mut Vec<SymbolOccurrence>,
    scoped_binder: ScopeBinder,
    current_impl: Option<ImplCtx>,
    current_struct: Option<String>,
}


fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let resolver = Resolver::new(module_path, alias_graph, symbol_table);
    resolver.resolve_path_segments(&segments)
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


fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let resolver = Resolver::new(module_path, alias_graph, symbol_table);
    resolver.resolve_path_segments(&segments)
}


fn path_to_symbol(
    path: &syn::Path,
    module_path: &str,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
) -> Option<String> {
    let segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let resolver = Resolver::new(module_path, alias_graph, symbol_table);
    resolver.resolve_path_segments(&segments)
}


#[derive(Clone)]
struct ImplCtx {
    type_name: String,
}


#[derive(Clone)]
struct ImplCtx {
    type_name: String,
}


#[derive(Clone)]
struct ImplCtx {
    type_name: String,
}


#[derive(Clone)]
struct ImplCtx {
    type_name: String,
}
