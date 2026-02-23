use crate::alias::AliasGraph;


use crate::model::types::SymbolIndex;


use std::sync::Arc;


use std::collections::HashMap;


use std::cell::RefCell;


pub struct Resolver<'a> {
    pub module_path: &'a str,
    pub alias_graph: &'a AliasGraph,
    pub symbol_table: &'a SymbolIndex,
    cache: RefCell<HashMap<String, Option<String>>>,
}


#[derive(Clone)]
pub struct ResolverContext {
    pub module_path: String,
    pub alias_graph: Arc<AliasGraph>,
    pub symbol_table: Arc<SymbolIndex>,
}


fn split_module_and_name(path: &str) -> Option<(&str, &str)> {
    let (mod_path, name) = path.rsplit_once("::")?;
    Some((mod_path, name))
}
