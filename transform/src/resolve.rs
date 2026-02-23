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

impl ResolverContext {
    pub fn resolver(&self) -> Resolver<'_> {
        Resolver {
            module_path: &self.module_path,
            alias_graph: &self.alias_graph,
            symbol_table: &self.symbol_table,
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl<'a> Resolver<'a> {
    pub fn new(
        module_path: &'a str,
        alias_graph: &'a AliasGraph,
        symbol_table: &'a SymbolIndex,
    ) -> Self {
        Self {
            module_path,
            alias_graph,
            symbol_table,
            cache: RefCell::new(HashMap::new()),
        }
    }
    pub fn resolve_path_segments(&self, segments: &[String]) -> Option<String> {
        if segments.is_empty() {
            return None;
        }

        let key = format!("{}::{}", self.module_path, segments.join("::"));

        if let Some(cached) = self.cache.borrow().get(&key) {
            return cached.clone();
        }

        let mut visited = std::collections::HashSet::new();
        let result = self.resolve_path_segments_internal(segments, &mut visited);

        self.cache.borrow_mut().insert(key, result.clone());

        result
    }

    fn resolve_path_segments_internal(
        &self,
        segments: &[String],
        visited: &mut std::collections::HashSet<String>,
    ) -> Option<String> {
        if segments.is_empty() {
            return None;
        }

        let key = format!("{}::{}", self.module_path, segments.join("::"));
        if !visited.insert(key.clone()) {
            return None;
        }

        if segments[0] == "crate" || segments[0] == "self" || segments[0] == "super" {
            let normalized = normalize_relative_prefix(segments, self.module_path);
            let abs = normalized.join("::");
            return self.resolve_via_alias_chain(&abs);
        }

        let head = &segments[0];
        let tail = &segments[1..];

        if let Some(resolved) = self
            .alias_graph
            .resolve_alias_chain(self.module_path, head)
            .resolved_symbol
        {
            let mut candidate = resolved;
            if !tail.is_empty() {
                candidate = format!("{}::{}", candidate, tail.join("::"));
            }
            if let Some(found) = self.resolve_via_alias_chain(&candidate) {
                return Some(found);
            }
        }

        let abs = format!("{}::{}", self.module_path, segments.join("::"));
        self.resolve_via_alias_chain(&abs)
    }

    fn resolve_via_alias_chain(&self, path: &str) -> Option<String> {
        if self.symbol_table.symbols.contains_key(path) {
            return Some(path.to_string());
        }
        let mut current = path.to_string();
        for _ in 0..8 {
            let Some((mod_path, name)) = split_module_and_name(&current) else {
                break;
            };
            let chain = self.alias_graph.resolve_alias_chain(mod_path, name);
            let Some(next) = chain.resolved_symbol else {
                break;
            };
            if next == current {
                break;
            }
            current = next;
            if self.symbol_table.symbols.contains_key(&current) {
                return Some(current.clone());
            }
        }
        None
    }
}

fn normalize_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    if prefix.first().map(|s| s.as_str()) == Some("crate") {
        return prefix.to_vec();
    }
    if prefix.first().map(|s| s.as_str()) == Some("self") || prefix.first().map(|s| s.as_str()) == Some("super") {
        return resolve_relative_prefix(prefix, module_path);
    }
    let mut out: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
    out.extend(prefix.iter().cloned());
    out
}

fn resolve_relative_prefix(prefix: &[String], module_path: &str) -> Vec<String> {
    let mut module_parts: Vec<String> = module_path.split("::").map(|s| s.to_string()).collect();
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
