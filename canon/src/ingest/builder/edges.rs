use std::collections::{BTreeSet, HashMap};

use syn::visit::{self, Visit};

use crate::ir::{CallEdge, Function, Visibility};

use super::super::parser::ParsedWorkspace;
use super::modules::{collect_use_aliases, module_key};
use super::types::{AliasBinding, UseEntry, module_segments_from_key, slugify};

// ── Shared data structures ────────────────────────────────────────────────────

#[derive(Clone)]
pub(crate) struct FunctionLookupEntry {
    pub id: String,
    pub visibility: Visibility,
}

// ── Call edge builder ─────────────────────────────────────────────────────────

pub(crate) fn build_call_edges(
    parsed: &ParsedWorkspace,
    module_lookup: &HashMap<String, String>,
    functions: &[Function],
) -> Vec<CallEdge> {
    let mut module_reverse = HashMap::new();
    for (key, id) in module_lookup {
        module_reverse.insert(id.clone(), key.clone());
    }
    let mut function_lookup: HashMap<(String, String), FunctionLookupEntry> = HashMap::new();
    for function in functions {
        if let Some(mk) = module_reverse.get(&function.module) {
            if let Some(slug) = function_name_slug(function) {
                function_lookup.insert(
                    (mk.clone(), slug),
                    FunctionLookupEntry {
                        id: function.id.clone(),
                        visibility: function.visibility,
                    },
                );
            }
        }
    }
    let mut discovered: BTreeSet<(String, String)> = BTreeSet::new();
    for file in &parsed.files {
        let mk = module_key(file);
        let Some(_module_id) = module_lookup.get(&mk) else {
            continue;
        };
        let module_segments = module_segments_from_key(&mk);
        let alias_map = collect_use_aliases(file, &mk, module_lookup);
        for item in &file.ast.items {
            match item {
                syn::Item::Fn(item_fn) => {
                    if let Some(caller_id) =
                        lookup_function_id(&mk, &item_fn.sig.ident, &function_lookup)
                    {
                        collect_calls_in_block(
                            &item_fn.block,
                            &module_segments,
                            &caller_id,
                            &function_lookup,
                            &alias_map,
                            &mut discovered,
                        );
                    }
                }
                syn::Item::Impl(impl_block) => {
                    for impl_item in &impl_block.items {
                        if let syn::ImplItem::Fn(method) = impl_item {
                            if let Some(caller_id) =
                                lookup_function_id(&mk, &method.sig.ident, &function_lookup)
                            {
                                collect_calls_in_block(
                                    &method.block,
                                    &module_segments,
                                    &caller_id,
                                    &function_lookup,
                                    &alias_map,
                                    &mut discovered,
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    discovered
        .into_iter()
        .enumerate()
        .map(|(idx, (caller, callee))| CallEdge {
            id: format!("call_edge.{}", idx + 1),
            caller,
            callee,
            rationale: "ING-001: discovered call expression".to_owned(),
        })
        .collect()
}

pub(crate) fn lookup_function_id(
    module_key: &str,
    ident: &syn::Ident,
    functions: &HashMap<(String, String), FunctionLookupEntry>,
) -> Option<String> {
    let slug = slugify(&ident.to_string());
    functions
        .get(&(module_key.to_owned(), slug))
        .map(|entry| entry.id.clone())
}

pub(crate) fn collect_calls_in_block(
    block: &syn::Block,
    module_segments: &[String],
    caller_id: &str,
    functions: &HashMap<(String, String), FunctionLookupEntry>,
    aliases: &HashMap<String, AliasBinding>,
    discovered: &mut BTreeSet<(String, String)>,
) {
    let mut visitor = CallCollector {
        caller_id,
        module_segments,
        functions,
        aliases,
        discovered,
    };
    visitor.visit_block(block);
}

struct CallCollector<'a> {
    caller_id: &'a str,
    module_segments: &'a [String],
    functions: &'a HashMap<(String, String), FunctionLookupEntry>,
    aliases: &'a HashMap<String, AliasBinding>,
    discovered: &'a mut BTreeSet<(String, String)>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_call(&mut self, node: &'ast syn::ExprCall) {
        if let Some(path) = extract_call_path(&node.func) {
            if let Some(entry) = resolve_function_path(self.module_segments, path, self.functions) {
                if entry.visibility == Visibility::Public && entry.id != self.caller_id {
                    self.discovered
                        .insert((self.caller_id.to_owned(), entry.id.clone()));
                }
            } else if path.segments.len() == 1 {
                if let Some(segment) = path.segments.first() {
                    let name = segment.ident.to_string();
                    if let Some(binding) = self.aliases.get(&name) {
                        let key = (binding.module_key.clone(), binding.function_slug.clone());
                        if let Some(entry) = self.functions.get(&key) {
                            if entry.visibility == Visibility::Public && entry.id != self.caller_id
                            {
                                self.discovered
                                    .insert((self.caller_id.to_owned(), entry.id.clone()));
                            }
                        }
                    }
                }
            }
        }
        visit::visit_expr_call(self, node);
    }
}

fn extract_call_path(expr: &syn::Expr) -> Option<&syn::Path> {
    match expr {
        syn::Expr::Path(p) => Some(&p.path),
        syn::Expr::Paren(paren) => extract_call_path(&paren.expr),
        syn::Expr::Group(group) => extract_call_path(&group.expr),
        syn::Expr::Reference(reference) => extract_call_path(&reference.expr),
        _ => None,
    }
}

pub(crate) fn resolve_function_path<'a>(
    module_segments: &[String],
    path: &syn::Path,
    functions: &'a HashMap<(String, String), FunctionLookupEntry>,
) -> Option<&'a FunctionLookupEntry> {
    let mut segments: Vec<String> = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect();
    if segments.is_empty() {
        return None;
    }
    let mut base = if path.leading_colon.is_some() {
        Vec::new()
    } else {
        module_segments.to_vec()
    };
    if let Some(first) = segments.first() {
        match first.as_str() {
            "crate" => {
                base.clear();
                segments.remove(0);
            }
            "self" => {
                base = module_segments.to_vec();
                segments.remove(0);
            }
            "super" => {
                base = module_segments.to_vec();
                while let Some(seg) = segments.first() {
                    if seg == "super" {
                        segments.remove(0);
                        if !base.is_empty() {
                            base.pop();
                        }
                    } else {
                        break;
                    }
                }
            }
            _ => {}
        }
    }
    base.extend(segments);
    if base.is_empty() {
        return None;
    }
    let fn_name = base.pop()?;
    let module_key = if base.is_empty() {
        "crate".to_owned()
    } else {
        base.join("::")
    };
    let slug = slugify(&fn_name);
    functions.get(&(module_key, slug))
}

pub(crate) fn function_name_slug(function: &Function) -> Option<String> {
    function.id.rsplit('.').next().map(|seg| seg.to_owned())
}
