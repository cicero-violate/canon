use anyhow::Result;


use std::collections::HashMap;


use std::path::Path;


use syn::visit_mut::VisitMut;


use crate::model::span::{span_to_range, SpanRange};


use super::config::StructuredEditOptions;


use super::orchestrator::StructuredPass;


struct AttributeRewriteVisitor {
    replacements: Vec<(String, String)>,
    rewrite_docs: bool,
    rewrite_attrs: bool,
    result: StructuredAttributeResult,
}


pub struct DocAttrPass {
    mapping: HashMap<String, String>,
    config: StructuredEditOptions,
}


pub struct StructuredAttributeResult {
    pub literal_spans: Vec<SpanRange>,
    pub changed: bool,
}


fn build_replacements(mapping: &HashMap<String, String>) -> Vec<(String, String)> {
    let mut replacements = Vec::new();
    for (old_id, new_name) in mapping {
        if let Some(tail) = old_id.rsplit("::").next() {
            if tail != new_name && !tail.is_empty() {
                replacements.push((tail.to_string(), new_name.clone()));
            }
        }
    }
    replacements
}


fn contains(outer: &SpanRange, inner: &SpanRange) -> bool {
    if inner.start.line < outer.start.line || inner.end.line > outer.end.line {
        return false;
    }
    if outer.start.line == inner.start.line && inner.start.column < outer.start.column {
        return false;
    }
    if outer.end.line == inner.end.line && inner.end.column > outer.end.column {
        return false;
    }
    true
}


fn is_boundary(text: &str, start: usize, end: usize) -> bool {
    let prev = text[..start].chars().rev().next();
    let next = text[end..].chars().next();
    let prev_ok = !matches!(prev, Some(c) if c.is_ascii_alphanumeric() || c == '_');
    let next_ok = !matches!(next, Some(c) if c.is_ascii_alphanumeric() || c == '_');
    prev_ok && next_ok
}


fn replace_identifier(text: &str, old: &str, new_name: &str) -> Option<String> {
    let mut result = String::new();
    let mut cursor = 0usize;
    let mut changed = false;
    while let Some(rel_pos) = text[cursor..].find(old) {
        let start = cursor + rel_pos;
        let end = start + old.len();
        if is_boundary(text, start, end) {
            changed = true;
            result.push_str(&text[cursor..start]);
            result.push_str(new_name);
            cursor = end;
        } else {
            result.push_str(&text[cursor..end]);
            cursor = end;
        }
    }
    result.push_str(&text[cursor..]);
    if changed { Some(result) } else { None }
}


pub fn rewrite_doc_and_attr_literals(
    _file: &Path,
    _content: &str,
    ast: &mut syn::File,
    mapping: &HashMap<String, String>,
    config: &StructuredEditOptions,
) -> Result<StructuredAttributeResult> {
    let mut visitor = AttributeRewriteVisitor::new(mapping, config);
    visitor.visit_file_mut(ast);
    Ok(visitor.finish())
}


fn rewrite_literal(value: &str, replacements: &[(String, String)]) -> Option<String> {
    let mut updated = value.to_string();
    let mut changed = false;
    for (old, new_name) in replacements {
        if let Some(next) = replace_identifier(&updated, old, new_name) {
            updated = next;
            changed = true;
        }
    }
    if changed { Some(updated) } else { None }
}
