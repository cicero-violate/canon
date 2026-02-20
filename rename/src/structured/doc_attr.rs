use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use syn::visit_mut::VisitMut;

use crate::core::{span_to_range, SpanRange};

use super::config::StructuredEditOptions;
use super::orchestrator::StructuredPass;

pub struct StructuredAttributeResult {
    pub literal_spans: Vec<SpanRange>,
    pub changed: bool,
}

impl StructuredAttributeResult {
    pub fn new() -> Self {
        Self {
            literal_spans: Vec::new(),
            changed: false,
        }
    }

    pub fn should_skip(&self, span: &SpanRange) -> bool {
        self.literal_spans.iter().any(|outer| contains(outer, span))
    }
}

pub struct DocAttrPass {
    mapping: HashMap<String, String>,
    config: StructuredEditOptions,
}

impl DocAttrPass {
    pub fn new(mapping: HashMap<String, String>, config: StructuredEditOptions) -> Self {
        Self { mapping, config }
    }
}

impl StructuredPass for DocAttrPass {
    fn name(&self) -> &'static str {
        "doc_attr"
    }

    fn execute(&mut self, file: &Path, content: &str, ast: &mut syn::File) -> Result<bool> {
        let result =
            rewrite_doc_and_attr_literals(file, content, ast, &self.mapping, &self.config)?;
        Ok(result.changed)
    }

    fn is_enabled(&self) -> bool {
        self.config.doc_or_attr_enabled()
    }
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

struct AttributeRewriteVisitor {
    replacements: Vec<(String, String)>,
    rewrite_docs: bool,
    rewrite_attrs: bool,
    result: StructuredAttributeResult,
}

impl AttributeRewriteVisitor {
    fn new(mapping: &HashMap<String, String>, config: &StructuredEditOptions) -> Self {
        Self {
            replacements: build_replacements(mapping),
            rewrite_docs: config.doc_literals_enabled(),
            rewrite_attrs: config.attr_literals_enabled(),
            result: StructuredAttributeResult::new(),
        }
    }

    fn finish(self) -> StructuredAttributeResult {
        self.result
    }

    fn process_attribute(&mut self, attr: &mut syn::Attribute) {
        let is_doc = attr.path().is_ident("doc");
        if (is_doc && !self.rewrite_docs) || (!is_doc && !self.rewrite_attrs) {
            return;
        }

        let syn::Meta::NameValue(meta) = &mut attr.meta else {
            return;
        };
        let syn::Expr::Lit(expr_lit) = &mut meta.value else {
            return;
        };
        let syn::Lit::Str(lit) = &mut expr_lit.lit else {
            return;
        };

        let original = lit.value();
        if let Some(updated) = rewrite_literal(&original, &self.replacements) {
            if updated != original {
                let span = span_to_range(lit.span());
                *lit = syn::LitStr::new(&updated, lit.span());
                self.result.literal_spans.push(span);
                self.result.changed = true;
            }
        }
    }
}

impl VisitMut for AttributeRewriteVisitor {
    fn visit_attribute_mut(&mut self, attr: &mut syn::Attribute) {
        self.process_attribute(attr);
    }
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

fn rewrite_literal(value: &str, replacements: &[(String, String)]) -> Option<String> {
    let mut updated = value.to_string();
    let mut changed = false;

    for (old, new_name) in replacements {
        if let Some(next) = replace_identifier(&updated, old, new_name) {
            updated = next;
            changed = true;
        }
    }

    if changed {
        Some(updated)
    } else {
        None
    }
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
    if changed {
        Some(result)
    } else {
        None
    }
}

fn is_boundary(text: &str, start: usize, end: usize) -> bool {
    let prev = text[..start].chars().rev().next();
    let next = text[end..].chars().next();

    let prev_ok = !matches!(prev, Some(c) if c.is_ascii_alphanumeric() || c == '_');
    let next_ok = !matches!(next, Some(c) if c.is_ascii_alphanumeric() || c == '_');

    prev_ok && next_ok
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
