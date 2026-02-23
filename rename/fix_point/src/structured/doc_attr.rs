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
