use anyhow::{bail, Result};


use serde::{Deserialize, Serialize};


use std::collections::HashMap;


use std::path::PathBuf;


use syn::spanned::Spanned;


use super::ast_render;


use crate::model::core_span::span_to_offsets;


use crate::model::span::LineColumn;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct AstEdit {
    pub file: PathBuf,
    pub start: usize,
    pub end: usize,
    pub replacement: String,
}
