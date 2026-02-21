use std::path::{Path, PathBuf};


use syn::visit::{self, Visit};


pub struct IncludeVisitor<'a> {
    pub(crate) base_path: &'a Path,
    pub(crate) targets: &'a mut Vec<PathBuf>,
}
