use crate::core::paths::module_path_for_file;


use crate::core::project_editor::propagate::build_symbol_index_and_occurrences;


use crate::core::symbol_id::normalize_symbol_id;


use crate::core::use_map::{normalize_use_prefix, path_to_string};


use crate::resolve::Resolver;


use crate::state::NodeRegistry;


use anyhow::Result;


use quote::ToTokens;


use std::collections::{HashMap, HashSet};


use std::path::PathBuf;


use syn::visit::{self, Visit};


use syn::visit_mut::VisitMut;


struct CanonicalRewriteVisitor {
    module_path: String,
    rewrite_map: HashMap<String, (String, String)>,
    changed: bool,
}


#[derive(Debug, Default, Clone)]
pub(crate) struct MoveSet {
    pub entries: HashMap<String, (String, String)>,
}


struct ReferenceCollector<'a> {
    resolver: Resolver<'a>,
    refs: HashSet<String>,
}


struct UseImport {
    source_path: String,
    is_glob: bool,
}
