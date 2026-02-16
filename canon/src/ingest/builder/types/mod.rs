pub(crate) mod ports;
pub(crate) mod strings;
pub(crate) mod types;
pub(crate) mod uses;
pub(crate) mod visibility;

pub(crate) use ports::{
    convert_fields, convert_generics, convert_inputs, convert_receiver, convert_return_type,
};
pub(crate) use strings::{
    attribute_to_string, collect_derives, collect_doc_string, expr_to_string, path_to_string,
    slugify, to_pascal_case, word_from_ident, word_from_string,
};
pub(crate) use types::{convert_type, path_type, type_from_bound};
pub(crate) use uses::{
    flatten_use_tree, module_segments_from_key, render_use_item, resolve_use_entry,
};
pub(crate) use visibility::map_visibility;

/// Shared binding type used by both `edges` and `modules`.
/// Lives here to break the mutual import between those two sibling modules.
#[derive(Clone)]
pub(crate) struct AliasBinding {
    pub module_key: String,
    pub function_slug: String,
}

/// Flattened representation of a single `use` path entry.
/// Defined here (not in `edges`) so that `types::uses` can use it
/// without importing from `edges`, breaking the types→edges→modules cycle.
#[derive(Clone, Debug)]
pub(crate) struct UseEntry {
    pub segments: Vec<String>,
    pub alias: Option<String>,
    pub is_glob: bool,
    pub leading_colon: bool,
}
