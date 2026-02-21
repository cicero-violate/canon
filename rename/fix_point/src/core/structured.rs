use crate::structured::StructuredEditOptions;


use std::collections::HashSet;


#[derive(Default)]
pub struct EditSessionTracker {
    files: HashSet<String>,
    pub(crate) doc_files: HashSet<String>,
    pub(crate) attr_files: HashSet<String>,
    pub(crate) use_files: HashSet<String>,
}
