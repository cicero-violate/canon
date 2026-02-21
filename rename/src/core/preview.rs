use super::structured::StructuredEditTracker;
use crate::model::types::{FileRename, SymbolEdit};
use crate::structured::StructuredEditOptions;
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;
/// B2: Write preview with structured edit tracking
pub(crate) fn write_preview(out: &Path, edits: &[SymbolEdit], renames: &[FileRename], structured_tracker: &StructuredEditTracker, config: &StructuredEditOptions) -> Result<()> {
    let mut by_file: BTreeMap<String, Vec<&SymbolEdit>> = BTreeMap::new();
    for edit in edits {
        by_file.entry(edit.file.clone()).or_default().push(edit);
    }
    let mut files = BTreeMap::new();
    for (file, edits) in by_file {
        let list: Vec<_> = edits
            .into_iter()
            .map(|e| {
                serde_json::json!(
                    { "id" : e.id, "kind" : e.kind, "start" : e.start, "end" : e.end,
                    "new_name" : e.new_name, }
                )
            })
            .collect();
        files.insert(file, list);
    }
    let rename_list: Vec<_> = renames
        .iter()
        .map(|r| {
            serde_json::json!(
                { "from" : r.from, "to" : r.to, "is_directory_move" : r
                .is_directory_move, "old_module_id" : r.old_module_id, "new_module_id" :
                r.new_module_id }
            )
        })
        .collect();
    let mut structured: Vec<_> = structured_tracker.all_files().iter().cloned().collect();
    structured.sort();
    let mut doc_files: Vec<_> = structured_tracker.doc_files().iter().cloned().collect();
    doc_files.sort();
    let mut attr_files: Vec<_> = structured_tracker.attr_files().iter().cloned().collect();
    attr_files.sort();
    let mut use_files: Vec<_> = structured_tracker.use_files().iter().cloned().collect();
    use_files.sort();
    let preview = serde_json::json!(
        { "files" : files, "file_renames" : rename_list, "structured_files" : structured,
        "structured_edits" : { "enabled" : config.is_enabled(), "config" : config
        .summary(), "summary" : structured_tracker.summary(config), "total_files" :
        structured_tracker.all_files().len(), "by_pass" : { "doc_literals" : { "enabled"
        : config.doc_literals_enabled(), "files" : doc_files, "count" :
        structured_tracker.doc_files().len(), }, "attr_literals" : { "enabled" : config
        .attr_literals_enabled(), "files" : attr_files, "count" : structured_tracker
        .attr_files().len(), }, "use_statements" : { "enabled" : config
        .use_statements_enabled(), "files" : use_files, "count" : structured_tracker
        .use_files().len(), }, }, }, }
    );
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out, serde_json::to_vec_pretty(&preview)?)?;
    Ok(())
}
