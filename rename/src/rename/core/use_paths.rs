use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::fs;
use crate::rename::alias::AliasGraph;
use crate::rename::structured::StructuredEditConfig;

use super::structured::StructuredEditTracker;

/// B2: Update use statement paths after module moves with structured tracking
pub(crate) fn update_use_paths(
    project: &std::path::Path,
    file_renames: &[super::types::FileRename],
    structured_config: &StructuredEditConfig,
    alias_graph: &AliasGraph,
    structured_tracker: &mut StructuredEditTracker,
    touched_files: &mut HashSet<PathBuf>,
) -> Result<()> {
    if file_renames.is_empty() {
        return Ok(());
    }

    // Build mapping of old module paths to new module paths
    let mut path_updates: HashMap<String, String> = HashMap::new();
    for rename in file_renames {
        path_updates.insert(rename.old_module_id.clone(), rename.new_module_id.clone());
    }

    let files = fs::collect_rs_files(project)?;
    for file in &files {
        let content = std::fs::read_to_string(file)?;
        let mut ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;

        use crate::rename::structured::orchestrator::StructuredPass;
        use crate::rename::structured::use_tree::UseTreePass;

        let file_key = file.to_string_lossy().to_string();
        let alias_nodes = alias_graph
            .nodes_in_file(&file_key)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();

        let use_config = if structured_config.use_statements_enabled() {
            structured_config.clone()
        } else {
            StructuredEditConfig::new(false, false, true)
        };

        let mut pass = UseTreePass::new(path_updates.clone(), alias_nodes, use_config);

        if pass.execute(file, &content, &mut ast)? {
            let rendered = prettyplease::unparse(&ast);
            if rendered != content {
                std::fs::write(file, rendered)?;
                touched_files.insert(file.to_path_buf());
            }
            if structured_config.use_statements_enabled() {
                structured_tracker.mark_use_edit(file_key);
            }
        }
    }

    Ok(())
}
