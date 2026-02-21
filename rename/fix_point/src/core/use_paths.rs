use super::structured::EditSessionTracker;


use crate::alias::AliasGraph;


use crate::fs;


use crate::structured::StructuredEditOptions;


use crate::resolve::ResolverContext;


use crate::model::types::SymbolIndex;


use std::sync::Arc;


use anyhow::{Context, Result};


use std::collections::{HashMap, HashSet};


use std::path::PathBuf;


pub fn update_use_paths(
    project: &std::path::Path,
    file_renames: &[crate::model::types::FileRename],
    symbol_mapping: &HashMap<String, String>,
    structured_config: &StructuredEditOptions,
    alias_graph: &AliasGraph,
    symbol_table: &SymbolIndex,
    structured_tracker: &mut EditSessionTracker,
    touched_files: &mut HashSet<PathBuf>,
) -> Result<()> {
    let mut path_updates: HashMap<String, String> = HashMap::new();
    for rename in file_renames {
        path_updates.insert(rename.old_module_id.clone(), rename.new_module_id.clone());
    }
    for (old_id, new_name) in symbol_mapping {
        path_updates
            .insert(
                old_id.clone(),
                {
                    if let Some(pos) = old_id.rfind("::") {
                        format!("{}{}{}", & old_id[..pos + 2], new_name, "")
                    } else {
                        new_name.clone()
                    }
                },
            );
    }
    if path_updates.is_empty() {
        return Ok(());
    }
    let files = fs::collect_rs_files(project)?;
    let alias_graph = Arc::new(alias_graph.clone());
    let symbol_table = Arc::new(symbol_table.clone());
    for file in &files {
        let content = std::fs::read_to_string(file)?;
        let mut ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        use crate::structured::orchestrator::StructuredPass;
        use crate::structured::use_tree::UsePathRewritePass;
        let file_key = file.to_string_lossy().to_string();
        let alias_nodes = alias_graph
            .nodes_in_file(&file_key)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let use_config = if structured_config.use_statements_enabled() {
            structured_config.clone()
        } else {
            StructuredEditOptions::new(false, false, true)
        };
        let resolver = ResolverContext {
            module_path: super::paths::module_path_for_file(project, file),
            alias_graph: alias_graph.clone(),
            symbol_table: symbol_table.clone(),
        };
        let mut pass = UsePathRewritePass::new(
            path_updates.clone(),
            alias_nodes,
            use_config,
            resolver,
        );
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
