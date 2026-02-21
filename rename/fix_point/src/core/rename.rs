use super::alias::collect_and_rename_aliases;


use super::collect::{add_file_module_symbol, collect_symbols};


use super::format::format_files;


use super::mod_decls::update_mod_declarations;


use super::paths::{module_path_for_file, plan_file_renames};


use super::preview::write_preview;


use super::structured::EditSessionTracker;


use crate::model::types::{SpanRange, SymbolEdit, SymbolIndex, SymbolOccurrence};


use super::use_map::build_use_map;


use super::use_paths::update_use_paths;


use crate::alias::AliasGraph;


use crate::fs;


use crate::structured::{
    rewrite_doc_and_attr_literals, structured_edit_config, StructuredAttributeResult,
};


use anyhow::{bail, Context, Result};


use proc_macro2::Span;


use std::collections::{HashMap, HashSet};


use std::path::{Path, PathBuf};


use syn::visit::Visit;


use syn::visit_mut::VisitMut;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SpanRangeKey {
    start_line: i64,
    start_col: i64,
    end_line: i64,
    end_col: i64,
}


struct SpanRangeRenamer {
    map: HashMap<SpanRangeKey, String>,
    changed: bool,
}


pub fn apply_rename(
    project: &Path,
    map_path: &Path,
    dry_run: bool,
    out_path: Option<&Path>,
) -> Result<()> {
    let mapping: HashMap<String, String> = serde_json::from_str(
        &std::fs::read_to_string(map_path)?,
    )?;
    apply_rename_with_map(project, &mapping, dry_run, out_path)
}


pub fn apply_rename_with_map(
    project: &Path,
    mapping: &HashMap<String, String>,
    dry_run: bool,
    out_path: Option<&Path>,
) -> Result<()> {
    if mapping.is_empty() {
        println!("No renames applied (empty map).");
        return Ok(());
    }
    for (id, name) in mapping {
        if !is_valid_ident(name) {
            bail!("Invalid identifier for {}: {}", id, name);
        }
    }
    let files = fs::collect_rs_files(project)?;
    let mut symbol_table = SymbolIndex::default();
    let mut symbol_set: HashSet<String> = HashSet::new();
    let mut symbols = Vec::new();
    let mut global_alias_graph = AliasGraph::new();
    let structured_config = structured_edit_config();
    let structured_mode = structured_config.is_enabled();
    let mut structured_tracker = EditSessionTracker::new();
    let mut touched_files: HashSet<PathBuf> = HashSet::new();
    if structured_mode {
        eprintln!("Structured editing enabled: {}", structured_config.summary());
    }
    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        add_file_module_symbol(
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        let file_alias_graph = collect_symbols(
            &ast,
            &module_path,
            file,
            &mut symbol_table,
            &mut symbols,
            &mut symbol_set,
        );
        for node in file_alias_graph.all_nodes() {
            global_alias_graph.add_use_node(node.clone());
        }
    }
    global_alias_graph.build_edges();
    let file_renames = plan_file_renames(&symbol_table, &mapping)?;
    let alias_edits = collect_and_rename_aliases(project, &symbol_table, &mapping)?;
    let mut alias_by_file: HashMap<String, Vec<SymbolEdit>> = HashMap::new();
    for edit in &alias_edits {
        alias_by_file.entry(edit.file.clone()).or_default().push(edit.clone());
    }
    let mut all_edits: Vec<SymbolEdit> = Vec::new();
    for file in &files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let mut ast = syn::parse_file(&content)
            .with_context(|| format!("Failed to parse {}", file.display()))?;
        let use_map = build_use_map(&ast, &module_path);
        let mut structured_attr = StructuredAttributeResult::new();
        if structured_config.doc_or_attr_enabled() {
            structured_attr = rewrite_doc_and_attr_literals(
                file,
                &content,
                &mut ast,
                &mapping,
                &structured_config,
            )?;
            if structured_attr.changed {
                let file_str = file.to_string_lossy().to_string();
                if structured_config.doc_literals_enabled() {
                    structured_tracker.mark_doc_edit(file_str.clone());
                }
                if structured_config.attr_literals_enabled() {
                    structured_tracker.mark_attr_edit(file_str);
                }
            }
        }
        let mut occurrences = Vec::new();
        let mut visitor = crate::occurrence::OccurrenceVisitor::new(
            &module_path,
            file,
            &symbol_table,
            &use_map,
            &global_alias_graph,
            &mut occurrences,
        );
        visitor.visit_file(&ast);
        for (symbol_id, symbol_entry) in &symbol_table.symbols {
            if symbol_entry.file == file.to_string_lossy().to_string() {
                if mapping.contains_key(symbol_id.as_str()) {
                    occurrences
                        .push(SymbolOccurrence {
                            id: symbol_id.clone(),
                            file: symbol_entry.file.clone(),
                            kind: format!("{}_definition", symbol_entry.kind),
                            span: symbol_entry.span.clone(),
                        });
                }
            }
        }
        let mut edits = Vec::new();
        for occ in occurrences {
            if structured_config.doc_or_attr_enabled() && occ.kind == "attribute"
                && structured_attr.should_skip(&occ.span)
            {
                continue;
            }
            if let Some(new_name) = mapping.get(&occ.id) {
                edits
                    .push(SymbolEdit {
                        id: occ.id.clone(),
                        file: occ.file.clone(),
                        kind: occ.kind.clone(),
                        start: occ.span.start.clone(),
                        end: occ.span.end.clone(),
                        new_name: new_name.clone(),
                    });
            }
        }
        let file_key = file.to_string_lossy().to_string();
        if let Some(alias_edits) = alias_by_file.remove(&file_key) {
            edits.extend(alias_edits);
        }
        if !dry_run {
            let mut changed = false;
            if structured_attr.changed {
                changed = true;
            }
            if apply_symbol_edits_to_ast(&mut ast, &edits)? {
                changed = true;
            }
            if changed {
                let rendered = prettyplease::unparse(&ast);
                if rendered != content {
                    std::fs::write(file, rendered)?;
                    touched_files.insert(file.to_path_buf());
                }
            }
        }
        all_edits.extend(edits);
    }
    if !dry_run {
        for (path_str, edits) in alias_by_file {
            let path = Path::new(&path_str);
            let content = std::fs::read_to_string(path)?;
            let mut ast = syn::parse_file(&content)
                .with_context(|| format!("Failed to parse {}", path.display()))?;
            if apply_symbol_edits_to_ast(&mut ast, &edits)? {
                let rendered = prettyplease::unparse(&ast);
                if rendered != content {
                    std::fs::write(path, rendered)?;
                    touched_files.insert(path.to_path_buf());
                }
            }
            all_edits.extend(edits);
        }
    } else {
        for edits in alias_by_file.values() {
            all_edits.extend(edits.iter().cloned());
        }
    }
    if dry_run {
        let out = out_path
            .unwrap_or_else(|| Path::new(".semantic-lint/rename_preview.json"));
        write_preview(
            out,
            &all_edits,
            &file_renames,
            &structured_tracker,
            &structured_config,
        )?;
        return Ok(());
    }
    for rename in &file_renames {
        if rename.from == rename.to {
            continue;
        }
        if Path::new(&rename.to).exists() {
            bail!("File already exists: {}", rename.to);
        }
        if rename.is_directory_move {
            if let Some(parent) = Path::new(&rename.to).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::rename(&rename.from, &rename.to)?;
    }
    update_mod_declarations(project, &symbol_table, &file_renames, &mut touched_files)?;
    update_use_paths(
        project,
        &file_renames,
        &mapping,
        &structured_config,
        &global_alias_graph,
        &symbol_table,
        &mut structured_tracker,
        &mut touched_files,
    )?;
    if structured_mode && !structured_tracker.all_files().is_empty() {
        eprintln!("{}", structured_tracker.summary(& structured_config));
    }
    if !touched_files.is_empty() {
        let touched: Vec<PathBuf> = touched_files.iter().cloned().collect();
        let _ = format_files(&touched)?;
    }
    let mut edited_files = HashSet::new();
    for edit in &all_edits {
        edited_files.insert(edit.file.clone());
    }
    if file_renames.is_empty() {
        println!(
            "Renamed {} occurrences across {} files.", all_edits.len(), edited_files
            .len()
        );
    } else {
        println!(
            "Renamed {} occurrences across {} files ({} file renames).", all_edits.len(),
            edited_files.len(), file_renames.len()
        );
    }
    Ok(())
}


pub fn apply_symbol_edits_to_ast(
    ast: &mut syn::File,
    edits: &[SymbolEdit],
) -> Result<bool> {
    if edits.is_empty() {
        return Ok(false);
    }
    let mut map: HashMap<SpanRangeKey, String> = HashMap::new();
    for edit in edits {
        let key = SpanRangeKey::from_range(
            &SpanRange {
                start: edit.start.clone(),
                end: edit.end.clone(),
            },
        );
        if let Some(existing) = map.get(&key) {
            if existing != &edit.new_name {
                bail!(
                    "Conflicting rename edits for span {:?}: {} vs {}", key, existing,
                    edit.new_name
                );
            }
        } else {
            map.insert(key, edit.new_name.clone());
        }
    }
    let mut renamer = SpanRangeRenamer {
        map,
        changed: false,
    };
    renamer.visit_file_mut(ast);
    Ok(renamer.changed)
}


fn is_valid_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}


fn rewrite_token_stream(
    tokens: proc_macro2::TokenStream,
    map: &HashMap<SpanRangeKey, String>,
    changed: &mut bool,
) -> proc_macro2::TokenStream {
    use proc_macro2::{Ident, TokenTree};
    let mut out = proc_macro2::TokenStream::new();
    for tt in tokens.into_iter() {
        match tt {
            TokenTree::Ident(ident) => {
                let key = SpanRangeKey::from_span(ident.span());
                if let Some(new_name) = map.get(&key) {
                    if ident.to_string() != *new_name {
                        let new_ident = Ident::new(new_name, ident.span());
                        out.extend([TokenTree::Ident(new_ident)]);
                        *changed = true;
                        continue;
                    }
                }
                out.extend([TokenTree::Ident(ident)]);
            }
            TokenTree::Group(group) => {
                let stream = rewrite_token_stream(group.stream(), map, changed);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), stream);
                new_group.set_span(group.span());
                out.extend([TokenTree::Group(new_group)]);
            }
            other => {
                out.extend([other]);
            }
        }
    }
    out
}
