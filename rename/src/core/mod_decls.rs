use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::fs;

use super::paths::{module_child_path, module_path_for_file};
use super::types::{FileRename, SymbolIndex};

/// Update mod declarations after file renames/moves
pub(crate) fn update_mod_declarations(project: &Path, table: &SymbolIndex, file_renames: &[FileRename], touched_files: &mut HashSet<PathBuf>) -> Result<()> {
    if file_renames.is_empty() {
        return Ok(());
    }

    let mut rename_lookup: HashMap<String, PathBuf> = HashMap::new();
    for rename in file_renames {
        rename_lookup.insert(rename.from.clone(), PathBuf::from(&rename.to));
    }

    // Build mapping of affected modules
    let mut module_changes: HashMap<String, ModuleRenamePlan> = HashMap::new();
    let mut files_to_process: HashSet<PathBuf> = HashSet::new();
    let mut fallback_required = false;

    for rename in file_renames {
        let old_parts: Vec<&str> = rename.old_module_id.split("::").collect();
        let new_parts: Vec<&str> = rename.new_module_id.split("::").collect();

        if old_parts.len() < 2 || new_parts.len() < 2 {
            continue; // crate-level modules don't need mod declarations updated
        }

        let old_module_name = old_parts.last().unwrap();
        let new_module_name = new_parts.last().unwrap();
        let old_parent = old_parts[..old_parts.len() - 1].join("::");
        let new_parent = new_parts[..new_parts.len() - 1].join("::");

        let change = ModuleRenamePlan { old_name: old_module_name.to_string(), new_name: new_module_name.to_string(), old_parent: old_parent.clone(), new_parent: new_parent.clone() };
        module_changes.insert(rename.old_module_id.clone(), change);

        if let Some(entry) = table.symbols.get(&rename.old_module_id) {
            if let Some(decl_file) = entry.declaration_file.as_ref() {
                let candidate = resolve_renamed_path(PathBuf::from(decl_file), &rename_lookup);
                if candidate.exists() {
                    files_to_process.insert(candidate);
                } else {
                    fallback_required = true;
                }
            } else {
                fallback_required = true;
            }
        } else {
            fallback_required = true;
        }

        if old_parent != new_parent {
            match resolve_parent_definition_file(project, table, &new_parent, &rename_lookup) {
                Some(path) if path.exists() => {
                    files_to_process.insert(path);
                }
                Some(_) | None => fallback_required = true,
            }
        }
    }

    let target_files: Vec<PathBuf> = if fallback_required || files_to_process.is_empty() { fs::collect_rs_files(project)? } else { files_to_process.into_iter().collect() };

    for file in &target_files {
        let module_path = module_path_for_file(project, file);
        let content = std::fs::read_to_string(file)?;
        let mut ast = syn::parse_file(&content).with_context(|| format!("Failed to parse {}", file.display()))?;

        let mut changed = false;
        let mut remove_indices = Vec::new();

        // Find mod declarations that need updating
        for (index, item) in ast.items.iter_mut().enumerate() {
            if let syn::Item::Mod(item_mod) = item {
                let mod_name = item_mod.ident.to_string();
                let child_module_id = module_child_path(&module_path, mod_name.clone());

                if let Some(change) = module_changes.get(&child_module_id) {
                    if change.old_parent == change.new_parent && change.old_parent == module_path {
                        if change.old_name != change.new_name {
                            item_mod.ident = syn::Ident::new(&change.new_name, item_mod.ident.span());
                            changed = true;
                        }
                    } else if change.old_parent == module_path {
                        remove_indices.push(index);
                    } else if change.new_parent == module_path {
                        if change.old_name != change.new_name {
                            item_mod.ident = syn::Ident::new(&change.new_name, item_mod.ident.span());
                            changed = true;
                        }
                    }
                }
            }
        }

        if !remove_indices.is_empty() {
            remove_indices.sort();
            for index in remove_indices.into_iter().rev() {
                ast.items.remove(index);
            }
            changed = true;
        }

        // Check if we need to add new mod declarations
        let mut insert_index = ast.items.iter().rposition(|item| matches!(item, syn::Item::Mod(_))).map(|idx| idx + 1).unwrap_or(0);
        for change in module_changes.values() {
            if change.new_parent == module_path {
                let has_declaration = ast.items.iter().any(|item| if let syn::Item::Mod(item_mod) = item { item_mod.ident.to_string() == change.new_name } else { false });

                if !has_declaration && change.old_parent != module_path {
                    let ident = syn::Ident::new(&change.new_name, proc_macro2::Span::call_site());
                    let new_mod: syn::ItemMod = syn::parse_quote! { mod #ident; };
                    ast.items.insert(insert_index, syn::Item::Mod(new_mod));
                    insert_index += 1;
                    changed = true;
                }
            }
        }

        if changed {
            let rendered = prettyplease::unparse(&ast);
            if rendered != content {
                std::fs::write(file, rendered)?;
                touched_files.insert(file.to_path_buf());
            }
        }
    }

    Ok(())
}

fn resolve_renamed_path(path: PathBuf, lookup: &HashMap<String, PathBuf>) -> PathBuf {
    let key = path.to_string_lossy().to_string();
    lookup.get(&key).cloned().unwrap_or(path)
}

fn resolve_parent_definition_file(project: &Path, table: &SymbolIndex, parent_id: &str, rename_lookup: &HashMap<String, PathBuf>) -> Option<PathBuf> {
    let raw_path = if parent_id == "crate" {
        find_crate_root_file(project)?
    } else {
        let entry = table.symbols.get(parent_id)?;
        let file = entry.definition_file.as_ref().or(entry.declaration_file.as_ref())?.to_string();
        PathBuf::from(file)
    };
    Some(resolve_renamed_path(raw_path, rename_lookup))
}

fn find_crate_root_file(project: &Path) -> Option<PathBuf> {
    let lib = project.join("src/lib.rs");
    if lib.exists() {
        return Some(lib);
    }
    let main = project.join("src/main.rs");
    if main.exists() {
        return Some(main);
    }
    None
}

#[derive(Debug)]
struct ModuleRenamePlan {
    old_name: String,
    new_name: String,
    old_parent: String,
    new_parent: String,
}
