use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::types::{FileRename, SymbolIndex};

pub(crate) fn plan_file_renames(
    table: &SymbolIndex,
    mapping: &HashMap<String, String>,
) -> Result<Vec<FileRename>> {
    let mut renames = Vec::new();
    for (id, new_name) in mapping {
        let Some(sym) = table.symbols.get(id) else {
            continue;
        };
        if sym.kind != "module" {
            continue;
        }

        let Some(def_path_string) = sym.definition_file.clone() else {
            continue;
        };
        let def_path = PathBuf::from(&def_path_string);

        // Check if new_name contains path separators (directory move)
        let is_directory_move = new_name.contains('/') || new_name.contains("::");

        if is_directory_move {
            // Handle directory moves: new_name is a new module path
            if let Some(new_path) = compute_new_file_path(&def_path_string, &sym.id, new_name)? {
                let new_module_id = if new_name.starts_with("crate::") {
                    new_name.to_string()
                } else if new_name.contains("::") {
                    new_name.to_string()
                } else {
                    // Simple name - replace last component
                    let parts: Vec<&str> = sym.id.split("::").collect();
                    if parts.len() > 1 {
                        format!("{}::{}", parts[..parts.len() - 1].join("::"), new_name)
                    } else {
                        format!("crate::{}", new_name)
                    }
                };
                renames.push(FileRename {
                    from: def_path_string.clone(),
                    to: new_path,
                    is_directory_move: true,
                    old_module_id: sym.id.clone(),
                    new_module_id,
                });
            }
        } else {
            // Handle simple renames within same directory
            let path = def_path.as_path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };

            // For mod.rs files, the module name is the parent directory name
            let matches_module = if stem == "mod" {
                // For src/refactor/mod.rs, check if parent dir name matches
                path.parent()
                    .and_then(|p| p.file_name())
                    .and_then(|n| n.to_str())
                    .map(|dir_name| dir_name == sym.name)
                    .unwrap_or(false)
            } else {
                stem == sym.name
            };

            if !matches_module {
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }

            // Determine the new path
            let new_path = if stem == "mod" {
                // For mod.rs files, we rename the parent directory
                // src/refactor/mod.rs -> the directory src/refactor/ should become src/refactoring/
                // Store the directory path, not the mod.rs file path
                path.parent().unwrap().to_path_buf()
            } else {
                // Rename file: src/refactor.rs -> src/refactoring.rs
                let mut new_path = path.to_path_buf();
                new_path.set_file_name(format!("{}.rs", new_name));
                new_path
            };

            let new_module_id = {
                let parts: Vec<&str> = sym.id.split("::").collect();
                if parts.len() > 1 {
                    format!("{}::{}", parts[..parts.len() - 1].join("::"), new_name)
                } else {
                    format!("crate::{}", new_name)
                }
            };

            let (from, to) = if stem == "mod" {
                // Rename the directory
                let old_dir = path.parent().unwrap();
                let new_dir = old_dir.parent().unwrap().join(new_name);
                (
                    old_dir.to_string_lossy().to_string(),
                    new_dir.to_string_lossy().to_string(),
                )
            } else {
                // Rename the file
                (
                    def_path_string.clone(),
                    new_path.to_string_lossy().to_string(),
                )
            };

            renames.push(FileRename {
                from,
                to,
                is_directory_move: false,
                old_module_id: sym.id.clone(),
                new_module_id,
            });
        }
    }
    Ok(renames)
}

/// Compute new file path when moving a module to a new location
/// new_module_path can be either:
///   - A module path like "crate::new::location::module"
///   - A file path like "src/new/location/module.rs"
fn compute_new_file_path(
    old_file: &str,
    _old_module_id: &str,
    new_module_path: &str,
) -> Result<Option<String>> {
    let old_path = Path::new(old_file);

    // Determine project root by finding 'src' directory
    let mut project_root = old_path.to_path_buf();
    let mut found_src = false;
    while let Some(parent) = project_root.parent() {
        if project_root.file_name().and_then(|s| s.to_str()) == Some("src") {
            found_src = true;
            project_root = parent.to_path_buf();
            break;
        }
        project_root = parent.to_path_buf();
    }

    if !found_src {
        // Can't determine project structure
        return Ok(None);
    }

    // Parse new module path
    let new_path_str = if new_module_path.starts_with("crate::") {
        // Module path format: "crate::foo::bar"
        new_module_path.trim_start_matches("crate::")
    } else if new_module_path.contains("::") {
        // Relative module path
        new_module_path
    } else if new_module_path.contains('/') {
        // File path format: "src/foo/bar.rs" or "foo/bar"
        new_module_path
            .trim_start_matches("src/")
            .trim_end_matches(".rs")
    } else {
        // Simple name
        new_module_path
    };

    // Convert module path to file path
    let parts: Vec<&str> = new_path_str.split("::").collect();
    let mut new_file_path = project_root.join("src");

    // Check if old file was mod.rs
    let is_mod_rs = old_path.file_name().and_then(|s| s.to_str()) == Some("mod.rs");

    if parts.is_empty() {
        return Ok(None);
    }

    // Build directory structure
    for part in &parts[..parts.len() - 1] {
        new_file_path.push(part);
    }

    // Add final component
    let last_part = parts[parts.len() - 1];
    if is_mod_rs {
        // mod.rs stays as mod.rs in new location
        new_file_path.push(last_part);
        new_file_path.push("mod.rs");
    } else {
        // Regular file: foo.rs
        new_file_path.push(format!("{}.rs", last_part));
    }

    Ok(Some(new_file_path.to_string_lossy().to_string()))
}

pub(crate) fn module_path_for_file(project: &Path, file: &Path) -> String {
    let mut rel = file.strip_prefix(project).unwrap_or(file).to_path_buf();
    if rel.components().next().map(|c| c.as_os_str()) == Some("src".as_ref()) {
        rel = rel.strip_prefix("src").unwrap_or(&rel).to_path_buf();
    }
    let mut parts: Vec<String> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_string()))
        .collect();
    if parts.is_empty() {
        return "crate".to_string();
    }
    if let Some(last) = parts.last_mut() {
        if last == "lib.rs" || last == "main.rs" {
            parts.pop();
        } else if last == "mod.rs" {
            parts.pop();
        } else if last.ends_with(".rs") {
            *last = last.trim_end_matches(".rs").to_string();
        }
    }
    if parts.is_empty() {
        "crate".to_string()
    } else {
        format!("crate::{}", parts.join("::"))
    }
}

pub(crate) fn module_child_path(module_path: &str, child: String) -> String {
    if module_path == "crate" {
        format!("crate::{}", child)
    } else {
        format!("{}::{}", module_path, child)
    }
}
