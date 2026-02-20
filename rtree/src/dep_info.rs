use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

mod parser;
use parser::extract_module_deps;

/// Builds a map from source file to its direct dependencies using syn parsing
/// This gives us semantic (not just syntactic) dependencies
pub fn build_file_dependency_map(src_root: &Path) -> Result<HashMap<PathBuf, HashSet<PathBuf>>> {
    let mut file_deps: HashMap<PathBuf, HashSet<PathBuf>> = HashMap::new();
    let mut module_to_file: HashMap<String, PathBuf> = HashMap::new();

    // First pass: build module name to file path mapping
    for entry in WalkDir::new(src_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let path = entry.path();
        let module_name = file_to_module_name(path, src_root)?;
        module_to_file.insert(module_name, path.to_path_buf());
    }

    // Second pass: extract dependencies for each file
    for entry in WalkDir::new(src_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let path = entry.path();
        let module_name = file_to_module_name(path, src_root)?;

        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Warning: Failed to read {}: {}", path.display(), e);
                continue;
            }
        };
        let dep_modules = extract_module_deps(&content, &module_name);

        let mut deps = HashSet::new();
        for dep_module in dep_modules {
            if let Some(dep_file) = resolve_module(&dep_module, &module_to_file) {
                if dep_file != path {
                    deps.insert(dep_file.clone());
                }
            }
        }

        file_deps.insert(path.to_path_buf(), deps);
    }

    Ok(file_deps)
}

fn file_to_module_name(path: &Path, src_root: &Path) -> Result<String> {
    let rel_path = path
        .strip_prefix(src_root)
        .map_err(|_| anyhow::anyhow!("Path not under src root"))?
        .with_extension("");

    let parts: Vec<&str> = rel_path
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let module_name = if parts.last() == Some(&"mod") {
        parts[..parts.len() - 1].join("::")
    } else if parts.last() == Some(&"lib") {
        "lib".to_string()
    } else if parts.last() == Some(&"main") {
        "main".to_string()
    } else {
        parts.join("::")
    };

    Ok(module_name)
}
/// Resolve a dep string to a file: try the full path first, then walk up
/// the module hierarchy until we find a known file or exhaust segments.
fn resolve_module<'a>(dep: &str, map: &'a HashMap<String, PathBuf>) -> Option<&'a PathBuf> {
    let mut candidate = dep.to_string();
    loop {
        if let Some(f) = map.get(candidate.as_str()) {
            return Some(f);
        }
        match candidate.rfind("::") {
            Some(pos) => candidate = candidate[..pos].to_string(),
            None => return map.get(candidate.as_str()),
        }
    }
}
