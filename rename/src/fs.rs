//! semantic: domain=tooling

use anyhow::Result;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct DiscoveredFiles {
    pub rust_files: Vec<PathBuf>,
    pub auxiliary_files: Vec<AuxiliaryFile>,
}

#[derive(Debug, Clone)]
pub struct AuxiliaryFile {
    pub path: PathBuf,
    pub kind: AuxiliaryKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxiliaryKind {
    CargoToml,
    BuildScript,
}

/// Legacy function for backward compatibility
pub fn collect_rs_files(root: &Path) -> Result<Vec<PathBuf>> {
    let discovered = discover_all_files(root)?;
    Ok(discovered.rust_files)
}

/// Comprehensive file discovery including benches, tests, examples, build scripts
pub fn discover_all_files(root: &Path) -> Result<DiscoveredFiles> {
    let mut files = Vec::new();
    let mut auxiliary_files = Vec::new();
    let root = root.to_path_buf();

    let walker = WalkDir::new(&root).into_iter();
    for entry_result in walker {
        let entry = match entry_result {
            Ok(e) => e,
            Err(_) => continue,
        };
        {
            let should_filter = entry.path() != root && is_ignored_dir(entry.path());
            if should_filter {
                continue;
            }
        }
        let path = entry.path();

        // Collect Rust source files
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext == "rs" {
                files.push(path.to_path_buf());
            }
        }

        // Collect auxiliary files
        if let Some(file_name) = path.file_name().and_then(|s| s.to_str()) {
            if file_name == "Cargo.toml" {
                auxiliary_files.push(AuxiliaryFile { path: path.to_path_buf(), kind: AuxiliaryKind::CargoToml });
            } else if file_name == "build.rs" {
                auxiliary_files.push(AuxiliaryFile { path: path.to_path_buf(), kind: AuxiliaryKind::BuildScript });
            }
        }
    }

    // Follow include! macros to discover additional files
    let mut include_targets = discover_include_targets(&files)?;
    files.append(&mut include_targets);

    // Deduplicate files
    files.sort();
    files.dedup();

    Ok(DiscoveredFiles { rust_files: files, auxiliary_files })
}

/// Parse Rust files to find include! macro invocations
fn discover_include_targets(files: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut targets = Vec::new();

    for file in files {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue, // Skip files we can't read
        };

        let ast = match syn::parse_file(&content) {
            Ok(a) => a,
            Err(_) => continue, // Skip files we can't parse
        };

        let mut visitor = IncludeVisitor { base_path: file.parent().unwrap_or_else(|| Path::new(".")), targets: &mut targets };
        visitor.visit_file(&ast);
    }

    Ok(targets)
}

struct IncludeVisitor<'a> {
    base_path: &'a Path,
    targets: &'a mut Vec<PathBuf>,
}

impl<'ast> Visit<'ast> for IncludeVisitor<'_> {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        if mac.path.is_ident("include") {
            if let Ok(lit) = mac.parse_body::<syn::LitStr>() {
                let target = self.base_path.join(lit.value());
                if target.exists() && target.extension().and_then(|s| s.to_str()) == Some("rs") {
                    self.targets.push(target);
                }
            }
        }
        visit::visit_macro(self, mac);
    }

    fn visit_expr_macro(&mut self, expr: &'ast syn::ExprMacro) {
        if expr.mac.path.is_ident("include") {
            if let Ok(lit) = expr.mac.parse_body::<syn::LitStr>() {
                let target = self.base_path.join(lit.value());
                if target.exists() && target.extension().and_then(|s| s.to_str()) == Some("rs") {
                    self.targets.push(target);
                }
            }
        }
        visit::visit_expr_macro(self, expr);
    }
}

fn is_ignored_dir(path: &Path) -> bool {
    if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
        matches!(name, "target" | ".git" | ".semantic-lint" | "dogfood-output")
    } else {
        false
    }
}
