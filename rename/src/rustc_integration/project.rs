//! Cargo project helper utilities (extern discovery, build metadata).

use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::rename::core::StructuralEditOracle;

/// Handle discovery of Cargo metadata and extern artifacts for a project.
#[derive(Debug, Clone)]
pub struct CargoProject {
    root: PathBuf,
    target_dir: PathBuf,
}

impl CargoProject {
    /// Attempts to load a Cargo project by walking up from the entry file.
    pub fn from_entry(entry: &Path) -> io::Result<Self> {
        let mut current = entry.canonicalize()?;
        if current.is_file() {
            current = current
                .parent()
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "entry has no parent"))?
                .to_path_buf();
        }

        let root = find_cargo_root(&current)?;
        let target_dir = fetch_target_dir(&root).unwrap_or_else(|_| root.join("target"));

        Ok(Self { root, target_dir })
    }

    /// Ensures dependencies are built (debug profile).
    pub fn build_dependencies(&self) -> io::Result<()> {
        let status = Command::new("cargo")
            .arg("build")
            .current_dir(&self.root)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::Other,
                "cargo build failed inside project",
            ))
        }
    }

    /// Collects `--extern` and `-L dependency=` flags for the requested profile.
    pub fn extern_args(&self, profile: &str) -> io::Result<Vec<String>> {
        let deps_dir = self.target_dir.join(profile).join("deps");
        if !deps_dir.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("deps dir missing: {}", deps_dir.display()),
            ));
        }

        let mut by_crate: HashMap<String, Vec<Artifact>> = HashMap::new();
        for entry in fs::read_dir(&deps_dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some((crate_name, kind)) = classify_artifact(name) {
                    by_crate
                        .entry(crate_name.to_string())
                        .or_default()
                        .push(Artifact { path, kind });
                }
            }
        }

        let mut args = Vec::new();
        for (crate_name, artifacts) in by_crate {
            if let Some(best) = select_best_artifact(&artifacts) {
                args.push("--extern".to_string());
                args.push(format!("{}={}", crate_name, best.display()));
            }
        }

        args.push("-L".to_string());
        args.push(format!("dependency={}", deps_dir.display()));

        Ok(args)
    }

    /// Returns parsed `cargo metadata` information for the project.
    pub fn metadata(&self) -> io::Result<ProjectMetadata> {
        let meta = load_cargo_metadata(&self.root)?;
        let CargoMetadata {
            target_directory,
            workspace_root,
            packages,
        } = meta;
        let (edition, rust_version) = packages
            .get(0)
            .map(|pkg| (Some(pkg.edition.clone()), pkg.rust_version.clone()))
            .unwrap_or((None, None));
        Ok(ProjectMetadata {
            workspace_root,
            target_directory,
            packages,
            edition,
            rust_version,
        })
    }

    /// Returns all cargo targets for the primary package.
    pub fn targets(&self) -> io::Result<Vec<CargoTarget>> {
        let meta = self.metadata()?;
        if let Some(pkg) = meta.packages.first() {
            // Filter out example targets that might come from dependencies
            let targets: Vec<CargoTarget> = pkg
                .targets
                .iter()
                .filter(|t| {
                    // Only include lib and bin targets, skip examples
                    t.kind.iter().any(|k| k == "lib" || k == "bin")
                })
                .cloned()
                .collect();
            Ok(targets)
        } else {
            Ok(Vec::new())
        }
    }

    /// Returns the workspace root directory.
    pub fn workspace_root(&self) -> &Path {
        &self.root
    }
}

fn find_cargo_root(start: &Path) -> io::Result<PathBuf> {
    let mut current = Some(start.to_path_buf());
    while let Some(dir) = current {
        if dir.join("Cargo.toml").exists() {
            return Ok(dir);
        }
        current = dir.parent().map(|p| p.to_path_buf());
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "Cargo.toml not found in parent chain",
    ))
}

fn fetch_target_dir(root: &Path) -> io::Result<PathBuf> {
    let meta = load_cargo_metadata(root)?;
    Ok(meta.target_directory.clone())
}

fn load_cargo_metadata(root: &Path) -> io::Result<CargoMetadata> {
    let output = Command::new("cargo")
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--no-deps")
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "cargo metadata failed",
        ));
    }
    let meta: CargoMetadata = serde_json::from_slice(&output.stdout)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(meta)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArtifactKind {
    ProcMacro,
    Rlib,
    Rmeta,
}

#[derive(Debug, Clone)]
struct Artifact {
    path: PathBuf,
    kind: ArtifactKind,
}

fn classify_artifact(file_name: &str) -> Option<(&str, ArtifactKind)> {
    if !file_name.starts_with("lib") {
        return None;
    }
    let crate_part = file_name.strip_prefix("lib")?;
    let (name_part, ext) = crate_part.split_once('.')?;
    let crate_name = name_part.split('-').next().unwrap_or(name_part);
    let kind = match ext {
        "so" => ArtifactKind::ProcMacro,
        "rlib" => ArtifactKind::Rlib,
        "rmeta" => ArtifactKind::Rmeta,
        _ => return None,
    };
    Some((crate_name, kind))
}

fn select_best_artifact(artifacts: &[Artifact]) -> Option<PathBuf> {
    artifacts
        .iter()
        .max_by(|a, b| match (a.kind, b.kind) {
            (ArtifactKind::ProcMacro, ArtifactKind::ProcMacro) => std::cmp::Ordering::Equal,
            (ArtifactKind::ProcMacro, _) => std::cmp::Ordering::Greater,
            (_, ArtifactKind::ProcMacro) => std::cmp::Ordering::Less,
            (ArtifactKind::Rlib, ArtifactKind::Rlib) => std::cmp::Ordering::Equal,
            (ArtifactKind::Rlib, ArtifactKind::Rmeta) => std::cmp::Ordering::Greater,
            (ArtifactKind::Rmeta, ArtifactKind::Rlib) => std::cmp::Ordering::Less,
            (ArtifactKind::Rmeta, ArtifactKind::Rmeta) => std::cmp::Ordering::Equal,
        })
        .map(|artifact| artifact.path.clone())
}

/// Raw Cargo metadata payload.
#[derive(Debug, Deserialize)]
struct CargoMetadata {
    target_directory: PathBuf,
    workspace_root: PathBuf,
    packages: Vec<CargoPackage>,
}

/// Cargo package metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct CargoPackage {
    /// Package name (e.g., crate name).
    pub name: String,
    /// Version declared in `Cargo.toml`.
    pub version: String,
    /// Declared Rust edition.
    pub edition: String,
    /// Optional `rust-version` constraint.
    pub rust_version: Option<String>,
    /// Feature definitions declared by the package.
    pub features: HashMap<String, Vec<String>>,
    /// Targets built by the package.
    pub targets: Vec<CargoTarget>,
}

/// Cargo target metadata.
#[derive(Debug, Clone, Deserialize)]
pub struct CargoTarget {
    /// Target name.
    pub name: String,
    /// Target kinds (e.g., ["lib"], ["bin"]).
    pub kind: Vec<String>,
    /// Source path for the target root.
    pub src_path: PathBuf,
}

/// Simplified project metadata derived from `cargo metadata`.
#[derive(Debug, Clone)]
pub struct ProjectMetadata {
    /// Workspace root directory.
    pub workspace_root: PathBuf,
    /// Target directory for build artifacts.
    pub target_directory: PathBuf,
    /// Packages discovered in the workspace.
    pub packages: Vec<CargoPackage>,
    /// Edition of the primary package (if any).
    pub edition: Option<String>,
    /// Rust toolchain requirement for the primary package (if any).
    pub rust_version: Option<String>,
}

impl StructuralEditOracle for CargoProject {
    fn impact_of(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }

    fn satisfies_bounds(&self, _id: &str, _new_sig: &syn::Signature) -> bool {
        true
    }

    fn is_macro_generated(&self, _symbol_id: &str) -> bool {
        false
    }

    fn cross_crate_users(&self, _symbol_id: &str) -> Vec<String> {
        Vec::new()
    }
}
