//! Cargo project helper utilities (extern discovery, build metadata).
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use crate::rename::core::{StructuralEditOracle, normalize_symbol_id};
use crate::state::graph::GraphSnapshot;
/// Handle discovery of Cargo metadata and extern artifacts for a project.
#[derive(Debug, Clone)]
pub struct CargoProject {
    root: PathBuf,
    target_dir: PathBuf,
    oracle: Option<OracleData>,
}
impl CargoProject {
    /// Attempts to load a Cargo project by walking up from the entry file.
    pub fn from_entry(entry: &Path) -> io::Result<Self> {
        let mut current = entry.canonicalize()?;
        if current.is_file() {
            current = current
                .parent()
                .ok_or_else(|| io::Error::new(
                    io::ErrorKind::NotFound,
                    "entry has no parent",
                ))?
                .to_path_buf();
        }
        let root = find_cargo_root(&current)?;
        let target_dir = fetch_target_dir(&root).unwrap_or_else(|_| root.join("target"));
        Ok(Self {
            root,
            target_dir,
            oracle: None,
        })
    }
    /// Attach a graph snapshot to enable oracle queries.
    pub fn with_snapshot(mut self, snapshot: GraphSnapshot) -> Self {
        self.oracle = Some(OracleData::from_snapshot(snapshot));
        self
    }
    /// Ensures dependencies are built (debug profile).
    pub fn ensure_dependencies_built(&self) -> io::Result<()> {
        let status = Command::new("cargo")
            .arg("build")
            .current_dir(&self.root)
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(
                io::Error::new(io::ErrorKind::Other, "cargo build failed inside project"),
            )
        }
    }
    /// Collects `--extern` and `-L dependency=` flags for the requested profile.
    pub fn extern_args(&self, profile: &str) -> io::Result<Vec<String>> {
        let deps_dir = self.target_dir.join(profile).join("deps");
        if !deps_dir.exists() {
            return Err(
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("deps dir missing: {}", deps_dir.display()),
                ),
            );
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
        let mut meta = load_cargo_metadata(&self.root)?;
        let root_manifest = self.root.join("Cargo.toml");
        let root_manifest = root_manifest.canonicalize().unwrap_or(root_manifest);
        if let Some(index) = meta
            .packages
            .iter()
            .position(|pkg| {
                pkg
                    .manifest_path
                    .canonicalize()
                    .unwrap_or_else(|_| pkg.manifest_path.clone()) == root_manifest
            })
        {
            if index != 0 {
                let pkg = meta.packages.remove(index);
                meta.packages.insert(0, pkg);
            }
        }
        let CargoMetadata { target_directory, workspace_root, packages } = meta;
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
            let targets: Vec<CargoTarget> = pkg
                .targets
                .iter()
                .filter(|t| { t.kind.iter().any(|k| k == "lib" || k == "bin") })
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
    Err(io::Error::new(io::ErrorKind::NotFound, "Cargo.toml not found in parent chain"))
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
        return Err(io::Error::new(io::ErrorKind::Other, "cargo metadata failed"));
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
        .max_by_key(|artifact| {
            let rank = match artifact.kind {
                ArtifactKind::ProcMacro => 2,
                ArtifactKind::Rlib => 1,
                ArtifactKind::Rmeta => 0,
            };
            let meta = std::fs::metadata(&artifact.path).ok();
            let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            let mtime = meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            (rank, size, mtime)
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
    /// Manifest path reported by cargo metadata.
    pub manifest_path: PathBuf,
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
#[derive(Debug, Clone)]
struct OracleData {
    adjacency: HashMap<String, Vec<String>>,
    macro_generated: HashSet<String>,
    crate_by_key: HashMap<String, String>,
    signature_by_key: HashMap<String, String>,
}
impl OracleData {
    fn from_snapshot(snapshot: GraphSnapshot) -> Self {
        let mut id_to_key = HashMap::new();
        let mut macro_generated = HashSet::new();
        let mut crate_by_key = HashMap::new();
        let mut signature_by_key = HashMap::new();
        for node in snapshot.nodes() {
            let key = node.key.to_string();
            id_to_key.insert(node.id, key.clone());
            if is_macro_generated(&node.metadata) {
                macro_generated.insert(key.clone());
            }
            if let Some(crate_name) = node
                .metadata
                .get("crate")
                .or_else(|| node.metadata.get("crate_name"))
                .or_else(|| node.metadata.get("package"))
            {
                crate_by_key.insert(key.clone(), crate_name.clone());
            }
            if let Some(signature) = node.metadata.get("signature") {
                signature_by_key.insert(key.clone(), signature.clone());
            }
        }
        let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
        for edge in snapshot.edges() {
            let from = id_to_key.get(&edge.from).cloned();
            let to = id_to_key.get(&edge.to).cloned();
            if let (Some(from), Some(to)) = (from, to) {
                adjacency.entry(from.clone()).or_default().push(to.clone());
                adjacency.entry(to).or_default().push(from);
            }
        }
        Self {
            adjacency,
            macro_generated,
            crate_by_key,
            signature_by_key,
        }
    }
}
fn is_macro_generated(metadata: &std::collections::BTreeMap<String, String>) -> bool {
    let value = metadata
        .get("macro_generated")
        .or_else(|| metadata.get("generated_by_macro"))
        .or_else(|| metadata.get("macro"))
        .or_else(|| metadata.get("is_macro"));
    matches!(value.map(| v | v.as_str()), Some("true") | Some("1") | Some("yes"))
}
impl StructuralEditOracle for CargoProject {
    fn impact_of(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        if let Some(oracle) = &self.oracle {
            if let Some(edges) = oracle.adjacency.get(&symbol_id) {
                return edges.clone();
            }
        }
        Vec::new()
    }
    fn satisfies_bounds(&self, id: &str, new_sig: &syn::Signature) -> bool {
        let id = normalize_symbol_id(id);
        if let Some(oracle) = &self.oracle {
            if let Some(sig) = oracle.signature_by_key.get(&id) {
                let new_sig = quote::quote!(# new_sig).to_string();
                return sig == &new_sig;
            }
        }
        true
    }
    fn is_macro_generated(&self, symbol_id: &str) -> bool {
        let symbol_id = normalize_symbol_id(symbol_id);
        if let Some(oracle) = &self.oracle {
            return oracle.macro_generated.contains(&symbol_id);
        }
        false
    }
    fn cross_crate_users(&self, symbol_id: &str) -> Vec<String> {
        let symbol_id = normalize_symbol_id(symbol_id);
        let Some(oracle) = &self.oracle else { return Vec::new() };
        let Some(symbol_crate) = oracle.crate_by_key.get(&symbol_id) else {
            return Vec::new();
        };
        oracle
            .adjacency
            .get(&symbol_id)
            .into_iter()
            .flatten()
            .filter_map(|neighbor| {
                let other_crate = oracle.crate_by_key.get(neighbor)?;
                if other_crate != symbol_crate { Some(neighbor.clone()) } else { None }
            })
            .collect()
    }
}
