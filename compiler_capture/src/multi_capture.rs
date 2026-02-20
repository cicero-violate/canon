//! Capture helpers that merge multiple cargo targets into a single delta stream.

use crate::compiler_capture::frontends::rustc::{RustcFrontend, RustcFrontendError};
use crate::compiler_capture::graph::{GraphDelta, NodeId};
use crate::compiler_capture::project::CargoProject;
use crate::compiler_capture::workspace::{GraphWorkspace, WorkspaceBuilder};
use database::graph_log::WireEdgeId;
use database::{MemoryEngine, MemoryEngineConfig, MemoryEngineError};

/// Captured artifacts from a workspace build (delta stream + workspace overlay).
pub struct CaptureArtifacts {
    pub workspace: GraphWorkspace,
    pub graph_deltas: Vec<GraphDelta>,
}

/// Errors that may arise when merging multiple targets into a single snapshot.
#[derive(Debug)]
pub enum CaptureError {
    /// Generic orchestration failure.
    Generic(String),
    /// I/O error while talking to cargo/metadata.
    Io(std::io::Error),
    /// Rustc frontend failure.
    Frontend(RustcFrontendError),
    /// Memory engine failure.
    Engine(MemoryEngineError),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Generic(msg) => write!(f, "capture error: {msg}"),
            CaptureError::Io(err) => write!(f, "{err}"),
            CaptureError::Frontend(err) => write!(f, "{err}"),
            CaptureError::Engine(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for CaptureError {}

impl From<std::io::Error> for CaptureError {
    fn from(err: std::io::Error) -> Self {
        CaptureError::Io(err)
    }
}

impl From<RustcFrontendError> for CaptureError {
    fn from(err: RustcFrontendError) -> Self {
        CaptureError::Frontend(err)
    }
}

impl From<MemoryEngineError> for CaptureError {
    fn from(err: MemoryEngineError) -> Self {
        CaptureError::Engine(err)
    }
}

/// Capture all targets (lib + bins) of a cargo project and merge into one delta stream.
pub fn capture_project(
    frontend: &RustcFrontend,
    project: &CargoProject,
    extra_args: &[String],
) -> Result<CaptureArtifacts, CaptureError> {
    let mut graph_deltas = Vec::new();
    let project_meta = project.metadata()?;
    let workspace_root = project_meta.workspace_root.clone();
    let primary_package = project_meta.packages.first().cloned();
    let engine = open_engine(&workspace_root)?;

    let mut externs = project
        .extern_args("debug")
        .map_err(|e| CaptureError::Generic(format!("failed to gather extern args: {e}")))?;

    let mut args = extra_args.to_vec();
    args.append(&mut externs);

    // Find OUT_DIR for build scripts
    let mut env_vars = vec![];
    if let Some(out_dir) = find_build_out_dir(project) {
        env_vars.push(("OUT_DIR".to_string(), out_dir.display().to_string()));
        eprintln!("Using OUT_DIR: {}", out_dir.display());
    }

    for target in project.targets()? {
        let mut target_frontend = frontend.clone().with_target_name(&target.name);
        target_frontend = target_frontend.with_workspace_root(workspace_root.clone());
        if let Some(pkg) = &primary_package {
            target_frontend = target_frontend
                .with_package_info(pkg.name.clone(), pkg.version.clone())
                .with_edition(pkg.edition.clone())
                .with_package_features(pkg.features.keys().cloned());
            if let Some(rust_version) = &pkg.rust_version {
                target_frontend = target_frontend.with_rust_version(rust_version.clone());
            }
        }
        let deltas = target_frontend.capture_deltas(&target.src_path, &args, &env_vars)?;

        use std::collections::HashMap;
        let mut id_map = HashMap::new();

        // Strip rustc crate hash: foo[abcd]::bar -> foo::bar
        fn normalize_key(key: &str) -> String {
            let mut out = String::with_capacity(key.len());
            let mut in_brackets = false;
            for c in key.chars() {
                match c {
                    '[' => in_brackets = true,
                    ']' => in_brackets = false,
                    _ if !in_brackets => out.push(c),
                    _ => {}
                }
            }
            out
        }

        for delta in deltas {
            match delta {
                GraphDelta::AddNode(mut node) => {
                    let target_kind = target.kind.join(",");
                    let norm_key = normalize_key(node.key.as_str());
                    let new_id = NodeId::from_key(&norm_key);
                    id_map.insert(node.id.clone(), new_id.clone());
                    node.id = new_id;
                    node.key = norm_key;
                    node.metadata
                        .insert("target_name".into(), target.name.clone());
                    node.metadata.insert("target_kind".into(), target_kind);
                    graph_deltas.push(GraphDelta::AddNode(node.clone()));
                    engine
                        .commit_graph_delta(graph_deltas.last().unwrap().clone())
                        .map_err(CaptureError::Engine)?;
                }
                GraphDelta::AddEdge(mut edge) => {
                    let from = id_map.get(&edge.from).ok_or_else(|| {
                        CaptureError::Generic("edge.from missing in id_map".into())
                    })?;
                    let to = id_map
                        .get(&edge.to)
                        .ok_or_else(|| CaptureError::Generic("edge.to missing in id_map".into()))?;
                    edge.from = from.clone();
                    edge.to = to.clone();
                    edge.id = WireEdgeId::from_components(from, to, edge.kind.as_str());
                    graph_deltas.push(GraphDelta::AddEdge(edge.clone()));
                    engine
                        .commit_graph_delta(graph_deltas.last().unwrap().clone())
                        .map_err(CaptureError::Engine)?;
                }
            }
        }
    }

    let workspace = WorkspaceBuilder::new(engine.graph_delta_count()).finalize();
    Ok(CaptureArtifacts {
        workspace,
        graph_deltas,
    })
}

fn open_engine(root: &std::path::Path) -> Result<MemoryEngine, CaptureError> {
    let state_dir = root.join(".rename");
    std::fs::create_dir_all(&state_dir)?;
    let tlog_path = state_dir.join("state.tlog");
    Ok(MemoryEngine::new(MemoryEngineConfig { tlog_path })?)
}

fn find_build_out_dir(project: &CargoProject) -> Option<std::path::PathBuf> {
    let build_dir = project.workspace_root().join("target/debug/build");

    // Try to find the package-specific build output directory
    // The directory name format is: {package_name}-{hash}
    // We need to extract the package name from the project

    if let Ok(entries) = std::fs::read_dir(&build_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let dir_name = match file_name.to_str() {
                Some(name) => name.to_string(),
                None => continue,
            };

            // Skip if this is clearly not the right package
            // (e.g., proc-macro2, serde, etc.)
            if dir_name.starts_with("proc-macro")
                || dir_name.starts_with("serde")
                || dir_name.starts_with("tokio")
            {
                continue;
            }

            let out_path = entry.path().join("out");
            if out_path.exists() && out_path.join("ext.rs").exists() {
                // Found a build output with ext.rs - this is likely the right one
                return Some(out_path);
            }
        }
    }
    None
}

// tests removed: compiler_capture no longer materializes graphs
