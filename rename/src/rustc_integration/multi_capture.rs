//! Capture helpers that merge multiple cargo targets into a single snapshot.

use std::sync::Arc;

use crate::rustc_integration::frontends::rustc::{RustcFrontend, RustcFrontendError};
use crate::rustc_integration::project::CargoProject;
use crate::state::graph::{GraphDelta, GraphDeltaError, GraphMaterializer, GraphSnapshot};
use crate::state::ids::{EdgeId, NodeId};
use crate::state::workspace::{GraphWorkspace, WorkspaceBuilder};

/// Captured artifacts from a workspace build (snapshot + workspace overlay).
pub struct CaptureArtifacts {
    pub snapshot: GraphSnapshot,
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
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Generic(msg) => write!(f, "capture error: {msg}"),
            CaptureError::Io(err) => write!(f, "{err}"),
            CaptureError::Frontend(err) => write!(f, "{err}"),
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

/// Capture all targets (lib + bins) of a cargo project and merge into one snapshot.
pub fn capture_project(
    frontend: &RustcFrontend,
    project: &CargoProject,
    extra_args: &[String],
) -> Result<CaptureArtifacts, CaptureError> {
    let mut materializer = GraphMaterializer::new();
    let mut graph_deltas = Vec::new();
    let project_meta = project.metadata()?;
    let workspace_root = project_meta.workspace_root.clone();
    let primary_package = project_meta.packages.first().cloned();

    let mut externs = project
        .extern_args("debug")
        .map_err(|e| CaptureError::Generic(format!("failed to gather extern args: {e}")))?;

    let mut args = extra_args.to_vec();
    args.append(&mut externs);

    // Find OUT_DIR for build scripts
    let mut env_vars = vec![];
    if let Some(out_dir) = find_out_dir(project) {
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
        let snap = target_frontend.capture_snapshot(&target.src_path, &args, &env_vars)?;

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

        // Reinsert nodes via graph deltas
        for node in snap.nodes() {
            let target_kind = target.kind.join(",");
            let norm_key = normalize_key(node.key.as_ref());
            let mut cloned = node.clone();
            let new_id = NodeId::from_key(&norm_key);
            cloned.id = new_id;
            cloned.key = Arc::<str>::from(norm_key);
            cloned
                .metadata
                .insert("target_name".into(), target.name.clone());
            cloned
                .metadata
                .insert("target_kind".into(), target_kind.clone());
            match materializer.apply(GraphDelta::AddNode(cloned.clone())) {
                Ok(_) => {
                    graph_deltas.push(GraphDelta::AddNode(cloned));
                    id_map.insert(node.id, new_id);
                }
                Err(GraphDeltaError::NodeExists(existing)) => {
                    id_map.insert(node.id, existing);
                }
                Err(e) => return Err(CaptureError::Generic(format!("merge node failed: {e:?}"))),
            };
        }

        // Reinsert edges
        for edge in snap.edges() {
            let from = *id_map
                .get(&edge.from)
                .ok_or_else(|| CaptureError::Generic("edge.from missing in id_map".into()))?;
            let to = *id_map
                .get(&edge.to)
                .ok_or_else(|| CaptureError::Generic("edge.to missing in id_map".into()))?;
            let mut cloned = edge.clone();
            cloned.from = from;
            cloned.to = to;
            cloned.id = EdgeId::from_components(&from, &to, cloned.kind.as_str());
            match materializer.apply(GraphDelta::AddEdge(cloned.clone())) {
                Ok(_) => graph_deltas.push(GraphDelta::AddEdge(cloned)),
                Err(GraphDeltaError::EdgeExists(_)) => {}
                Err(e) => return Err(CaptureError::Generic(format!("merge edge failed: {e:?}"))),
            }
        }
    }

    let snapshot = materializer.snapshot();
    let workspace = WorkspaceBuilder::new(snapshot.hash()).finalize();
    Ok(CaptureArtifacts {
        snapshot,
        workspace,
        graph_deltas,
    })
}

fn find_out_dir(project: &CargoProject) -> Option<std::path::PathBuf> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::graph::{EdgeKind, EdgeRecord, GraphDelta, GraphMaterializer, NodeRecord};
    use crate::state::ids::{EdgeId, NodeId};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn node(id: u8) -> NodeRecord {
        NodeRecord {
            id: NodeId::from_bytes([id; 16]),
            key: Arc::<str>::from(format!("key{id}")),
            label: Arc::<str>::from(format!("label{id}")),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn duplicate_edges_are_tolerated() {
        let mut materializer = GraphMaterializer::new();
        let a = node(1);
        let b = node(2);
        materializer.apply(GraphDelta::AddNode(a.clone())).unwrap();
        materializer.apply(GraphDelta::AddNode(b.clone())).unwrap();

        let edge = EdgeRecord {
            id: EdgeId::from_components(&a.id, &b.id, "call"),
            from: a.id,
            to: b.id,
            kind: EdgeKind::Call,
            metadata: BTreeMap::new(),
        };

        assert!(materializer
            .apply(GraphDelta::AddEdge(edge.clone()))
            .is_ok());
        // second insert should be ignored
        assert!(materializer.apply(GraphDelta::AddEdge(edge)).is_err());
    }
}
