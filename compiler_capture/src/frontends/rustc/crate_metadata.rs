#![cfg(feature = "rustc_frontend")]
//! Crate-level metadata capture (hashes, dependencies, etc.).
use super::frontend_context::FrontendMetadata;
use crate::compiler_capture::graph::{DeltaCollector, NodePayload};
use rustc_middle::ty::TyCtxt;
use rustc_span::def_id::LOCAL_CRATE;
use serde::Serialize;
use serde_json;
pub(super) fn capture_crate_metadata<'tcx>(
    builder: &mut DeltaCollector,
    tcx: TyCtxt<'tcx>,
    metadata: &FrontendMetadata,
) {
    let crate_name = tcx.crate_name(LOCAL_CRATE).to_string();
    let crate_hash = tcx.crate_hash(LOCAL_CRATE);
    let dependencies = serialize_crate_dependencies(tcx);
    let mut payload = NodePayload::new(
            format!("crate::{crate_name}"),
            crate_name.clone(),
        )
        .with_metadata("type", "crate")
        .with_metadata("crate", format!("crate:{crate_name}"))
        .with_metadata("crate_hash", format!("{crate_hash:?}"))
        .with_metadata("edition", metadata.edition.clone())
        .with_metadata("crate_type", metadata.crate_type.clone())
        .with_metadata("target_triple", metadata.target_triple.clone())
        .with_metadata("dependencies", dependencies);
    if let Some(name) = &metadata.target_name {
        payload = payload.with_metadata("target_name", name.clone());
    }
    if let Some(version) = &metadata.rust_version {
        payload = payload.with_metadata("rust_version", version.clone());
    }
    if let Some(package_version) = &metadata.package_version {
        payload = payload.with_metadata("crate_version", package_version.clone());
    }
    if let Some(root) = &metadata.workspace_root {
        payload = payload.with_metadata("workspace_root", root.clone());
    }
    if !metadata.package_features.is_empty() {
        if let Ok(json) = serde_json::to_string(&metadata.package_features) {
            payload = payload.with_metadata("crate_features", json);
        }
    }
    if !metadata.cfg_flags.is_empty() {
        if let Ok(json) = serde_json::to_string(&metadata.cfg_flags) {
            payload = payload.with_metadata("crate_cfg", json);
        }
    }
    let _ = builder.add_node(payload);
}
fn serialize_crate_dependencies(tcx: TyCtxt<'_>) -> String {
    #[derive(Serialize)]
    struct DependencyCapture {
        name: String,
        hash: String,
        source: String,
    }
    let deps: Vec<DependencyCapture> = tcx
        .crates(())
        .iter()
        .map(|crate_num| DependencyCapture {
            name: tcx.crate_name(*crate_num).to_string(),
            hash: format!("{:?}", tcx.crate_hash(* crate_num)),
            source: format!("{:?}", tcx.used_crate_source(* crate_num)),
        })
        .collect();
    serde_json::to_string(&deps).unwrap_or_else(|_| "[]".into())
}
