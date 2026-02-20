#![cfg(feature = "rustc_frontend")]
use super::frontend_context::FrontendMetadata;
use super::hir_dump;
use super::metadata_capture;
use crate::compiler_capture::graph::{DeltaCollector, NodeId, NodePayload};
use crate::rename::core::symbol_id::normalize_symbol_id_with_crate;
use rustc_hir::def::DefKind;
use rustc_middle::{mir, ty::TyCtxt};
use rustc_span::def_id::{DefId, LocalDefId};
use serde::Serialize;
use std::collections::HashMap;
pub(super) fn ensure_node<'tcx>(
    builder: &mut DeltaCollector,
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    cache: &mut HashMap<DefId, NodeId>,
    metadata: &FrontendMetadata,
) -> NodeId {
    if let Some(id) = cache.get(&def_id) {
        return id.clone();
    }
    let raw_def_path = tcx.def_path_str(def_id);
    let node_key = format!("{def_id:?}");
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));
    let def_kind = format!("{:?}", tcx.def_kind(def_id));
    let span = tcx.def_span(def_id);
    let source_map = tcx.sess.source_map();
    let loc = source_map.lookup_char_pos(span.lo());
    let source_file = loc.file.name.prefer_local_unconditionally().to_string();
    let signature = matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn)
        .then(|| tcx.fn_sig(def_id).skip_binder().to_string());
    let mut payload = NodePayload::new(&node_key, def_path.clone())
        .with_metadata("crate", crate_name)
        .with_metadata("def_kind", def_kind)
        .with_metadata("local", def_id.is_local().to_string())
        .with_metadata("source_file", source_file)
        .with_metadata("line", loc.line.to_string())
        .with_metadata("column", loc.col.0.to_string())
        .with_metadata("def_path", def_path.clone());
    if let Some(sig) = signature {
        payload = payload.with_metadata("signature", sig);
    }
    if let Some(sig_json) = serialize_fn_signature(tcx, def_id) {
        payload = payload.with_metadata("fn_signature", sig_json);
    }
    payload = metadata_capture::apply_common_metadata(payload, tcx, def_id, metadata);
    if let Some(local_def) = def_id.as_local() {
        if tcx.is_mir_available(def_id) {
            if let Some(locals_json) = serialize_locals(tcx, local_def) {
                payload = payload.with_metadata("locals", locals_json);
            }
        }
        if let Some(hir_body) = hir_dump::encode_hir_body_json(tcx, local_def) {
            payload = payload.with_metadata("hir_body", hir_body);
        }
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
        payload = payload.with_metadata("crate_type", metadata.crate_type.clone());
        payload = payload.with_metadata("target_triple", metadata.target_triple.clone());
        if let Some(name) = &metadata.target_name {
            payload = payload.with_metadata("crate_target_name", name.clone());
        }
        if let Some(rust_version) = &metadata.rust_version {
            payload = payload.with_metadata("crate_rust_version", rust_version.clone());
        }
    }
    let node_id = builder.add_node(payload);
    cache.insert(def_id, node_id.clone());
    node_id
}
fn serialize_locals<'tcx>(tcx: TyCtxt<'tcx>, local_def: LocalDefId) -> Option<String> {
    let body = tcx.optimized_mir(local_def);
    let mut debug_names: HashMap<mir::Local, String> = HashMap::new();
    for info in &body.var_debug_info {
        if let mir::VarDebugInfoContents::Place(place) = info.value {
            debug_names.insert(place.local, info.name.to_string());
        }
    }
    #[derive(Serialize)]
    struct LocalMetadata {
        index: usize,
        debug_name: Option<String>,
        ty: String,
    }
    let locals: Vec<LocalMetadata> = body
        .local_decls
        .iter_enumerated()
        .map(|(local, decl)| LocalMetadata {
            index: local.as_usize(),
            debug_name: debug_names.get(&local).cloned(),
            ty: format!("{:?}", decl.ty),
        })
        .collect();
    serde_json::to_string(&locals).ok()
}
fn serialize_fn_signature<'tcx>(tcx: TyCtxt<'tcx>, def_id: DefId) -> Option<String> {
    if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn) {
        return None;
    }
    #[derive(Serialize)]
    struct FnSigCapture {
        inputs: Vec<String>,
        output: String,
        abi: String,
        safety: String,
    }
    let sig = tcx.fn_sig(def_id);
    let sig = sig.skip_binder();
    let inputs = sig.inputs().iter().map(|ty| format!("{:?}", ty)).collect();
    let capture = FnSigCapture {
        inputs,
        output: format!("{:?}", sig.output()),
        abi: format!("{:?}", sig.abi()),
        safety: format!("{:?}", sig.safety()),
    };
    serde_json::to_string(&capture).ok()
}
