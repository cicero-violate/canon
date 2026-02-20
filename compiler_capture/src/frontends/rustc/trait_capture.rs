#![cfg(feature = "rustc_frontend")]
use super::frontend_context::FrontendMetadata;
use super::metadata_capture;
use crate::compiler_capture::graph::NodeId;
use crate::compiler_capture::graph::{DeltaCollector, NodePayload};
use crate::rename::core::symbol_id::normalize_symbol_id_with_crate;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::TyCtxt;
use serde::Serialize;
use std::collections::HashMap;
pub(super) fn capture_trait<'tcx>(builder: &mut DeltaCollector, tcx: TyCtxt<'tcx>, def_id: DefId, cache: &mut HashMap<DefId, NodeId>, metadata: &FrontendMetadata) -> NodeId {
    if let Some(id) = cache.get(&def_id) {
        return id.clone();
    }
    let trait_def = tcx.trait_def(def_id);
    let raw_def_path = tcx.def_path_str(def_id);
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));
    let node_key = format!("{def_id:?}");
    let mut payload = NodePayload::new(&node_key, def_path.clone())
        .with_metadata("type", "trait")
        .with_metadata("has_auto_impl", trait_def.has_auto_impl.to_string())
        .with_metadata("safety", format!("{:?}", trait_def.safety))
        .with_metadata("constness", format!("{:?}", trait_def.constness))
        .with_metadata("paren_sugar", trait_def.paren_sugar.to_string());
    if let Some(items) = serialize_associated_items(tcx, def_id) {
        payload = payload.with_metadata("trait_items", items);
    }
    if def_id.is_local() {
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
    }
    payload = metadata_capture::apply_common_metadata(payload, tcx, def_id, metadata);
    let node_id = builder.add_node(payload);
    cache.insert(def_id, node_id.clone());
    node_id
}
pub(super) fn capture_impl<'tcx>(builder: &mut DeltaCollector, tcx: TyCtxt<'tcx>, def_id: DefId, cache: &mut HashMap<DefId, NodeId>, metadata: &FrontendMetadata, of_trait_hint: bool) -> NodeId {
    if let Some(id) = cache.get(&def_id) {
        return id.clone();
    }
    let raw_def_path = tcx.def_path_str(def_id);
    let crate_name = tcx.crate_name(def_id.krate).to_string();
    let def_path = normalize_symbol_id_with_crate(&raw_def_path, Some(&crate_name));
    let node_key = format!("{def_id:?}");
    let def_kind = tcx.def_kind(def_id);
    let def_impl_trait = matches!(def_kind, DefKind::Impl { of_trait: true });
    let is_trait_impl = of_trait_hint || def_impl_trait;
    let mut payload = NodePayload::new(&node_key, def_path.clone()).with_metadata("type", "impl").with_metadata("of_trait", is_trait_impl.to_string());
    if is_trait_impl {
        let trait_ref = tcx.impl_trait_ref(def_id).instantiate_identity();
        payload = payload.with_metadata("impl_trait_ref", format!("{:?}", trait_ref));
        payload = payload.with_metadata("impl_polarity", format!("{:?}", tcx.impl_polarity(def_id)));
    }
    let impl_ty = tcx.type_of(def_id).instantiate_identity();
    payload =
        payload.with_metadata("impl_target", format!("{impl_ty:?}")).with_metadata("impl_for", format!("{impl_ty:?}")).with_metadata("impl_kind", if is_trait_impl { "trait" } else { "inherent" });
    if let Some(items) = serialize_impl_items(tcx, def_id) {
        payload = payload.with_metadata("impl_items", items);
    }
    if def_id.is_local() {
        payload = payload.with_metadata("crate_edition", metadata.edition.clone());
    }
    payload = metadata_capture::apply_common_metadata(payload, tcx, def_id, metadata);
    let node_id = builder.add_node(payload);
    cache.insert(def_id, node_id.clone());
    node_id
}
fn serialize_associated_items(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    #[derive(Serialize)]
    struct AssocItemCapture {
        name: String,
        def_path: String,
        kind: String,
    }
    let captures: Vec<AssocItemCapture> = tcx
        .associated_items(def_id)
        .in_definition_order()
        .map(|item| AssocItemCapture {
            name: item.ident(tcx).to_string(),
            def_path: normalize_symbol_id_with_crate(&tcx.def_path_str(item.def_id), Some(&tcx.crate_name(item.def_id.krate).to_string())),
            kind: format!("{:?}", item.kind),
        })
        .collect();
    if captures.is_empty() {
        None
    } else {
        serde_json::to_string(&captures).ok()
    }
}
fn serialize_impl_items(tcx: TyCtxt<'_>, def_id: DefId) -> Option<String> {
    let item_ids = tcx.associated_item_def_ids(def_id);
    if item_ids.is_empty() {
        return None;
    }
    #[derive(Serialize)]
    struct ImplItemCapture {
        def_path: String,
    }
    let captures: Vec<ImplItemCapture> =
        item_ids.iter().map(|item_def| ImplItemCapture { def_path: normalize_symbol_id_with_crate(&tcx.def_path_str(*item_def), Some(&tcx.crate_name(item_def.krate).to_string())) }).collect();
    serde_json::to_string(&captures).ok()
}
